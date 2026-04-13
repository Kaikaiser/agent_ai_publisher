from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Iterable
from uuid import uuid4

from langchain_core.prompts import ChatPromptTemplate
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.db.models import ConversationRecord, ConversationSession, MemoryItem

try:
    from redis import Redis
except ImportError:  # pragma: no cover - exercised when redis extra is unavailable.
    Redis = None


logger = logging.getLogger(__name__)


@dataclass
class MemorySnippet:
    id: str
    scope: str
    memory_type: str
    summary: str
    content: str


@dataclass
class MemoryRecordView:
    id: str
    session_id: int | None
    book_id: int | None
    project_id: int | None
    scope: str
    memory_type: str
    summary: str
    content: str
    salience_score: float
    confidence_score: float
    source_conversation_id: int | None
    created_at: datetime | None
    updated_at: datetime | None


@dataclass
class RankedMemory:
    score: float
    snippet: MemorySnippet


class ConversationSessionService:
    def __init__(self, db: Session) -> None:
        self.db = db

    def ensure_session(
        self,
        username: str,
        session_id: int | None,
        question: str,
        book_id: int | None = None,
        project_id: int | None = None,
    ) -> ConversationSession:
        if session_id is not None:
            session = (
                self.db.query(ConversationSession)
                .filter(ConversationSession.id == session_id, ConversationSession.username == username)
                .first()
            )
            if session is None:
                raise ValueError("Conversation session not found.")
            if not session.title:
                session.title = self._build_title(question)
            if book_id is not None and session.book_id is None:
                session.book_id = book_id
            if project_id is not None and session.project_id is None:
                session.project_id = project_id
            session.updated_at = datetime.now(timezone.utc)
            return session

        session = ConversationSession(
            username=username,
            book_id=book_id,
            project_id=project_id,
            title=self._build_title(question),
        )
        self.db.add(session)
        self.db.flush()
        return session

    def build_effective_question(self, session: ConversationSession, question: str) -> tuple[str, bool]:
        if not session.pending_clarification or not session.pending_question:
            return question, False
        slot = session.pending_clarification_slot or "missing_context"
        merged_question = (
            f"Original question:\n{session.pending_question}\n\n"
            f"Clarification slot:\n{slot}\n\n"
            f"User clarification:\n{question}"
        )
        session.pending_clarification = False
        session.pending_question = ""
        session.pending_clarification_slot = ""
        session.pending_prompt = ""
        session.updated_at = datetime.now(timezone.utc)
        return merged_question, True

    def mark_pending_clarification(
        self,
        session: ConversationSession,
        *,
        original_question: str,
        clarification_slot: str,
        clarification_prompt: str,
    ) -> None:
        session.pending_clarification = True
        session.pending_question = original_question
        session.pending_clarification_slot = clarification_slot
        session.pending_prompt = clarification_prompt
        session.updated_at = datetime.now(timezone.utc)

    def clear_pending_clarification(self, session: ConversationSession) -> None:
        if not session.pending_clarification and not session.pending_question and not session.pending_prompt:
            return
        session.pending_clarification = False
        session.pending_question = ""
        session.pending_clarification_slot = ""
        session.pending_prompt = ""
        session.updated_at = datetime.now(timezone.utc)

    @staticmethod
    def _build_title(question: str) -> str:
        normalized = " ".join(question.split())
        return normalized[:60] if normalized else "New conversation"


class RedisSessionMemoryStore:
    def __init__(self) -> None:
        self.settings = get_settings()
        self._client = None
        self._disabled = False

    def is_enabled(self) -> bool:
        return bool(self.settings.redis_url and not self._disabled and Redis is not None)

    def list_memories(
        self,
        username: str,
        book_id: int | None = None,
        session_id: int | None = None,
    ) -> list[MemoryRecordView]:
        if not self.is_enabled() or (session_id is not None and book_id is None):
            return []

        keys = [self._key(username, book_id, session_id)] if session_id is not None else self._scan_keys(username, book_id)
        records: list[MemoryRecordView] = []
        for key in keys:
            for payload in self._load_payloads(key):
                records.append(self._view_from_payload(payload))

        records.sort(
            key=lambda item: item.updated_at or datetime.min.replace(tzinfo=timezone.utc),
            reverse=True,
        )
        return records

    def save_memories(
        self,
        username: str,
        book_id: int,
        session_id: int,
        project_id: int | None,
        candidates: list[dict],
        conversation_id: int,
        embeddings,
    ) -> None:
        if not self.is_enabled() or not candidates:
            return

        key = self._key(username, book_id, session_id)
        existing = self._load_payloads(key)

        for candidate in candidates:
            payload = self._build_payload(
                username=username,
                book_id=book_id,
                session_id=session_id,
                project_id=project_id,
                conversation_id=conversation_id,
                candidate=candidate,
                embeddings=embeddings,
            )
            index = next(
                (
                    idx
                    for idx, item in enumerate(existing)
                    if item["memory_type"] == payload["memory_type"] and item["summary"] == payload["summary"]
                ),
                None,
            )
            if index is None:
                existing.append(payload)
                continue

            current = existing[index]
            current["content"] = payload["content"]
            current["summary"] = payload["summary"]
            current["book_id"] = payload["book_id"]
            current["project_id"] = payload["project_id"]
            current["source_conversation_id"] = conversation_id
            current["salience_score"] = max(current["salience_score"], payload["salience_score"])
            current["confidence_score"] = max(current["confidence_score"], payload["confidence_score"])
            current["embedding"] = payload["embedding"]
            current["updated_at"] = payload["updated_at"]

        existing.sort(key=lambda item: item["updated_at"], reverse=True)
        trimmed = existing[: self.settings.session_memory_limit]
        self._write_payloads(key, trimmed)

    def search(
        self,
        username: str,
        book_id: int | None,
        session_id: int | None,
        query_embedding: Iterable[float],
        limit: int,
    ) -> list[RankedMemory]:
        if not self.is_enabled() or session_id is None or book_id is None:
            return []

        ranked: list[RankedMemory] = []
        for payload in self._load_payloads(self._key(username, book_id, session_id)):
            score = self._memory_score(payload, query_embedding)
            if score <= 0:
                continue
            ranked.append(
                RankedMemory(
                    score=score,
                    snippet=MemorySnippet(
                        id=payload["id"],
                        scope=payload["scope"],
                        memory_type=payload["memory_type"],
                        summary=payload["summary"],
                        content=payload["content"],
                    ),
                )
            )

        ranked.sort(key=lambda item: item.score, reverse=True)
        return ranked[:limit]

    def delete(self, username: str, memory_id: str) -> bool:
        if not self.is_enabled() or not memory_id.startswith("redis|"):
            return False

        _, memory_username, book_id_text, session_id_text, _ = memory_id.split("|", 4)
        if memory_username != username:
            return False

        key = self._key(username, int(book_id_text), int(session_id_text))
        payloads = self._load_payloads(key)
        remaining = [item for item in payloads if item["id"] != memory_id]
        if len(remaining) == len(payloads):
            return False
        self._write_payloads(key, remaining)
        return True

    def _build_payload(
        self,
        username: str,
        book_id: int,
        session_id: int,
        project_id: int | None,
        conversation_id: int,
        candidate: dict,
        embeddings,
    ) -> dict:
        now = datetime.now(timezone.utc).isoformat()
        return {
            "id": f"redis|{username}|{book_id}|{session_id}|{uuid4().hex}",
            "session_id": session_id,
            "book_id": book_id,
            "project_id": project_id,
            "scope": "session",
            "memory_type": candidate["memory_type"],
            "summary": candidate["summary"],
            "content": candidate["content"],
            "salience_score": candidate["salience_score"],
            "confidence_score": candidate["confidence_score"],
            "source_conversation_id": conversation_id,
            "embedding": [float(value) for value in embeddings.embed_query(candidate["content"])],
            "created_at": now,
            "updated_at": now,
        }

    def _memory_score(self, payload: dict, query_embedding: Iterable[float]) -> float:
        return cosine_similarity(query_embedding, payload.get("embedding", [])) + (payload["salience_score"] * 0.2) + 0.1

    def _view_from_payload(self, payload: dict) -> MemoryRecordView:
        return MemoryRecordView(
            id=payload["id"],
            session_id=payload.get("session_id"),
            book_id=payload.get("book_id"),
            project_id=payload.get("project_id"),
            scope=payload["scope"],
            memory_type=payload["memory_type"],
            summary=payload["summary"],
            content=payload["content"],
            salience_score=float(payload["salience_score"]),
            confidence_score=float(payload["confidence_score"]),
            source_conversation_id=payload.get("source_conversation_id"),
            created_at=_parse_datetime(payload.get("created_at")),
            updated_at=_parse_datetime(payload.get("updated_at")),
        )

    def _load_payloads(self, key: str) -> list[dict]:
        client = self._get_client()
        if client is None:
            return []
        try:
            raw = client.get(key)
        except Exception as exc:  # pragma: no cover - depends on external redis runtime.
            logger.warning("Redis session memory read failed, disabling Redis memory: %s", exc)
            self._disabled = True
            return []
        if not raw:
            return []
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8")
        return json.loads(raw)

    def _write_payloads(self, key: str, payloads: list[dict]) -> None:
        client = self._get_client()
        if client is None:
            return
        try:
            client.setex(
                key,
                self.settings.redis_session_memory_ttl_minutes * 60,
                json.dumps(payloads, ensure_ascii=False),
            )
        except Exception as exc:  # pragma: no cover - depends on external redis runtime.
            logger.warning("Redis session memory write failed, disabling Redis memory: %s", exc)
            self._disabled = True

    def _scan_keys(self, username: str, book_id: int | None) -> list[str]:
        client = self._get_client()
        if client is None:
            return []
        pattern = f"session_memory:{username}:{book_id}:*" if book_id is not None else f"session_memory:{username}:*"
        try:
            keys = client.keys(pattern)
        except Exception as exc:  # pragma: no cover - depends on external redis runtime.
            logger.warning("Redis session memory key scan failed, disabling Redis memory: %s", exc)
            self._disabled = True
            return []
        return [key.decode("utf-8") if isinstance(key, bytes) else str(key) for key in keys]

    def _get_client(self):
        if not self.is_enabled():
            return None
        if self._client is not None:
            return self._client
        try:
            self._client = Redis.from_url(self.settings.redis_url, decode_responses=False)
            self._client.ping()
            return self._client
        except Exception as exc:  # pragma: no cover - depends on external redis runtime.
            logger.warning("Redis is unavailable, session memory falls back to PostgreSQL: %s", exc)
            self._disabled = True
            return None

    @staticmethod
    def _key(username: str, book_id: int | None, session_id: int | None) -> str:
        if session_id is None or book_id is None:
            raise ValueError("book_id and session_id are required for Redis session memory.")
        return f"session_memory:{username}:{book_id}:{session_id}"


class MemoryService:
    def __init__(self, db: Session, embeddings=None, llm=None) -> None:
        self.db = db
        self.embeddings = embeddings
        self.llm = llm
        self.settings = get_settings()
        self.redis_store = RedisSessionMemoryStore()

    def list_for_user(
        self,
        username: str,
        scope: str | None = None,
        session_id: int | None = None,
        book_id: int | None = None,
        project_id: int | None = None,
    ) -> list[MemoryRecordView]:
        items: list[MemoryRecordView] = []

        if scope in (None, "user"):
            items.extend(self._list_pg_memories(username=username, scope="user"))

        if scope in (None, "book"):
            items.extend(self._list_pg_memories(username=username, scope="book", book_id=book_id))

        if scope in (None, "project"):
            items.extend(self._list_pg_memories(username=username, scope="project", project_id=project_id))

        if scope in (None, "session"):
            redis_items = self.redis_store.list_memories(username=username, book_id=book_id, session_id=session_id)
            if redis_items:
                items.extend(redis_items)
            items.extend(
                self._list_pg_memories(
                    username=username,
                    scope="session",
                    session_id=session_id,
                    book_id=book_id,
                    memory_types=["session_summary"],
                )
            )
            if not redis_items:
                items.extend(
                    self._list_pg_memories(
                        username=username,
                        scope="session",
                        session_id=session_id,
                        book_id=book_id,
                        exclude_memory_types=["session_summary"],
                    )
                )

        items.sort(
            key=lambda item: (
                item.updated_at or datetime.min.replace(tzinfo=timezone.utc),
                item.created_at or datetime.min.replace(tzinfo=timezone.utc),
            ),
            reverse=True,
        )
        return items

    def delete_for_user(self, username: str, memory_id: str | int) -> None:
        memory_key = str(memory_id)
        if memory_key.startswith("redis|"):
            deleted = self.redis_store.delete(username=username, memory_id=memory_key)
            if not deleted:
                raise ValueError("Memory not found.")
            return

        if not memory_key.isdigit():
            raise ValueError("Memory not found.")

        memory = self.db.query(MemoryItem).filter(MemoryItem.id == int(memory_key)).first()
        if memory is None:
            raise ValueError("Memory not found.")
        if memory.scope not in {"book", "project"} and memory.username != username:
            raise ValueError("Memory not found.")

        self.db.delete(memory)
        self.db.commit()

    def search(
        self,
        username: str,
        session_id: int | None,
        book_id: int | None,
        project_id: int | None,
        query: str,
        limit: int | None = None,
    ) -> list[MemorySnippet]:
        self._require_embeddings()
        top_k = limit or self.settings.memory_top_k
        query_embedding = self.embeddings.embed_query(query)

        ranked: list[RankedMemory] = []
        ranked.extend(
            self.redis_store.search(
                username=username,
                book_id=book_id,
                session_id=session_id,
                query_embedding=query_embedding,
                limit=top_k,
            )
        )
        ranked.extend(
            self._search_pg_memories(
                username=username,
                scope="session",
                session_id=session_id,
                book_id=book_id,
                project_id=project_id,
                query_embedding=query_embedding,
                memory_types=["session_summary"],
            )
        )
        if not ranked:
            ranked.extend(
                self._search_pg_memories(
                    username=username,
                    scope="session",
                    session_id=session_id,
                    book_id=book_id,
                    project_id=project_id,
                    query_embedding=query_embedding,
                )
            )

        ranked.extend(
            self._search_pg_memories(
                username=username,
                scope="book",
                session_id=session_id,
                book_id=book_id,
                project_id=project_id,
                query_embedding=query_embedding,
            )
        )
        ranked.extend(
            self._search_pg_memories(
                username=username,
                scope="project",
                session_id=session_id,
                book_id=book_id,
                project_id=project_id,
                query_embedding=query_embedding,
            )
        )
        ranked.extend(
            self._search_pg_memories(
                username=username,
                scope="user",
                session_id=session_id,
                book_id=book_id,
                project_id=project_id,
                query_embedding=query_embedding,
            )
        )

        ranked.sort(key=lambda item: item.score, reverse=True)
        deduped: list[MemorySnippet] = []
        seen: set[str] = set()
        for item in ranked:
            if item.snippet.id in seen:
                continue
            seen.add(item.snippet.id)
            deduped.append(item.snippet)
            if len(deduped) >= top_k:
                break
        return deduped

    def extract_and_upsert(
        self,
        username: str,
        session_id: int,
        book_id: int | None,
        project_id: int | None,
        conversation_id: int,
        question: str,
        answer: str,
        grounded: bool,
        book_title: str | None,
        doc_type: str | None,
        decision_mode: str | None = None,
    ) -> list[MemoryRecordView]:
        self._require_embeddings()
        candidates = self._build_candidates(
            question=question,
            answer=answer,
            grounded=grounded,
            book_title=book_title,
            doc_type=doc_type,
            decision_mode=decision_mode,
        )
        session_candidates = [item for item in candidates if item["scope"] == "session"]
        book_candidates = [item for item in candidates if item["scope"] == "book"]
        project_candidates = [item for item in candidates if item["scope"] == "project"]
        user_candidates = [item for item in candidates if item["scope"] == "user"]

        if session_candidates and book_id is not None:
            if self.redis_store.is_enabled():
                self.redis_store.save_memories(
                    username=username,
                    book_id=book_id,
                    session_id=session_id,
                    project_id=project_id,
                    candidates=session_candidates,
                    conversation_id=conversation_id,
                    embeddings=self.embeddings,
                )
            else:
                for candidate in session_candidates:
                    self._upsert_pg_candidate(
                        username=username,
                        session_id=session_id,
                        book_id=book_id,
                        project_id=project_id,
                        conversation_id=conversation_id,
                        candidate=candidate,
                    )

        views: list[MemoryRecordView] = []
        for candidate in book_candidates + project_candidates + user_candidates:
            memory = self._upsert_pg_candidate(
                username=username,
                session_id=session_id,
                book_id=book_id,
                project_id=project_id,
                conversation_id=conversation_id,
                candidate=candidate,
            )
            views.append(self._view_from_model(memory))

        session_summary = self._refresh_session_summary(
            username=username,
            session_id=session_id,
            book_id=book_id,
            project_id=project_id,
            conversation_id=conversation_id,
        )
        if session_summary is not None:
            views.append(self._view_from_model(session_summary))

        self.db.flush()
        return views

    def _list_pg_memories(
        self,
        username: str,
        scope: str,
        session_id: int | None = None,
        book_id: int | None = None,
        project_id: int | None = None,
        memory_types: list[str] | None = None,
        exclude_memory_types: list[str] | None = None,
    ) -> list[MemoryRecordView]:
        query = (
            self.db.query(MemoryItem)
            .filter(MemoryItem.scope == scope, MemoryItem.status == "active")
            .order_by(MemoryItem.updated_at.desc(), MemoryItem.id.desc())
        )
        if memory_types:
            query = query.filter(MemoryItem.memory_type.in_(memory_types))
        if exclude_memory_types:
            query = query.filter(~MemoryItem.memory_type.in_(exclude_memory_types))

        if scope == "user":
            query = query.filter(MemoryItem.username == username)
        elif scope == "session":
            if session_id is None or book_id is None:
                return []
            query = query.filter(
                MemoryItem.username == username,
                MemoryItem.session_id == session_id,
                MemoryItem.book_id == book_id,
            )
        elif scope == "book":
            if book_id is None:
                return []
            query = query.filter(MemoryItem.book_id == book_id)
        elif scope == "project":
            if project_id is None:
                return []
            query = query.filter(MemoryItem.project_id == project_id)

        return [self._view_from_model(item) for item in query.all()]

    def _search_pg_memories(
        self,
        username: str,
        scope: str,
        session_id: int | None,
        book_id: int | None,
        project_id: int | None,
        query_embedding: Iterable[float],
        memory_types: list[str] | None = None,
    ) -> list[RankedMemory]:
        rows = self.db.query(MemoryItem).filter(MemoryItem.scope == scope, MemoryItem.status == "active").all()
        if memory_types:
            rows = [item for item in rows if item.memory_type in memory_types]

        if scope == "session":
            rows = [
                item
                for item in rows
                if item.username == username and item.session_id == session_id and item.book_id == book_id
            ]
        elif scope == "book":
            rows = [item for item in rows if item.book_id == book_id] if book_id is not None else []
        elif scope == "user":
            rows = [item for item in rows if item.username == username]
        elif scope == "project":
            rows = [item for item in rows if item.project_id == project_id] if project_id is not None else []

        ranked: list[RankedMemory] = []
        for item in rows:
            score = self._memory_score(item, query_embedding)
            if score <= 0:
                continue
            ranked.append(
                RankedMemory(
                    score=score,
                    snippet=MemorySnippet(
                        id=str(item.id),
                        scope=item.scope,
                        memory_type=item.memory_type,
                        summary=item.summary,
                        content=item.content,
                    ),
                )
            )
        return ranked

    def _build_candidates(
        self,
        question: str,
        answer: str,
        grounded: bool,
        book_title: str | None,
        doc_type: str | None,
        decision_mode: str | None,
    ) -> list[dict]:
        candidates: list[dict] = []
        normalized_question = " ".join(question.split())
        normalized_answer = " ".join(answer.split())

        if book_title or doc_type:
            focus_parts = []
            if book_title:
                focus_parts.append(f"book_title={book_title}")
            if doc_type:
                focus_parts.append(f"doc_type={doc_type}")
            focus_content = "Current conversation focus: " + ", ".join(focus_parts)
            candidates.append(
                {
                    "scope": "session",
                    "memory_type": "context",
                    "summary": focus_content[:80],
                    "content": focus_content,
                    "salience_score": 0.65,
                    "confidence_score": 0.95,
                }
            )
            candidates.append(
                {
                    "scope": "book",
                    "memory_type": "context",
                    "summary": f"Book focus: {focus_content[:64]}",
                    "content": focus_content,
                    "salience_score": 0.55,
                    "confidence_score": 0.85,
                }
            )
            candidates.append(
                {
                    "scope": "project",
                    "memory_type": "context",
                    "summary": f"Project focus: {focus_content[:64]}",
                    "content": focus_content,
                    "salience_score": 0.5,
                    "confidence_score": 0.8,
                }
            )

        if decision_mode:
            candidates.append(
                {
                    "scope": "book",
                    "memory_type": "rule",
                    "summary": f"Decision mode: {decision_mode}",
                    "content": f"This book uses the decision mode '{decision_mode}'.",
                    "salience_score": 0.7,
                    "confidence_score": 0.95,
                }
            )

        if self._looks_like_preference(normalized_question):
            candidates.append(
                {
                    "scope": "user",
                    "memory_type": "preference",
                    "summary": f"User preference: {normalized_question[:72]}",
                    "content": normalized_question,
                    "salience_score": 0.8,
                    "confidence_score": 0.85,
                }
            )

        if self._looks_like_task(normalized_question):
            candidates.append(
                {
                    "scope": "user",
                    "memory_type": "task",
                    "summary": f"User task: {normalized_question[:72]}",
                    "content": normalized_question,
                    "salience_score": 0.75,
                    "confidence_score": 0.8,
                }
            )

        if book_rule := self._extract_book_rule(normalized_question, book_title):
            candidates.append(
                {
                    "scope": "book",
                    "memory_type": "rule",
                    "summary": f"Book rule: {book_rule[:72]}",
                    "content": book_rule,
                    "salience_score": 0.85,
                    "confidence_score": 0.8,
                }
            )

        if project_rule := self._extract_project_rule(normalized_question, book_title):
            candidates.append(
                {
                    "scope": "project",
                    "memory_type": "rule",
                    "summary": f"Project rule: {project_rule[:72]}",
                    "content": project_rule,
                    "salience_score": 0.85,
                    "confidence_score": 0.8,
                }
            )

        if grounded and normalized_answer:
            candidates.append(
                {
                    "scope": "session",
                    "memory_type": "context",
                    "summary": f"Answered topic: {normalized_question[:64]}",
                    "content": f"The current session already discussed: {normalized_question}. Answer summary: {normalized_answer[:120]}",
                    "salience_score": 0.45,
                    "confidence_score": 0.7,
                }
            )

        return candidates

    def _refresh_session_summary(
        self,
        *,
        username: str,
        session_id: int,
        book_id: int | None,
        project_id: int | None,
        conversation_id: int,
    ) -> MemoryItem | None:
        if self.llm is None or book_id is None:
            return None

        conversation_query = (
            self.db.query(ConversationRecord)
            .filter(
                ConversationRecord.username == username,
                ConversationRecord.session_id == session_id,
                ConversationRecord.book_id == book_id,
            )
            .order_by(ConversationRecord.id.desc())
        )
        conversation_count = conversation_query.count()
        if conversation_count < self.settings.session_summary_trigger_count:
            return None

        existing_summary = (
            self.db.query(MemoryItem)
            .filter(
                MemoryItem.scope == "session",
                MemoryItem.memory_type == "session_summary",
                MemoryItem.username == username,
                MemoryItem.session_id == session_id,
                MemoryItem.book_id == book_id,
                MemoryItem.status == "active",
            )
            .order_by(MemoryItem.updated_at.desc(), MemoryItem.id.desc())
            .first()
        )
        if (
            existing_summary is not None
            and existing_summary.source_conversation_id is not None
            and conversation_id - existing_summary.source_conversation_id < self.settings.session_summary_refresh_interval
        ):
            return None

        recent_records = (
            conversation_query.limit(self.settings.session_summary_recent_turns).all()
        )
        recent_records.reverse()
        summary_text = self._generate_session_summary(existing_summary, recent_records)
        if not summary_text:
            return None

        return self._upsert_pg_candidate(
            username=username,
            session_id=session_id,
            book_id=book_id,
            project_id=project_id,
            conversation_id=conversation_id,
            candidate={
                "scope": "session",
                "memory_type": "session_summary",
                "summary": "Session summary",
                "content": summary_text,
                "salience_score": 0.9,
                "confidence_score": 0.85,
            },
        )

    def _generate_session_summary(
        self,
        existing_summary: MemoryItem | None,
        recent_records: list[ConversationRecord],
    ) -> str:
        if self.llm is None or not recent_records:
            return ""

        history_lines = []
        for item in recent_records:
            history_lines.append(f"User: {' '.join(item.question.split())}")
            history_lines.append(f"Assistant: {' '.join(item.answer.split())}")
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    (
                        "You compress conversation memory for a publishing-domain assistant. "
                        "Summarize only durable session context: user goal, confirmed constraints, preferences, "
                        "resolved subproblems, unresolved questions, and current work state. "
                        "Do not restate long factual book content. Retrieved knowledge remains the factual source of truth. "
                        "Write one concise paragraph under 220 words."
                    ),
                ),
                (
                    "human",
                    (
                        "Previous session summary:\n{existing_summary}\n\n"
                        "Recent conversation turns:\n{recent_history}\n\n"
                        "Produce an updated session memory summary."
                    ),
                ),
            ]
        )
        messages = prompt.format_messages(
            existing_summary=existing_summary.content if existing_summary is not None else "None",
            recent_history="\n".join(history_lines),
        )
        try:
            result = self.llm.invoke(messages)
        except Exception as exc:
            logger.warning("Session summary generation failed: %s", exc)
            return ""
        summary = getattr(result, "content", str(result))
        normalized = " ".join(str(summary).split()).strip()
        return normalized[:1000]

    def _upsert_pg_candidate(
        self,
        username: str,
        session_id: int,
        book_id: int | None,
        project_id: int | None,
        conversation_id: int,
        candidate: dict,
    ) -> MemoryItem:
        scope = candidate["scope"]
        query = self.db.query(MemoryItem).filter(
            MemoryItem.scope == scope,
            MemoryItem.memory_type == candidate["memory_type"],
            MemoryItem.status == "active",
        )
        if scope == "session":
            query = query.filter(
                MemoryItem.username == username,
                MemoryItem.session_id == session_id,
                MemoryItem.book_id == book_id,
            )
        elif scope == "book":
            if book_id is None:
                raise ValueError("book_id is required for book memory.")
            query = query.filter(MemoryItem.book_id == book_id)
        elif scope == "user":
            query = query.filter(MemoryItem.username == username)
        elif scope == "project":
            if project_id is None:
                raise ValueError("project_id is required for project memory.")
            query = query.filter(MemoryItem.project_id == project_id)

        existing = query.filter(MemoryItem.summary == candidate["summary"]).first()
        embedding = self.embeddings.embed_query(candidate["content"])

        if existing is not None:
            existing.content = candidate["content"]
            existing.summary = candidate["summary"]
            if scope == "book":
                existing.book_id = book_id
            if scope == "project":
                existing.project_id = project_id
            existing.source_conversation_id = conversation_id
            existing.salience_score = max(existing.salience_score, candidate["salience_score"])
            existing.confidence_score = max(existing.confidence_score, candidate["confidence_score"])
            existing.embedding = embedding
            existing.updated_at = datetime.now(timezone.utc)
            return existing

        memory = MemoryItem(
            username=username if scope in {"session", "user"} else "",
            session_id=session_id if scope == "session" else None,
            book_id=book_id if scope in {"session", "book"} else None,
            project_id=project_id if scope == "project" else None,
            scope=scope,
            memory_type=candidate["memory_type"],
            content=candidate["content"],
            summary=candidate["summary"],
            source_conversation_id=conversation_id,
            salience_score=candidate["salience_score"],
            confidence_score=candidate["confidence_score"],
            embedding=embedding,
        )
        self.db.add(memory)
        return memory

    @staticmethod
    def _memory_score(item: MemoryItem, query_embedding: Iterable[float]) -> float:
        scope_bonus = {
            "session": 0.05,
            "book": 0.045,
            "project": 0.04,
            "user": 0.02,
        }.get(item.scope, 0.0)
        return (
            cosine_similarity(query_embedding, item.embedding)
            + (item.salience_score * 0.2)
            + (item.confidence_score * 0.1)
            + scope_bonus
        )

    @staticmethod
    def _view_from_model(item: MemoryItem) -> MemoryRecordView:
        return MemoryRecordView(
            id=str(item.id),
            session_id=item.session_id,
            book_id=item.book_id,
            project_id=item.project_id,
            scope=item.scope,
            memory_type=item.memory_type,
            summary=item.summary,
            content=item.content,
            salience_score=item.salience_score,
            confidence_score=item.confidence_score,
            source_conversation_id=item.source_conversation_id,
            created_at=item.created_at,
            updated_at=item.updated_at,
        )

    def _require_embeddings(self) -> None:
        if self.embeddings is None:
            raise ValueError("Embeddings provider is required for memory extraction and retrieval.")

    @staticmethod
    def _looks_like_preference(question: str) -> bool:
        trigger_words = (
            "prefer",
            "please use",
            "answer briefly",
            "be concise",
            "more concise",
            "first give the conclusion",
            "brief",
            "style",
            "format",
            "preference",
        )
        lowered = question.lower()
        return any(word in lowered for word in trigger_words)

    @staticmethod
    def _looks_like_task(question: str) -> bool:
        trigger_words = (
            "i am working on",
            "we are revising",
            "revising",
            "project",
            "working on",
            "editing",
            "task",
            "book line",
        )
        lowered = question.lower()
        return any(word in lowered for word in trigger_words)

    @staticmethod
    def _extract_book_rule(question: str, book_title: str | None) -> str | None:
        if not book_title:
            return None
        trigger_words = (
            "persona",
            "character",
            "roleplay",
            "tone",
            "welcome",
            "must answer as",
            "keep the style",
            "book rule",
            "terminology",
            "rule",
        )
        lowered = question.lower()
        if any(word in lowered for word in trigger_words):
            return question
        return None

    @staticmethod
    def _extract_project_rule(question: str, book_title: str | None) -> str | None:
        if not book_title:
            return None
        trigger_words = (
            "terminology",
            "style guide",
            "naming",
            "convention",
            "rule",
            "standard",
            "revision focus",
            "project rule",
        )
        lowered = question.lower()
        if any(word in lowered for word in trigger_words):
            return question
        return None


def cosine_similarity(left: Iterable[float], right: Iterable[float]) -> float:
    left_values = [float(item) for item in left]
    right_values = [float(item) for item in right]
    if not left_values or not right_values:
        return -1.0

    limit = min(len(left_values), len(right_values))
    left_values = left_values[:limit]
    right_values = right_values[:limit]
    numerator = sum(a * b for a, b in zip(left_values, right_values, strict=True))
    left_norm = math.sqrt(sum(item * item for item in left_values))
    right_norm = math.sqrt(sum(item * item for item in right_values))
    if left_norm == 0 or right_norm == 0:
        return -1.0
    return numerator / (left_norm * right_norm)


def _parse_datetime(value: str | None) -> datetime | None:
    if not value:
        return None
    return datetime.fromisoformat(value.replace("Z", "+00:00"))
