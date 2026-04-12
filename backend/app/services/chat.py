from __future__ import annotations

import json

from sqlalchemy.orm import Session

from app.agents.orchestration.service import AgentOrchestrator
from app.db.models import ConversationRecord, DecisionLogRecord
from app.knowledge.vector_store import VectorStoreService
from app.services.decision import DecisionService
from app.services.image_ocr import ImageOCRService
from app.services.memory import ConversationSessionService, MemoryService
from app.services.project import ProjectService


class ChatService:
    def __init__(self, db: Session, llm, embeddings, vision_llm=None) -> None:
        self.db = db
        self.vector_store = VectorStoreService(db, embeddings)
        self.memory_service = MemoryService(db, embeddings)
        self.session_service = ConversationSessionService(db)
        self.project_service = ProjectService(db)
        self.decision_service = DecisionService()
        self.orchestrator = AgentOrchestrator(llm)
        self.image_ocr = ImageOCRService(vision_llm or llm)

    def ask(
        self,
        username: str,
        role: str,
        question: str,
        book_title=None,
        doc_type=None,
        session_id: int | None = None,
    ) -> dict:
        return self._answer_and_store(
            username,
            role,
            question,
            book_title,
            doc_type,
            session_id=session_id,
            input_source="text",
        )

    def recognize_image(self, image_bytes: bytes, mime_type: str) -> str:
        return self.image_ocr.extract_text(image_bytes, mime_type)

    def ask_from_image(
        self,
        username: str,
        role: str,
        image_bytes: bytes,
        mime_type: str,
        book_title=None,
        doc_type=None,
        session_id: int | None = None,
    ) -> dict:
        recognized_text = self.recognize_image(image_bytes, mime_type)
        result = self._answer_and_store(
            username,
            role,
            recognized_text,
            book_title,
            doc_type,
            session_id=session_id,
            input_source="image",
        )
        result["recognized_text"] = recognized_text
        return result

    def _answer_and_store(
        self,
        username: str,
        role: str,
        question: str,
        book_title=None,
        doc_type=None,
        session_id: int | None = None,
        input_source: str = "text",
    ) -> dict:
        project = self.project_service.ensure_project(book_title=book_title, doc_type=doc_type)
        book_id = project.id if project is not None else None
        project_id = project.id if project is not None else None
        session = self.session_service.ensure_session(
            username=username,
            session_id=session_id,
            question=question,
            book_id=book_id,
            project_id=project_id,
        )
        effective_question, resumed_from_clarification = self.session_service.build_effective_question(session, question)
        decision = self.decision_service.plan(
            effective_question,
            decision_mode=project.decision_mode if project is not None else "strict_knowledge",
            fallback_policy=project.fallback_policy if project is not None else "conservative_answer",
            citation_policy=project.citation_policy if project is not None else "optional",
            allow_roleplay=bool(project.allow_roleplay) if project is not None else False,
            has_image_input=input_source == "image",
        )
        result = self._run_question(
            username=username,
            session_id=session.id,
            book_id=book_id,
            project_id=project_id,
            intent_type=decision.intent_type,
            route_name=decision.route_name,
            selected_tools=decision.selected_tools,
            fallback_policy=decision.fallback_policy,
            memory_scopes=decision.memory_scopes,
            clarification_needed=decision.clarification_needed,
            clarification_slot=decision.clarification_slot,
            clarification_prompt=decision.clarification_prompt,
            decision_mode=decision.decision_mode,
            citation_policy=project.citation_policy if project is not None else "optional",
            allow_roleplay=bool(project.allow_roleplay) if project is not None else False,
            decision_note=decision.note,
            role=role,
            question=effective_question,
            book_title=book_title,
            doc_type=doc_type,
        )
        sources = self._build_sources(result["documents"])
        if result.get("clarification_needed"):
            clarification_slot = decision.clarification_slot or "topic"
            self.session_service.mark_pending_clarification(
                session,
                original_question=effective_question,
                clarification_slot=clarification_slot,
                clarification_prompt=result["answer"],
            )
        else:
            clarification_slot = None
            self.session_service.clear_pending_clarification(session)
        record = ConversationRecord(
            session_id=session.id,
            book_id=book_id,
            project_id=project_id,
            username=username,
            question=question,
            answer=result["answer"],
            grounded=result["grounded"],
            sources_json=json.dumps(sources, ensure_ascii=False),
        )
        self.db.add(record)
        self.db.flush()
        if not result.get("clarification_needed"):
            self.memory_service.extract_and_upsert(
                username=username,
                session_id=session.id,
                book_id=book_id,
                project_id=project_id,
                conversation_id=record.id,
                question=effective_question,
                answer=result["answer"],
                grounded=result["grounded"],
                book_title=book_title,
                doc_type=doc_type,
                decision_mode=decision.decision_mode,
            )
        self.db.add(
            DecisionLogRecord(
                username=username,
                session_id=session.id,
                book_id=book_id,
                project_id=project_id,
                input_source=input_source,
                intent_type=decision.intent_type,
                route_name=decision.route_name,
                decision_mode=decision.decision_mode,
                fallback_policy=decision.fallback_policy,
                selected_tools_json=json.dumps(decision.selected_tools, ensure_ascii=False),
                memory_scopes_json=json.dumps(decision.memory_scopes, ensure_ascii=False),
                grounded=result["grounded"],
                clarification_needed=result.get("clarification_needed", decision.clarification_needed),
                note=self._compose_decision_note(
                    base_note=decision.note,
                    decision_trace=result.get("decision_trace"),
                    execution_trace=result.get("execution_trace"),
                    resumed_from_clarification=resumed_from_clarification,
                    evidence_quality=result.get("evidence_quality", "none"),
                ),
            )
        )
        self.db.commit()
        self.db.refresh(record)
        return {
            "conversation_id": record.id,
            "session_id": session.id,
            "book_id": book_id,
            "project_id": project_id,
            "route_name": decision.route_name,
            "decision_mode": decision.decision_mode,
            "intent_type": decision.intent_type,
            "clarification_needed": result.get("clarification_needed", False),
            "clarification_slot": clarification_slot,
            "resumed_from_clarification": resumed_from_clarification,
            "evidence_quality": result.get("evidence_quality", "none"),
            "answer": result["answer"],
            "grounded": result["grounded"],
            "sources": sources,
        }

    def _run_question(
        self,
        username: str,
        session_id: int,
        book_id: int | None,
        project_id: int | None,
        intent_type: str,
        route_name: str,
        selected_tools: list[str],
        fallback_policy: str,
        memory_scopes: list[str],
        clarification_needed: bool,
        clarification_slot: str | None,
        clarification_prompt: str | None,
        decision_mode: str,
        citation_policy: str,
        allow_roleplay: bool,
        decision_note: str,
        role: str,
        question: str,
        book_title=None,
        doc_type=None,
    ) -> dict:
        def search_func(query: str):
            return self.vector_store.search(query, role=role, book_title=book_title, doc_type=doc_type)

        self.orchestrator.memory_search_func = lambda query: self.memory_service.search(
            username=username,
            session_id=session_id,
            book_id=book_id,
            project_id=project_id,
            query=query,
        )
        self.orchestrator.response_policy = {
            "intent_type": intent_type,
            "route_name": route_name,
            "selected_tools": selected_tools,
            "fallback_policy": fallback_policy,
            "memory_scopes": memory_scopes,
            "clarification_needed": clarification_needed,
            "clarification_slot": clarification_slot,
            "clarification_prompt": clarification_prompt,
            "allow_query_retry": True,
            "decision_mode": decision_mode,
            "citation_policy": citation_policy,
            "allow_roleplay": allow_roleplay,
            "note": decision_note,
            "max_steps": 2,
        }
        return self.orchestrator.run(question, search_func)

    @staticmethod
    def _build_sources(documents) -> list[dict]:
        sources = []
        for item in documents:
            content = item.page_content.strip()
            sources.append(
                {
                    "document_id": item.metadata.get("document_id"),
                    "content": content,
                    "preview": ChatService._build_preview(content),
                    "filename": item.metadata.get("filename", ""),
                    "book_title": item.metadata.get("book_title", ""),
                    "doc_type": item.metadata.get("doc_type", ""),
                    "location": item.metadata.get("location", "unknown"),
                }
            )
        return sources

    @staticmethod
    def _build_preview(content: str, limit: int = 180) -> str:
        normalized = " ".join(content.split())
        if len(normalized) <= limit:
            return normalized
        return f"{normalized[:limit].rstrip()}..."

    @staticmethod
    def _compose_decision_note(
        base_note: str,
        decision_trace=None,
        execution_trace=None,
        resumed_from_clarification: bool = False,
        evidence_quality: str = "none",
    ) -> str:
        parts = [base_note] if base_note else []
        if decision_trace:
            thought_summary = [
                f"{item.intent}:{item.next_action}:{item.evidence_status}"
                for item in decision_trace
                if hasattr(item, "intent") and hasattr(item, "next_action") and hasattr(item, "evidence_status")
            ]
            parts.append(f"thought_trace={thought_summary}")
        if execution_trace:
            parts.append(f"execution_trace={execution_trace}")
        if resumed_from_clarification:
            parts.append("resumed_from_clarification=true")
        parts.append(f"evidence_quality={evidence_quality}")
        return " | ".join(parts)
