from dataclasses import dataclass
from datetime import datetime, timezone

from app.db.session import SessionLocal
from app.services.memory import MemoryRecordView, MemoryService


class FakeEmbeddings:
    def embed_query(self, text):
        return [float(len(text) % 7), 1.0, 0.5]


@dataclass
class FakeRankedMemory:
    score: float
    snippet: object


class FakeRedisStore:
    def __init__(self):
        self.saved_candidates = []

    def is_enabled(self):
        return True

    def save_memories(self, username, book_id, session_id, project_id, candidates, conversation_id, embeddings):
        self.saved_candidates.extend(candidates)

    def search(self, username, book_id, session_id, query_embedding, limit):
        return []

    def list_memories(self, username, book_id=None, session_id=None):
        return [
            MemoryRecordView(
                id="redis|admin|2|1|abc",
                session_id=1,
                book_id=2,
                project_id=2,
                scope="session",
                memory_type="context",
                summary="Current conversation focus: book_title=Math, doc_type=textbook",
                content="Current conversation focus: book_title=Math, doc_type=textbook",
                salience_score=0.65,
                confidence_score=0.95,
                source_conversation_id=1,
                created_at=datetime.now(timezone.utc),
                updated_at=datetime.now(timezone.utc),
            )
        ]

    def delete(self, username, memory_id):
        return memory_id == "redis|admin|2|1|abc"


def test_memory_service_uses_redis_for_session_and_postgres_for_user_and_project():
    db = SessionLocal()
    try:
        service = MemoryService(db, FakeEmbeddings())
        fake_redis = FakeRedisStore()
        service.redis_store = fake_redis

        service.extract_and_upsert(
            username="admin",
            session_id=1,
            book_id=2,
            project_id=2,
            conversation_id=1,
            question="Please answer briefly. We are revising the Math project and need one terminology rule.",
            answer="Use concise wording for the current revision task.",
            grounded=True,
            book_title="Math",
            doc_type="textbook",
        )
        db.commit()

        assert any(item["scope"] == "session" for item in fake_redis.saved_candidates)

        user_memories = service.list_for_user("admin", scope="user")
        assert any(item.scope == "user" and item.memory_type == "preference" for item in user_memories)
        assert any(item.scope == "user" and item.memory_type == "task" for item in user_memories)

        book_memories = service.list_for_user("admin", scope="book", book_id=2)
        assert any(item.scope == "book" and item.memory_type == "context" for item in book_memories)
        assert any(item.scope == "book" and item.memory_type == "rule" for item in book_memories)

        project_memories = service.list_for_user("admin", scope="project", project_id=2)
        assert any(item.scope == "project" and item.memory_type == "context" for item in project_memories)
        assert any(item.scope == "project" and item.memory_type == "rule" for item in project_memories)

        session_memories = service.list_for_user("admin", scope="session", session_id=1, book_id=2)
        assert len(session_memories) == 1
        assert session_memories[0].id.startswith("redis|")
        assert session_memories[0].book_id == 2
        assert session_memories[0].project_id == 2

        service.delete_for_user("admin", "redis|admin|2|1|abc")
    finally:
        db.close()
