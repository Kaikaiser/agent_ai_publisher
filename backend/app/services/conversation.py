import json

from sqlalchemy.orm import Session

from app.db.models import ConversationRecord


class ConversationService:
    def __init__(self, db: Session) -> None:
        self.db = db

    def list_for_user(
        self,
        username: str,
        query: str | None = None,
        grounded: bool | None = None,
        book_title: str | None = None,
        doc_type: str | None = None,
    ):
        records = (
            self.db.query(ConversationRecord)
            .filter(ConversationRecord.username == username)
            .order_by(ConversationRecord.id.desc())
            .all()
        )
        items = []
        query_lower = query.lower() if query else None

        for item in records:
            if grounded is not None and item.grounded != grounded:
                continue

            sources = json.loads(item.sources_json)
            book_titles = sorted({source.get("book_title", "") for source in sources if source.get("book_title")})
            doc_types = sorted({source.get("doc_type", "") for source in sources if source.get("doc_type")})
            source_preview = next((source.get("preview") or source.get("content", "") for source in sources if source.get("preview") or source.get("content")), "")

            if book_title and book_title not in book_titles:
                continue
            if doc_type and doc_type not in doc_types:
                continue
            if query_lower:
                haystack = " ".join([item.question, item.answer, source_preview, " ".join(book_titles), " ".join(doc_types)]).lower()
                if query_lower not in haystack:
                    continue

            items.append(
                {
                    "id": item.id,
                    "question": item.question,
                    "answer": item.answer,
                    "grounded": item.grounded,
                    "created_at": item.created_at.isoformat() if item.created_at else "",
                    "book_titles": book_titles,
                    "doc_types": doc_types,
                    "source_count": len(sources),
                    "source_preview": source_preview,
                }
            )

        return items
