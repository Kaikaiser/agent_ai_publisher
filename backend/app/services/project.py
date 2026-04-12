from __future__ import annotations

from sqlalchemy.orm import Session

from app.db.models import ProjectRecord


class ProjectService:
    def __init__(self, db: Session) -> None:
        self.db = db

    def ensure_project(self, book_title: str | None, doc_type: str | None) -> ProjectRecord | None:
        normalized_book_title = (book_title or "").strip()
        normalized_doc_type = (doc_type or "").strip()
        if not normalized_book_title and not normalized_doc_type:
            return None

        project_key = self._build_project_key(normalized_book_title, normalized_doc_type)
        project = self.db.query(ProjectRecord).filter(ProjectRecord.project_key == project_key).first()
        if project is not None:
            if not project.book_title and normalized_book_title:
                project.book_title = normalized_book_title
            if not project.doc_type and normalized_doc_type:
                project.doc_type = normalized_doc_type
            if not project.name:
                project.name = self._build_project_name(normalized_book_title, normalized_doc_type)
            return project

        project = ProjectRecord(
            project_type="book",
            project_key=project_key,
            name=self._build_project_name(normalized_book_title, normalized_doc_type),
            book_title=normalized_book_title,
            doc_type=normalized_doc_type,
            description="Auto-created from chat context.",
            status="active",
            decision_mode=self._build_decision_mode(normalized_doc_type),
            fallback_policy="conservative_answer",
            citation_policy="required" if normalized_doc_type in {"textbook", "exercise"} else "optional",
            allow_roleplay=self._build_decision_mode(normalized_doc_type) == "immersive_character",
            scope_guard="book_only",
            memory_policy="session_book_project_user",
            safety_level="education_safe" if normalized_doc_type in {"textbook", "exercise"} else "general",
        )
        self.db.add(project)
        self.db.flush()
        return project

    def list_projects(self) -> list[ProjectRecord]:
        return self.db.query(ProjectRecord).order_by(ProjectRecord.updated_at.desc(), ProjectRecord.id.desc()).all()

    @staticmethod
    def _build_project_key(book_title: str, doc_type: str) -> str:
        return f"{book_title or 'unknown'}|{doc_type or 'unknown'}".lower()

    @staticmethod
    def _build_project_name(book_title: str, doc_type: str) -> str:
        if book_title and doc_type:
            return f"{book_title} / {doc_type}"
        return book_title or doc_type or "Untitled Project"

    @staticmethod
    def _build_decision_mode(doc_type: str) -> str:
        lowered = (doc_type or "").lower()
        if lowered in {"textbook", "exercise", "worksheet"}:
            return "strict_knowledge"
        if lowered in {"novel", "story", "picture-book", "picture_book", "fiction"}:
            return "immersive_character"
        return "knowledge_with_style"
