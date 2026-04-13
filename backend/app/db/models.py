from __future__ import annotations

from datetime import datetime

from sqlalchemy import JSON, Boolean, DateTime, Float, ForeignKey, Integer, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.types import TypeDecorator

from app.core.config import get_settings
from app.db.base import Base

try:
    from pgvector.sqlalchemy import Vector as PgVector
except ImportError:  # pragma: no cover - exercised in environments without pgvector installed.
    PgVector = None


settings = get_settings()


class VectorJSON(TypeDecorator):
    impl = JSON
    cache_ok = True

    def process_bind_param(self, value, dialect):
        if value is None:
            return None
        return [float(item) for item in value]

    def process_result_value(self, value, dialect):
        if value is None:
            return []
        return [float(item) for item in value]


def build_vector_type(dimensions: int):
    if PgVector is not None:
        return PgVector(dimensions)
    return VectorJSON()


class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    username: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    password_hash: Mapped[str] = mapped_column(String(255))
    role: Mapped[str] = mapped_column(String(32), default="user")
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class ProjectRecord(Base):
    __tablename__ = "projects"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    project_type: Mapped[str] = mapped_column(String(32), default="book", index=True)
    project_key: Mapped[str] = mapped_column(String(255), unique=True, index=True)
    name: Mapped[str] = mapped_column(String(255))
    book_title: Mapped[str] = mapped_column(String(255), default="", index=True)
    doc_type: Mapped[str] = mapped_column(String(64), default="", index=True)
    description: Mapped[str] = mapped_column(Text, default="")
    status: Mapped[str] = mapped_column(String(32), default="active", index=True)
    decision_mode: Mapped[str] = mapped_column(String(32), default="strict_knowledge", index=True)
    fallback_policy: Mapped[str] = mapped_column(String(32), default="conservative_answer")
    citation_policy: Mapped[str] = mapped_column(String(32), default="optional")
    allow_roleplay: Mapped[bool] = mapped_column(Boolean, default=False)
    scope_guard: Mapped[str] = mapped_column(String(32), default="book_only")
    memory_policy: Mapped[str] = mapped_column(String(32), default="session_book_project_user")
    safety_level: Mapped[str] = mapped_column(String(32), default="education_safe")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
    )


class DocumentRecord(Base):
    __tablename__ = "documents"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    filename: Mapped[str] = mapped_column(String(255))
    file_path: Mapped[str] = mapped_column(String(512))
    book_title: Mapped[str] = mapped_column(String(255), index=True)
    doc_type: Mapped[str] = mapped_column(String(64), index=True)
    allowed_role: Mapped[str] = mapped_column(String(32), default="user", index=True)
    source_type: Mapped[str] = mapped_column(String(32), default="upload")
    created_by: Mapped[str] = mapped_column(String(64))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class IngestionJob(Base):
    __tablename__ = "ingestion_jobs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    filename: Mapped[str] = mapped_column(String(255))
    status: Mapped[str] = mapped_column(String(32), default="pending")
    message: Mapped[str] = mapped_column(Text, default="")
    created_by: Mapped[str] = mapped_column(String(64))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class ConversationSession(Base):
    __tablename__ = "conversation_sessions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    username: Mapped[str] = mapped_column(String(64), index=True)
    book_id: Mapped[int | None] = mapped_column(
        ForeignKey("projects.id", ondelete="SET NULL"),
        index=True,
        nullable=True,
    )
    project_id: Mapped[int | None] = mapped_column(
        ForeignKey("projects.id", ondelete="SET NULL"),
        index=True,
        nullable=True,
    )
    title: Mapped[str] = mapped_column(String(255), default="")
    pending_clarification: Mapped[bool] = mapped_column(Boolean, default=False, index=True)
    pending_question: Mapped[str] = mapped_column(Text, default="")
    pending_clarification_slot: Mapped[str] = mapped_column(String(64), default="")
    pending_prompt: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
    )


class ConversationRecord(Base):
    __tablename__ = "conversations"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    session_id: Mapped[int | None] = mapped_column(
        ForeignKey("conversation_sessions.id", ondelete="SET NULL"),
        index=True,
        nullable=True,
    )
    book_id: Mapped[int | None] = mapped_column(
        ForeignKey("projects.id", ondelete="SET NULL"),
        index=True,
        nullable=True,
    )
    project_id: Mapped[int | None] = mapped_column(
        ForeignKey("projects.id", ondelete="SET NULL"),
        index=True,
        nullable=True,
    )
    username: Mapped[str] = mapped_column(String(64), index=True)
    question: Mapped[str] = mapped_column(Text)
    answer: Mapped[str] = mapped_column(Text)
    grounded: Mapped[bool] = mapped_column(Boolean, default=False)
    sources_json: Mapped[str] = mapped_column(Text, default="[]")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class DecisionLogRecord(Base):
    __tablename__ = "decision_logs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    username: Mapped[str] = mapped_column(String(64), index=True)
    session_id: Mapped[int | None] = mapped_column(
        ForeignKey("conversation_sessions.id", ondelete="SET NULL"),
        index=True,
        nullable=True,
    )
    book_id: Mapped[int | None] = mapped_column(
        ForeignKey("projects.id", ondelete="SET NULL"),
        index=True,
        nullable=True,
    )
    project_id: Mapped[int | None] = mapped_column(
        ForeignKey("projects.id", ondelete="SET NULL"),
        index=True,
        nullable=True,
    )
    input_source: Mapped[str] = mapped_column(String(32), default="text")
    intent_type: Mapped[str] = mapped_column(String(32), index=True)
    route_name: Mapped[str] = mapped_column(String(32), default="knowledge_answer")
    decision_mode: Mapped[str] = mapped_column(String(32), index=True)
    fallback_policy: Mapped[str] = mapped_column(String(32))
    selected_tools_json: Mapped[str] = mapped_column(Text, default="[]")
    memory_scopes_json: Mapped[str] = mapped_column(Text, default="[]")
    grounded: Mapped[bool] = mapped_column(Boolean, default=False)
    clarification_needed: Mapped[bool] = mapped_column(Boolean, default=False)
    note: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class AgentStepLogRecord(Base):
    __tablename__ = "agent_step_logs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    decision_log_id: Mapped[int] = mapped_column(ForeignKey("decision_logs.id", ondelete="CASCADE"), index=True)
    username: Mapped[str] = mapped_column(String(64), index=True)
    session_id: Mapped[int | None] = mapped_column(
        ForeignKey("conversation_sessions.id", ondelete="SET NULL"),
        index=True,
        nullable=True,
    )
    book_id: Mapped[int | None] = mapped_column(
        ForeignKey("projects.id", ondelete="SET NULL"),
        index=True,
        nullable=True,
    )
    project_id: Mapped[int | None] = mapped_column(
        ForeignKey("projects.id", ondelete="SET NULL"),
        index=True,
        nullable=True,
    )
    step_index: Mapped[int] = mapped_column(Integer, default=1)
    executed_action: Mapped[str] = mapped_column(String(64), default="")
    used_query: Mapped[str] = mapped_column(Text, default="")
    knowledge_hits: Mapped[int] = mapped_column(Integer, default=0)
    memory_hits: Mapped[int] = mapped_column(Integer, default=0)
    evidence_quality: Mapped[str] = mapped_column(String(32), default="none")
    proposed_next_action: Mapped[str] = mapped_column(String(64), default="")
    chosen_next_action: Mapped[str] = mapped_column(String(64), default="")
    decision_source: Mapped[str] = mapped_column(String(32), default="rule")
    guard_reason: Mapped[str] = mapped_column(Text, default="")
    should_answer: Mapped[bool] = mapped_column(Boolean, default=False)
    should_clarify: Mapped[bool] = mapped_column(Boolean, default=False)
    should_refuse: Mapped[bool] = mapped_column(Boolean, default=False)
    thought_reason: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class MemoryItem(Base):
    __tablename__ = "memory_items"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    username: Mapped[str] = mapped_column(String(64), index=True, default="")
    session_id: Mapped[int | None] = mapped_column(
        ForeignKey("conversation_sessions.id", ondelete="SET NULL"),
        index=True,
        nullable=True,
    )
    book_id: Mapped[int | None] = mapped_column(
        ForeignKey("projects.id", ondelete="SET NULL"),
        index=True,
        nullable=True,
    )
    project_id: Mapped[int | None] = mapped_column(
        ForeignKey("projects.id", ondelete="SET NULL"),
        index=True,
        nullable=True,
    )
    scope: Mapped[str] = mapped_column(String(32), index=True)
    memory_type: Mapped[str] = mapped_column(String(32), index=True)
    content: Mapped[str] = mapped_column(Text)
    summary: Mapped[str] = mapped_column(String(255), default="")
    source_conversation_id: Mapped[int | None] = mapped_column(
        ForeignKey("conversations.id", ondelete="SET NULL"),
        index=True,
        nullable=True,
    )
    salience_score: Mapped[float] = mapped_column(Float, default=0.5)
    confidence_score: Mapped[float] = mapped_column(Float, default=0.5)
    hit_count: Mapped[int] = mapped_column(Integer, default=0)
    status: Mapped[str] = mapped_column(String(32), default="active", index=True)
    embedding: Mapped[list[float]] = mapped_column(build_vector_type(settings.embedding_dimensions))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
    )
    last_accessed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class KnowledgeChunk(Base):
    __tablename__ = "knowledge_chunks"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    document_id: Mapped[int] = mapped_column(ForeignKey("documents.id", ondelete="CASCADE"), index=True)
    chunk_index: Mapped[int] = mapped_column(Integer, default=0)
    page_number: Mapped[int | None] = mapped_column(Integer, nullable=True, index=True)
    chapter_title: Mapped[str] = mapped_column(String(255), default="", index=True)
    section_title: Mapped[str] = mapped_column(String(255), default="", index=True)
    citation_label: Mapped[str] = mapped_column(String(255), default="")
    content: Mapped[str] = mapped_column(Text)
    content_markdown: Mapped[str] = mapped_column(Text, default="")
    embedding: Mapped[list[float]] = mapped_column(build_vector_type(settings.embedding_dimensions))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
