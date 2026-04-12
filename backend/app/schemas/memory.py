from typing import List, Optional

from pydantic import BaseModel


class MemoryItemResponse(BaseModel):
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
    created_at: str
    updated_at: str


class MemoryListResponse(BaseModel):
    items: List[MemoryItemResponse]


class MemoryListRequest(BaseModel):
    scope: Optional[str] = None
    session_id: Optional[int] = None
    book_id: Optional[int] = None
    project_id: Optional[int] = None
