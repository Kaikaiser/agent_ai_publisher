from typing import List

from pydantic import BaseModel


class ConversationItem(BaseModel):
    id: int
    session_id: int | None
    book_id: int | None
    project_id: int | None
    question: str
    answer: str
    grounded: bool
    created_at: str
    book_titles: List[str]
    doc_types: List[str]
    source_count: int
    source_preview: str


class ConversationListResponse(BaseModel):
    items: List[ConversationItem]
