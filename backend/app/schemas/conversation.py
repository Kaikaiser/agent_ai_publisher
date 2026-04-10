from typing import List

from pydantic import BaseModel


class ConversationItem(BaseModel):
    id: int
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
