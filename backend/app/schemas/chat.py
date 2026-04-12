from typing import List, Optional

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    question: str = Field(min_length=1)
    session_id: Optional[int] = None
    book_title: Optional[str] = None
    doc_type: Optional[str] = None


class SourceItem(BaseModel):
    document_id: int | None = None
    content: str
    preview: str
    filename: str
    book_title: str
    doc_type: str
    location: str


class ChatResponse(BaseModel):
    conversation_id: int
    session_id: int
    book_id: int | None = None
    project_id: int | None = None
    route_name: str | None = None
    decision_mode: str | None = None
    intent_type: str | None = None
    clarification_needed: bool = False
    clarification_slot: str | None = None
    resumed_from_clarification: bool = False
    evidence_quality: str = "none"
    answer: str
    grounded: bool
    sources: List[SourceItem]


class ImageChatResponse(ChatResponse):
    recognized_text: str


class ImageRecognitionResponse(BaseModel):
    recognized_text: str
