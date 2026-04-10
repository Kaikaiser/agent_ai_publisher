from typing import List, Optional

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    question: str = Field(min_length=1)
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
    answer: str
    grounded: bool
    sources: List[SourceItem]


class ImageChatResponse(ChatResponse):
    recognized_text: str


class ImageRecognitionResponse(BaseModel):
    recognized_text: str
