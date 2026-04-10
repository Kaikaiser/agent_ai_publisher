from pydantic import BaseModel
from typing import List


class KnowledgeImportResponse(BaseModel):
    job_id: int
    status: str
    message: str


class KnowledgeDocumentItem(BaseModel):
    id: int
    filename: str
    file_path: str
    book_title: str
    doc_type: str
    allowed_role: str
    created_by: str
    created_at: str
    exists_on_disk: bool


class KnowledgeDocumentListResponse(BaseModel):
    items: List[KnowledgeDocumentItem]


class KnowledgeRebuildResponse(BaseModel):
    documents_indexed: int
    chunks_indexed: int
    missing_files: List[str]
