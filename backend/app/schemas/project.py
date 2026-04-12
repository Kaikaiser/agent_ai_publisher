from typing import List

from pydantic import BaseModel


class ProjectItemResponse(BaseModel):
    id: int
    project_type: str
    project_key: str
    name: str
    book_title: str
    doc_type: str
    description: str
    status: str
    decision_mode: str
    fallback_policy: str
    citation_policy: str
    allow_roleplay: bool
    scope_guard: str
    memory_policy: str
    safety_level: str


class ProjectListResponse(BaseModel):
    items: List[ProjectItemResponse]
