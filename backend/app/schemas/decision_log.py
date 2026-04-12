from typing import List

from pydantic import BaseModel


class DecisionLogItemResponse(BaseModel):
    id: int
    session_id: int | None
    book_id: int | None
    project_id: int | None
    input_source: str
    intent_type: str
    route_name: str
    decision_mode: str
    fallback_policy: str
    grounded: bool
    clarification_needed: bool
    selected_tools: List[str]
    memory_scopes: List[str]
    note: str
    created_at: str


class DecisionLogListResponse(BaseModel):
    items: List[DecisionLogItemResponse]
