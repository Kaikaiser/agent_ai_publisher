from typing import List

from pydantic import BaseModel


class AgentStepLogItemResponse(BaseModel):
    step_index: int
    executed_action: str
    used_query: str
    knowledge_hits: int
    memory_hits: int
    evidence_quality: str
    proposed_next_action: str
    chosen_next_action: str
    decision_source: str
    guard_reason: str
    should_answer: bool
    should_clarify: bool
    should_refuse: bool
    thought_reason: str
    created_at: str


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
    steps: List[AgentStepLogItemResponse]
    created_at: str


class DecisionLogListResponse(BaseModel):
    items: List[DecisionLogItemResponse]
