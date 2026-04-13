import json
from typing import Optional

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.auth.dependencies import get_current_user
from app.db.models import AgentStepLogRecord, DecisionLogRecord, User
from app.db.session import get_db
from app.schemas.decision_log import DecisionLogListResponse

router = APIRouter()


@router.get("", response_model=DecisionLogListResponse)
def list_decision_logs(
    session_id: Optional[int] = None,
    book_id: Optional[int] = None,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    query = db.query(DecisionLogRecord).filter(DecisionLogRecord.username == current_user.username).order_by(DecisionLogRecord.id.desc())
    if session_id is not None:
        query = query.filter(DecisionLogRecord.session_id == session_id)
    if book_id is not None:
        query = query.filter(DecisionLogRecord.book_id == book_id)

    decision_logs = query.all()
    step_rows = (
        db.query(AgentStepLogRecord)
        .filter(AgentStepLogRecord.decision_log_id.in_([item.id for item in decision_logs]))
        .order_by(AgentStepLogRecord.decision_log_id.asc(), AgentStepLogRecord.step_index.asc(), AgentStepLogRecord.id.asc())
        .all()
        if decision_logs
        else []
    )
    steps_by_log_id = {}
    for step in step_rows:
        steps_by_log_id.setdefault(step.decision_log_id, []).append(
            {
                "step_index": step.step_index,
                "executed_action": step.executed_action,
                "used_query": step.used_query,
                "knowledge_hits": step.knowledge_hits,
                "memory_hits": step.memory_hits,
                "evidence_quality": step.evidence_quality,
                "proposed_next_action": step.proposed_next_action,
                "chosen_next_action": step.chosen_next_action,
                "decision_source": step.decision_source,
                "guard_reason": step.guard_reason,
                "should_answer": step.should_answer,
                "should_clarify": step.should_clarify,
                "should_refuse": step.should_refuse,
                "thought_reason": step.thought_reason,
                "created_at": step.created_at.isoformat() if step.created_at else "",
            }
        )

    items = []
    for item in decision_logs:
        items.append(
            {
                "id": item.id,
                "session_id": item.session_id,
                "book_id": item.book_id,
                "project_id": item.project_id,
                "input_source": item.input_source,
                "intent_type": item.intent_type,
                "route_name": item.route_name,
                "decision_mode": item.decision_mode,
                "fallback_policy": item.fallback_policy,
                "grounded": item.grounded,
                "clarification_needed": item.clarification_needed,
                "selected_tools": json.loads(item.selected_tools_json or "[]"),
                "memory_scopes": json.loads(item.memory_scopes_json or "[]"),
                "note": item.note,
                "steps": steps_by_log_id.get(item.id, []),
                "created_at": item.created_at.isoformat() if item.created_at else "",
            }
        )
    return {"items": items}
