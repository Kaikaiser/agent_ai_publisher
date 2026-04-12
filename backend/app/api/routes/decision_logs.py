import json
from typing import Optional

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.auth.dependencies import get_current_user
from app.db.models import DecisionLogRecord, User
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

    items = []
    for item in query.all():
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
                "created_at": item.created_at.isoformat() if item.created_at else "",
            }
        )
    return {"items": items}
