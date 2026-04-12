from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.auth.dependencies import get_current_user
from app.db.models import User
from app.db.session import get_db
from app.schemas.memory import MemoryListResponse
from app.services.memory import MemoryService

router = APIRouter()


@router.get("", response_model=MemoryListResponse)
def list_memories(
    scope: Optional[str] = None,
    session_id: Optional[int] = None,
    book_id: Optional[int] = None,
    project_id: Optional[int] = None,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    service = MemoryService(db)
    items = service.list_for_user(
        current_user.username,
        scope=scope,
        session_id=session_id,
        book_id=book_id,
        project_id=project_id,
    )
    return {
        "items": [
            {
                "id": item.id,
                "session_id": item.session_id,
                "book_id": item.book_id,
                "project_id": item.project_id,
                "scope": item.scope,
                "memory_type": item.memory_type,
                "summary": item.summary,
                "content": item.content,
                "salience_score": item.salience_score,
                "confidence_score": item.confidence_score,
                "source_conversation_id": item.source_conversation_id,
                "created_at": item.created_at.isoformat() if item.created_at else "",
                "updated_at": item.updated_at.isoformat() if item.updated_at else "",
            }
            for item in items
        ]
    }


@router.delete("/{memory_id}")
def delete_memory(
    memory_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    try:
        MemoryService(db).delete_for_user(current_user.username, memory_id)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc))
    return {"deleted": True}
