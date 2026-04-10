from fastapi import APIRouter, Depends
from typing import Optional

from sqlalchemy.orm import Session

from app.auth.dependencies import get_current_user
from app.db.models import User
from app.db.session import get_db
from app.schemas.conversation import ConversationItem, ConversationListResponse
from app.services.conversation import ConversationService

router = APIRouter()


@router.get("", response_model=ConversationListResponse)
def list_conversations(
    query: Optional[str] = None,
    grounded: Optional[bool] = None,
    book_title: Optional[str] = None,
    doc_type: Optional[str] = None,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    service = ConversationService(db)
    items = service.list_for_user(
        current_user.username,
        query=query,
        grounded=grounded,
        book_title=book_title,
        doc_type=doc_type,
    )
    return {"items": items}
