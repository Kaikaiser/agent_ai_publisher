from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.auth.dependencies import get_current_user
from app.db.models import User
from app.db.session import get_db
from app.schemas.project import ProjectListResponse
from app.services.project import ProjectService

router = APIRouter()


@router.get("", response_model=ProjectListResponse)
def list_projects(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    items = ProjectService(db).list_projects()
    return {
        "items": [
            {
                "id": item.id,
                "project_type": item.project_type,
                "project_key": item.project_key,
                "name": item.name,
                "book_title": item.book_title,
                "doc_type": item.doc_type,
                "description": item.description,
                "status": item.status,
                "decision_mode": item.decision_mode,
                "fallback_policy": item.fallback_policy,
                "citation_policy": item.citation_policy,
                "allow_roleplay": item.allow_roleplay,
                "scope_guard": item.scope_guard,
                "memory_policy": item.memory_policy,
                "safety_level": item.safety_level,
            }
            for item in items
        ]
    }
