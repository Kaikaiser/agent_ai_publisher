from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from sqlalchemy.orm import Session

from app.api.deps import get_embeddings
from app.auth.dependencies import require_admin
from app.core.config import get_settings
from app.db.models import User
from app.db.session import get_db
from app.schemas.knowledge import KnowledgeDocumentListResponse, KnowledgeImportResponse, KnowledgeRebuildResponse
from app.services.knowledge import KnowledgeService

router = APIRouter()


@router.post("/import", response_model=KnowledgeImportResponse)
def import_knowledge(
    file: UploadFile = File(...),
    book_title: str = Form(...),
    doc_type: str = Form(...),
    allowed_role: str = Form("user"),
    current_user: User = Depends(require_admin),
    db: Session = Depends(get_db),
    embeddings=Depends(get_embeddings),
):
    settings = get_settings()
    upload_path = Path(settings.upload_dir) / file.filename
    upload_path.parent.mkdir(parents=True, exist_ok=True)
    with open(upload_path, "wb") as output:
        output.write(file.file.read())
    try:
        service = KnowledgeService(db, embeddings)
        job = service.import_file(
            file_path=str(upload_path),
            filename=file.filename,
            created_by=current_user.username,
            book_title=book_title,
            doc_type=doc_type,
            allowed_role=allowed_role,
        )
        return {"job_id": job.id, "status": job.status, "message": job.message}
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))


@router.get("/documents", response_model=KnowledgeDocumentListResponse)
def list_documents(
    query: str | None = None,
    book_title: str | None = None,
    doc_type: str | None = None,
    current_user: User = Depends(require_admin),
    db: Session = Depends(get_db),
    embeddings=Depends(get_embeddings),
):
    service = KnowledgeService(db, embeddings)
    items = service.list_documents(query=query, book_title=book_title, doc_type=doc_type)
    return {"items": items}


@router.delete("/documents/{document_id}")
def delete_document(
    document_id: int,
    current_user: User = Depends(require_admin),
    db: Session = Depends(get_db),
    embeddings=Depends(get_embeddings),
):
    try:
        service = KnowledgeService(db, embeddings)
        return service.delete_document(document_id)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc))


@router.post("/reindex", response_model=KnowledgeRebuildResponse)
def rebuild_index(
    current_user: User = Depends(require_admin),
    db: Session = Depends(get_db),
    embeddings=Depends(get_embeddings),
):
    service = KnowledgeService(db, embeddings)
    return service.rebuild_index()
