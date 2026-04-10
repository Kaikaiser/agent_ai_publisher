from typing import Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from sqlalchemy.orm import Session

from app.api.deps import get_embeddings, get_llm, get_vision_llm
from app.auth.dependencies import get_current_user
from app.db.models import User
from app.db.session import get_db
from app.schemas.chat import ChatRequest, ChatResponse, ImageChatResponse, ImageRecognitionResponse
from app.services.chat import ChatService

router = APIRouter()


def _read_image(file: UploadFile) -> tuple[bytes, str]:
    if not file.content_type or not file.content_type.startswith('image/'):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail='仅支持上传图片文件')
    image_bytes = file.file.read()
    if not image_bytes:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail='上传的图片为空')
    return image_bytes, file.content_type


@router.post('/ask', response_model=ChatResponse)
def ask(
    payload: ChatRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
    llm=Depends(get_llm),
    embeddings=Depends(get_embeddings),
):
    try:
        service = ChatService(db, llm, embeddings)
        return service.ask(
            username=current_user.username,
            role=current_user.role,
            question=payload.question,
            book_title=payload.book_title,
            doc_type=payload.doc_type,
        )
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))


@router.post('/recognize-image', response_model=ImageRecognitionResponse)
def recognize_image(
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
    llm=Depends(get_llm),
    vision_llm=Depends(get_vision_llm),
    embeddings=Depends(get_embeddings),
):
    image_bytes, mime_type = _read_image(file)
    try:
        service = ChatService(db, llm, embeddings, vision_llm=vision_llm)
        recognized_text = service.recognize_image(image_bytes, mime_type)
        return {'recognized_text': recognized_text}
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))


@router.post('/ask-image', response_model=ImageChatResponse)
def ask_image(
    file: UploadFile = File(...),
    book_title: Optional[str] = Form(default=None),
    doc_type: Optional[str] = Form(default=None),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
    llm=Depends(get_llm),
    vision_llm=Depends(get_vision_llm),
    embeddings=Depends(get_embeddings),
):
    image_bytes, mime_type = _read_image(file)
    try:
        service = ChatService(db, llm, embeddings, vision_llm=vision_llm)
        return service.ask_from_image(
            username=current_user.username,
            role=current_user.role,
            image_bytes=image_bytes,
            mime_type=mime_type,
            book_title=book_title or None,
            doc_type=doc_type or None,
        )
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))