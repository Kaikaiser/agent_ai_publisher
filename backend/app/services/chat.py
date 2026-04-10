import json

from sqlalchemy.orm import Session

from app.agents.orchestration.service import AgentOrchestrator
from app.db.models import ConversationRecord
from app.knowledge.vector_store import VectorStoreService
from app.services.image_ocr import ImageOCRService


class ChatService:
    def __init__(self, db: Session, llm, embeddings, vision_llm=None) -> None:
        self.db = db
        self.orchestrator = AgentOrchestrator(llm)
        self.vector_store = VectorStoreService(embeddings)
        # Image OCR can use a dedicated multimodal model while text answers still use the main chat model.
        self.image_ocr = ImageOCRService(vision_llm or llm)

    def ask(self, username: str, role: str, question: str, book_title=None, doc_type=None) -> dict:
        return self._answer_and_store(username, role, question, book_title, doc_type)

    def recognize_image(self, image_bytes: bytes, mime_type: str) -> str:
        return self.image_ocr.extract_text(image_bytes, mime_type)

    def ask_from_image(self, username: str, role: str, image_bytes: bytes, mime_type: str, book_title=None, doc_type=None) -> dict:
        recognized_text = self.recognize_image(image_bytes, mime_type)
        result = self._answer_and_store(username, role, recognized_text, book_title, doc_type)
        result['recognized_text'] = recognized_text
        return result

    def _answer_and_store(self, username: str, role: str, question: str, book_title=None, doc_type=None) -> dict:
        result = self._run_question(role, question, book_title, doc_type)
        sources = self._build_sources(result['documents'])
        record = ConversationRecord(
            username=username,
            question=question,
            answer=result['answer'],
            grounded=result['grounded'],
            sources_json=json.dumps(sources, ensure_ascii=False),
        )
        self.db.add(record)
        self.db.commit()
        self.db.refresh(record)
        return {
            'conversation_id': record.id,
            'answer': result['answer'],
            'grounded': result['grounded'],
            'sources': sources,
        }

    def _run_question(self, role: str, question: str, book_title=None, doc_type=None) -> dict:
        def search_func(query: str):
            return self.vector_store.search(query, role=role, book_title=book_title, doc_type=doc_type)

        return self.orchestrator.run(question, search_func)

    @staticmethod
    def _build_sources(documents) -> list[dict]:
        sources = []
        for item in documents:
            content = item.page_content.strip()
            sources.append(
                {
                    'document_id': item.metadata.get('document_id'),
                    'content': content,
                    'preview': ChatService._build_preview(content),
                    'filename': item.metadata.get('filename', ''),
                    'book_title': item.metadata.get('book_title', ''),
                    'doc_type': item.metadata.get('doc_type', ''),
                    'location': item.metadata.get('location', '未知位置'),
                }
            )
        return sources

    @staticmethod
    def _build_preview(content: str, limit: int = 180) -> str:
        normalized = " ".join(content.split())
        if len(normalized) <= limit:
            return normalized
        return f"{normalized[:limit].rstrip()}..."
