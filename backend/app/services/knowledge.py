from __future__ import annotations

from pathlib import Path

from sqlalchemy.orm import Session

from app.db.models import DocumentRecord, IngestionJob, KnowledgeChunk
from app.knowledge.loaders.file_loader import load_documents
from app.knowledge.splitter import build_splitter
from app.knowledge.vector_store import VectorStoreService


class KnowledgeService:
    def __init__(self, db: Session, embeddings) -> None:
        self.db = db
        self.vector_store = VectorStoreService(db, embeddings)

    def import_file(
        self,
        file_path: str,
        filename: str,
        created_by: str,
        book_title: str,
        doc_type: str,
        allowed_role: str,
    ) -> IngestionJob:
        job = self._create_job(filename=filename, created_by=created_by)

        try:
            record = DocumentRecord(
                filename=filename,
                file_path=file_path,
                book_title=book_title,
                doc_type=doc_type,
                allowed_role=allowed_role,
                created_by=created_by,
            )
            self.db.add(record)
            self.db.flush()

            chunks = self._load_chunks(file_path, record)
            chunk_count = self.vector_store.save_documents(chunks)

            job.status = "completed"
            job.message = f"导入完成，共建立 {chunk_count} 个检索切片"
            self.db.commit()
            self.db.refresh(job)
            return job
        except Exception as exc:
            self.db.rollback()
            job.status = "failed"
            job.message = str(exc)
            self.db.add(job)
            self.db.commit()
            self.db.refresh(job)
            raise

    def list_documents(
        self,
        query: str | None = None,
        book_title: str | None = None,
        doc_type: str | None = None,
    ) -> list[dict]:
        records_query = self.db.query(DocumentRecord).order_by(DocumentRecord.id.desc())
        if book_title:
            records_query = records_query.filter(DocumentRecord.book_title == book_title)
        if doc_type:
            records_query = records_query.filter(DocumentRecord.doc_type == doc_type)

        records = records_query.all()
        if query:
            lowered = query.lower()
            records = [
                item
                for item in records
                if lowered in item.filename.lower()
                or lowered in item.book_title.lower()
                or lowered in item.doc_type.lower()
                or lowered in item.created_by.lower()
            ]
        return [self._serialize_record(item) for item in records]

    def delete_document(self, document_id: int) -> dict:
        record = self.db.query(DocumentRecord).filter(DocumentRecord.id == document_id).first()
        if not record:
            raise ValueError("Document not found.")

        file_path = Path(record.file_path)
        self.vector_store.delete_document(record.id)
        self.db.delete(record)
        self.db.flush()
        rebuild_summary = self._summarize_current_index()
        self.db.commit()

        if file_path.exists():
            sibling_count = self.db.query(DocumentRecord).filter(DocumentRecord.file_path == str(file_path)).count()
            if sibling_count == 0:
                file_path.unlink()

        payload = self._serialize_record(record)
        payload.update(rebuild_summary)
        return payload

    def rebuild_index(self) -> dict:
        records = self.db.query(DocumentRecord).order_by(DocumentRecord.id.asc()).all()
        chunks = []
        missing_files = []

        for record in records:
            file_path = Path(record.file_path)
            if not file_path.exists():
                missing_files.append(record.filename)
                continue
            chunks.extend(self._load_chunks(str(file_path), record))

        chunk_count = self.vector_store.replace_documents(chunks)
        self.db.commit()
        return {
            "documents_indexed": len(records) - len(missing_files),
            "chunks_indexed": chunk_count,
            "missing_files": missing_files,
        }

    def _create_job(self, filename: str, created_by: str) -> IngestionJob:
        job = IngestionJob(filename=filename, status="processing", created_by=created_by, message="开始导入知识文件")
        self.db.add(job)
        self.db.commit()
        self.db.refresh(job)
        return job

    @staticmethod
    def _load_chunks(file_path: str, record: DocumentRecord) -> list:
        documents = load_documents(file_path, KnowledgeService._build_metadata(record))
        return build_splitter().split_documents(documents)

    def _summarize_current_index(self) -> dict:
        return {
            "documents_indexed": self.db.query(DocumentRecord).count(),
            "chunks_indexed": self.db.query(KnowledgeChunk).count(),
            "missing_files": [],
        }

    @staticmethod
    def _build_metadata(record: DocumentRecord) -> dict:
        return {
            "filename": record.filename,
            "book_title": record.book_title,
            "doc_type": record.doc_type,
            "allowed_role": record.allowed_role,
            "document_id": record.id,
        }

    @staticmethod
    def _serialize_record(record: DocumentRecord) -> dict:
        file_path = Path(record.file_path)
        return {
            "id": record.id,
            "filename": record.filename,
            "file_path": record.file_path,
            "book_title": record.book_title,
            "doc_type": record.doc_type,
            "allowed_role": record.allowed_role,
            "created_by": record.created_by,
            "created_at": record.created_at.isoformat() if record.created_at else "",
            "exists_on_disk": file_path.exists(),
        }
