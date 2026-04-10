import json
from pathlib import Path
from typing import List, Optional

import re
import shutil

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from app.core.config import get_settings

TOKEN_PATTERN = re.compile(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]{2,}")


class VectorStoreService:
    def __init__(self, embeddings) -> None:
        self.embeddings = embeddings
        self.settings = get_settings()
        self.index_path = Path(self.settings.vector_store_dir) / "faiss_index"

    def exists(self) -> bool:
        return self.index_path.exists()

    def clear(self) -> None:
        if self.index_path.exists():
            shutil.rmtree(self.index_path)

    def save_documents(self, documents: List[Document]) -> None:
        if not documents:
            return
        if self.exists():
            store = self._load_store()
            store.add_documents(documents)
        else:
            store = FAISS.from_documents(documents, self.embeddings)
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        store.save_local(str(self.index_path))

    def replace_documents(self, documents: List[Document]) -> None:
        self.clear()
        if not documents:
            return
        store = FAISS.from_documents(documents, self.embeddings)
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        store.save_local(str(self.index_path))

    def search(self, query: str, role: str, book_title: Optional[str] = None, doc_type: Optional[str] = None, k: int = 4):
        if not self.exists():
            return []
        store = self._load_store()

        def metadata_filter(metadata: dict) -> bool:
            allowed_role = metadata.get("allowed_role", "user")
            role_allowed = role == "admin" or allowed_role == role
            if not role_allowed:
                return False
            if book_title and metadata.get("book_title") != book_title:
                return False
            if doc_type and metadata.get("doc_type") != doc_type:
                return False
            return True

        fetch_k = max(k * 3, 12)
        documents = store.similarity_search(query, k=fetch_k, filter=metadata_filter)
        reranked = self._rerank_documents(documents, query=query, book_title=book_title, doc_type=doc_type)
        return reranked[:k]

    def _load_store(self) -> FAISS:
        return FAISS.load_local(
            str(self.index_path),
            self.embeddings,
            allow_dangerous_deserialization=True,
        )

    @staticmethod
    def _rerank_documents(documents: List[Document], query: str, book_title: Optional[str], doc_type: Optional[str]) -> List[Document]:
        tokens = [token.lower() for token in TOKEN_PATTERN.findall(query.lower()) if len(token.strip()) > 1]

        def score_document(item: Document, index: int) -> tuple[int, int]:
            metadata = item.metadata or {}
            haystack = " ".join(
                [
                    item.page_content.lower(),
                    str(metadata.get("filename", "")).lower(),
                    str(metadata.get("book_title", "")).lower(),
                    str(metadata.get("doc_type", "")).lower(),
                    str(metadata.get("location", "")).lower(),
                ]
            )

            score = 0
            for token in tokens:
                if token in haystack:
                    score += 2 if len(token) >= 4 else 1
            if book_title and metadata.get("book_title") == book_title:
                score += 6
            if doc_type and metadata.get("doc_type") == doc_type:
                score += 4
            return score, -index

        ranked = sorted(enumerate(documents), key=lambda pair: score_document(pair[1], pair[0]), reverse=True)
        return [item for _, item in ranked]


def serialize_documents(documents: List[Document]) -> str:
    payload = []
    for item in documents:
        payload.append(
            {
                "content": item.page_content,
                "filename": item.metadata.get("filename", ""),
                "book_title": item.metadata.get("book_title", ""),
                "doc_type": item.metadata.get("doc_type", ""),
                "location": item.metadata.get("location", "unknown"),
            }
        )
    return json.dumps(payload, ensure_ascii=False)
