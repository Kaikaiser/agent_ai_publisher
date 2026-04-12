from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass
from typing import Iterable, Optional

from langchain_core.documents import Document
from sqlalchemy import delete
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.db.models import DocumentRecord, KnowledgeChunk
from app.services.rerank import ZhipuRerankService

try:
    from elasticsearch import Elasticsearch
    from elasticsearch.helpers import bulk as elasticsearch_bulk
except ImportError:  # pragma: no cover - exercised in environments without elasticsearch installed.
    Elasticsearch = None
    elasticsearch_bulk = None


logger = logging.getLogger(__name__)


@dataclass
class SearchCandidate:
    chunk_id: int
    document: Document


class VectorStoreService:
    def __init__(self, db: Session, embeddings) -> None:
        self.db = db
        self.embeddings = embeddings
        self.settings = get_settings()
        self.reranker = ZhipuRerankService()
        self._elasticsearch = self._build_elasticsearch_client()
        self._index_ready = False

    def exists(self) -> bool:
        return self.db.query(KnowledgeChunk.id).first() is not None

    def clear(self) -> None:
        self.db.execute(delete(KnowledgeChunk))
        self.db.flush()
        self._reset_elasticsearch_index()

    def save_documents(self, documents: list[Document]) -> int:
        if not documents:
            return 0
        chunks = self._persist_documents(documents)
        self._index_chunks_in_elasticsearch(chunks)
        return len(chunks)

    def replace_documents(self, documents: list[Document]) -> int:
        self.clear()
        return self.save_documents(documents)

    def delete_document(self, document_id: int) -> int:
        chunk_ids = [
            chunk_id
            for (chunk_id,) in self.db.query(KnowledgeChunk.id).filter(KnowledgeChunk.document_id == document_id).all()
        ]
        if chunk_ids:
            self.db.execute(delete(KnowledgeChunk).where(KnowledgeChunk.document_id == document_id))
            self.db.flush()
            self._delete_chunks_from_elasticsearch(chunk_ids)
        return len(chunk_ids)

    def search(
        self,
        query: str,
        role: str,
        book_title: Optional[str] = None,
        doc_type: Optional[str] = None,
        k: int = 4,
    ) -> list[Document]:
        if not self.exists():
            return []

        dense_candidates = self._dense_search(
            query=query,
            role=role,
            book_title=book_title,
            doc_type=doc_type,
            limit=max(self.settings.hybrid_dense_top_k, k),
        )
        lexical_candidates = self._bm25_search(
            query=query,
            role=role,
            book_title=book_title,
            doc_type=doc_type,
            limit=max(self.settings.hybrid_bm25_top_k, k),
        )
        fused_candidates = self._reciprocal_rank_fusion(dense_candidates, lexical_candidates)
        documents = [item.document for item in fused_candidates[: max(self.settings.hybrid_final_top_k, k)]]
        reranked = self.reranker.rerank(
            query=query,
            documents=documents,
            top_n=min(max(self.settings.rerank_top_n, k), len(documents)),
        )
        return reranked[:k]

    def _persist_documents(self, documents: list[Document]) -> list[KnowledgeChunk]:
        embeddings = self.embeddings.embed_documents([item.page_content for item in documents])
        counters: dict[int, int] = {}
        chunks: list[KnowledgeChunk] = []

        for document, embedding in zip(documents, embeddings, strict=True):
            metadata = document.metadata or {}
            document_id = metadata.get("document_id")
            if document_id is None:
                raise ValueError("Chunk metadata is missing document_id.")

            chunk_index = counters.get(int(document_id), 0)
            counters[int(document_id)] = chunk_index + 1

            chunk = KnowledgeChunk(
                document_id=int(document_id),
                chunk_index=chunk_index,
                page_number=metadata.get("page_number"),
                chapter_title=str(metadata.get("chapter_title", "") or ""),
                section_title=str(metadata.get("section_title", "") or ""),
                citation_label=str(metadata.get("citation_label", "") or ""),
                content=document.page_content,
                content_markdown=str(metadata.get("content_markdown", "") or ""),
                embedding=[float(item) for item in embedding],
            )
            chunks.append(chunk)

        self.db.add_all(chunks)
        self.db.flush()
        return chunks

    def _dense_search(
        self,
        query: str,
        role: str,
        book_title: str | None,
        doc_type: str | None,
        limit: int,
    ) -> list[SearchCandidate]:
        query_vector = self.embeddings.embed_query(query)
        rows = self._load_candidate_rows(role=role, book_title=book_title, doc_type=doc_type)
        scored_rows = []

        for chunk, document_record in rows:
            score = cosine_similarity(query_vector, chunk.embedding)
            scored_rows.append((score, chunk, document_record))

        scored_rows.sort(key=lambda item: item[0], reverse=True)
        return [
            SearchCandidate(chunk_id=chunk.id, document=self._chunk_to_document(chunk, document_record))
            for _, chunk, document_record in scored_rows[:limit]
        ]

    def _bm25_search(
        self,
        query: str,
        role: str,
        book_title: str | None,
        doc_type: str | None,
        limit: int,
    ) -> list[SearchCandidate]:
        if self._elasticsearch is None:
            return []

        self._ensure_elasticsearch_index()
        if self._elasticsearch is None:
            return []

        filters = self._build_search_filters(role=role, book_title=book_title, doc_type=doc_type)

        try:
            response = self._elasticsearch.search(
                index=self.settings.elasticsearch_index_name,
                size=limit,
                query={
                    "bool": {
                        "must": [
                            {
                                "multi_match": {
                                    "query": query,
                                    "fields": [
                                        "content^3",
                                        "content_markdown^2",
                                        "book_title^2",
                                        "doc_type",
                                        "filename",
                                        "chapter_title",
                                        "section_title",
                                        "citation_label",
                                        "location",
                                    ],
                                }
                            }
                        ],
                        "filter": filters,
                    }
                },
            )
        except Exception as exc:  # pragma: no cover - depends on external Elasticsearch runtime.
            logger.warning("Elasticsearch search failed, falling back to dense retrieval only: %s", exc)
            return []

        hits = response.get("hits", {}).get("hits", [])
        return [self._search_candidate_from_hit(hit) for hit in hits]

    def _load_candidate_rows(
        self,
        role: str,
        book_title: str | None,
        doc_type: str | None,
    ) -> list[tuple[KnowledgeChunk, DocumentRecord]]:
        query = (
            self.db.query(KnowledgeChunk, DocumentRecord)
            .join(DocumentRecord, DocumentRecord.id == KnowledgeChunk.document_id)
            .order_by(KnowledgeChunk.id.asc())
        )
        if role != "admin":
            query = query.filter(DocumentRecord.allowed_role == role)
        if book_title:
            query = query.filter(DocumentRecord.book_title == book_title)
        if doc_type:
            query = query.filter(DocumentRecord.doc_type == doc_type)
        return query.all()

    def _chunk_to_document(self, chunk: KnowledgeChunk, document_record: DocumentRecord) -> Document:
        metadata = {
            "document_id": chunk.document_id,
            "filename": document_record.filename,
            "book_title": document_record.book_title,
            "doc_type": document_record.doc_type,
            "allowed_role": document_record.allowed_role,
            "page_number": chunk.page_number,
            "chapter_title": chunk.chapter_title,
            "section_title": chunk.section_title,
            "citation_label": chunk.citation_label,
            "location": self._build_location(chunk),
        }
        if chunk.content_markdown:
            metadata["content_markdown"] = chunk.content_markdown
        return Document(page_content=chunk.content, metadata=metadata)

    @staticmethod
    def _build_search_filters(role: str, book_title: str | None, doc_type: str | None) -> list[dict]:
        filters = []
        if role != "admin":
            filters.append({"term": {"allowed_role": role}})
        if book_title:
            filters.append({"term": {"book_title": book_title}})
        if doc_type:
            filters.append({"term": {"doc_type": doc_type}})
        return filters

    @staticmethod
    def _search_candidate_from_hit(hit: dict) -> SearchCandidate:
        source = hit.get("_source", {})
        return SearchCandidate(
            chunk_id=int(hit["_id"]),
            document=Document(
                page_content=source.get("content", ""),
                metadata={
                    "document_id": source.get("document_id"),
                    "filename": source.get("filename", ""),
                    "book_title": source.get("book_title", ""),
                    "doc_type": source.get("doc_type", ""),
                    "allowed_role": source.get("allowed_role", "user"),
                    "page_number": source.get("page_number"),
                    "chapter_title": source.get("chapter_title", ""),
                    "section_title": source.get("section_title", ""),
                    "citation_label": source.get("citation_label", ""),
                    "location": source.get("location", "unknown"),
                },
            ),
        )

    def _build_elasticsearch_client(self):
        if Elasticsearch is None or not self.settings.elasticsearch_url:
            return None
        try:
            return Elasticsearch(self.settings.elasticsearch_url)
        except Exception as exc:  # pragma: no cover - depends on external Elasticsearch runtime.
            logger.warning("Elasticsearch client init failed, lexical search disabled: %s", exc)
            return None

    def _ensure_elasticsearch_index(self) -> None:
        if self._elasticsearch is None or self._index_ready:
            return
        try:
            if self._elasticsearch.indices.exists(index=self.settings.elasticsearch_index_name):
                self._index_ready = True
                return
            self._elasticsearch.indices.create(
                index=self.settings.elasticsearch_index_name,
                mappings={
                    "properties": {
                        "document_id": {"type": "integer"},
                        "filename": {"type": "keyword"},
                        "book_title": {"type": "keyword"},
                        "doc_type": {"type": "keyword"},
                        "allowed_role": {"type": "keyword"},
                        "page_number": {"type": "integer"},
                        "chapter_title": {"type": "text"},
                        "section_title": {"type": "text"},
                        "citation_label": {"type": "text"},
                        "location": {"type": "text"},
                        "content": {"type": "text"},
                        "content_markdown": {"type": "text"},
                    }
                },
            )
            self._index_ready = True
        except Exception as exc:  # pragma: no cover - depends on external Elasticsearch runtime.
            logger.warning("Elasticsearch index init failed, lexical search disabled: %s", exc)
            self._elasticsearch = None

    def _reset_elasticsearch_index(self) -> None:
        if self._elasticsearch is None:
            return
        try:
            if self._elasticsearch.indices.exists(index=self.settings.elasticsearch_index_name):
                self._elasticsearch.indices.delete(index=self.settings.elasticsearch_index_name)
        except Exception as exc:  # pragma: no cover - depends on external Elasticsearch runtime.
            logger.warning("Failed to reset Elasticsearch index: %s", exc)
            self._elasticsearch = None
        self._index_ready = False

    def _index_chunks_in_elasticsearch(self, chunks: Iterable[KnowledgeChunk]) -> None:
        if self._elasticsearch is None:
            return

        self._ensure_elasticsearch_index()
        if self._elasticsearch is None:
            return

        document_records = self._load_document_records({chunk.document_id for chunk in chunks})
        actions = []
        for chunk in chunks:
            document_record = document_records.get(chunk.document_id)
            if document_record is None:
                continue
            actions.append(
                {
                    "_op_type": "index",
                    "_index": self.settings.elasticsearch_index_name,
                    "_id": str(chunk.id),
                    "_source": {
                        "document_id": chunk.document_id,
                        "filename": document_record.filename,
                        "book_title": document_record.book_title,
                        "doc_type": document_record.doc_type,
                        "allowed_role": document_record.allowed_role,
                        "page_number": chunk.page_number,
                        "chapter_title": chunk.chapter_title,
                        "section_title": chunk.section_title,
                        "citation_label": chunk.citation_label,
                        "location": self._build_location(chunk),
                        "content": chunk.content,
                        "content_markdown": chunk.content_markdown,
                    },
                }
            )

        if not actions:
            return

        try:
            if elasticsearch_bulk is not None:
                elasticsearch_bulk(self._elasticsearch, actions)
            else:  # pragma: no cover - current dependency includes helpers.
                for action in actions:
                    self._elasticsearch.index(
                        index=action["_index"],
                        id=action["_id"],
                        document=action["_source"],
                    )
            self._elasticsearch.indices.refresh(index=self.settings.elasticsearch_index_name)
        except Exception as exc:  # pragma: no cover - depends on external Elasticsearch runtime.
            logger.warning("Failed to index chunks into Elasticsearch: %s", exc)
            self._elasticsearch = None

    def _delete_chunks_from_elasticsearch(self, chunk_ids: list[int]) -> None:
        if self._elasticsearch is None or not chunk_ids:
            return
        try:
            self._elasticsearch.delete_by_query(
                index=self.settings.elasticsearch_index_name,
                query={"terms": {"_id": [str(item) for item in chunk_ids]}},
                refresh=True,
            )
        except Exception as exc:  # pragma: no cover - depends on external Elasticsearch runtime.
            logger.warning("Failed to delete chunks from Elasticsearch: %s", exc)
            self._elasticsearch = None

    def _load_document_records(self, document_ids: set[int]) -> dict[int, DocumentRecord]:
        if not document_ids:
            return {}
        rows = self.db.query(DocumentRecord).filter(DocumentRecord.id.in_(document_ids)).all()
        return {row.id: row for row in rows}

    @staticmethod
    def _reciprocal_rank_fusion(*rank_lists: list[SearchCandidate]) -> list[SearchCandidate]:
        scores: dict[int, float] = {}
        payloads: dict[int, SearchCandidate] = {}
        k = get_settings().rrf_k

        for rank_list in rank_lists:
            for position, candidate in enumerate(rank_list, start=1):
                payloads[candidate.chunk_id] = candidate
                scores[candidate.chunk_id] = scores.get(candidate.chunk_id, 0.0) + 1.0 / (k + position)

        ranked_ids = sorted(scores, key=lambda item: scores[item], reverse=True)
        return [payloads[item] for item in ranked_ids]

    @staticmethod
    def _build_location(chunk: KnowledgeChunk) -> str:
        parts = []
        if chunk.page_number:
            parts.append(f"page-{chunk.page_number}")
        if chunk.section_title:
            parts.append(chunk.section_title)
        elif chunk.chapter_title:
            parts.append(chunk.chapter_title)
        if not parts:
            parts.append(f"chunk-{chunk.chunk_index}")
        return " / ".join(parts)


def cosine_similarity(left: Iterable[float], right: Iterable[float]) -> float:
    left_values = [float(item) for item in left]
    right_values = [float(item) for item in right]

    if not left_values or not right_values:
        return -1.0

    limit = min(len(left_values), len(right_values))
    left_values = left_values[:limit]
    right_values = right_values[:limit]

    numerator = sum(a * b for a, b in zip(left_values, right_values, strict=True))
    left_norm = math.sqrt(sum(a * a for a in left_values))
    right_norm = math.sqrt(sum(b * b for b in right_values))
    if left_norm == 0 or right_norm == 0:
        return -1.0
    return numerator / (left_norm * right_norm)


def serialize_documents(documents: list[Document]) -> str:
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
