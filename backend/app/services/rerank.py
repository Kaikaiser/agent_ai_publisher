from __future__ import annotations

import logging

import httpx
from langchain_core.documents import Document

from app.core.config import get_settings


logger = logging.getLogger(__name__)


class ZhipuRerankService:
    def __init__(self) -> None:
        self.settings = get_settings()

    def is_enabled(self) -> bool:
        return self.settings.enable_rerank and bool(self.settings.zhipu_rerank_api_key)

    def rerank(self, query: str, documents: list[Document], top_n: int | None = None) -> list[Document]:
        if len(documents) <= 1 or not self.is_enabled():
            return documents

        try:
            ranked_indices = self._request_rerank(query=query, documents=documents, top_n=top_n or len(documents))
        except Exception as exc:  # pragma: no cover - depends on external API runtime.
            logger.warning("Zhipu rerank failed, keeping hybrid retrieval order: %s", exc)
            return documents

        ranked = [documents[index] for index in ranked_indices if 0 <= index < len(documents)]
        if not ranked:
            return documents

        seen = {id(item) for item in ranked}
        ranked.extend(item for item in documents if id(item) not in seen)
        return ranked

    def _request_rerank(self, query: str, documents: list[Document], top_n: int) -> list[int]:
        url = self._build_endpoint()
        payload = {
            "model": self.settings.zhipu_rerank_model,
            "query": query,
            "documents": [item.page_content for item in documents],
            "top_n": min(top_n, len(documents)),
        }
        headers = {
            "Authorization": f"Bearer {self.settings.zhipu_rerank_api_key}",
            "Content-Type": "application/json",
        }

        with httpx.Client(timeout=30.0) as client:
            response = client.post(url, json=payload, headers=headers)
            response.raise_for_status()
            data = response.json()

        results = data.get("results")
        if results is None and isinstance(data.get("data"), dict):
            results = data["data"].get("results")
        if results is None and isinstance(data.get("data"), list):
            results = data["data"]
        if not isinstance(results, list):
            raise ValueError("Unexpected rerank response payload.")

        ranked_pairs = []
        for item in results:
            if not isinstance(item, dict):
                continue
            index = item.get("index")
            if index is None:
                continue
            score = float(item.get("relevance_score") or item.get("score") or 0.0)
            ranked_pairs.append((score, int(index)))

        if not ranked_pairs:
            raise ValueError("Rerank response did not include usable indices.")

        ranked_pairs.sort(key=lambda item: item[0], reverse=True)
        return [index for _, index in ranked_pairs]

    def _build_endpoint(self) -> str:
        base_url = self.settings.zhipu_rerank_base_url.rstrip("/")
        if base_url.endswith("/rerank"):
            return base_url
        return f"{base_url}/rerank"
