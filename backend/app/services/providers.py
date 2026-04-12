from __future__ import annotations

from dataclasses import dataclass

from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from app.core.config import get_settings


@dataclass(frozen=True)
class ProviderConfig:
    api_key: str
    api_key_name: str
    base_url: str


class OpenAICompatibleProvider:
    def __init__(self) -> None:
        self.settings = get_settings()

    def get_llm(self):
        provider = self.settings.llm_provider.lower()
        config = self._get_provider_config(provider)
        model = self._require(self._get_chat_model(provider), self._get_chat_model_name(provider))
        return self._build_chat_client(model=model, api_key=config.api_key, base_url=config.base_url)

    def get_judge_llm(self):
        provider = (self.settings.faithfulness_judge_provider or self.settings.llm_provider).lower()
        config = self._get_provider_config(provider)
        model = self._require(self._get_judge_model(provider), self._get_judge_model_name(provider))
        return self._build_chat_client(model=model, api_key=config.api_key, base_url=config.base_url)

    def get_vision_llm(self):
        provider = self.settings.llm_provider.lower()
        config = self._get_provider_config(provider)
        model = self._require(self._get_vision_model(provider), self._get_vision_model_name(provider))
        return self._build_chat_client(model=model, api_key=config.api_key, base_url=config.base_url)

    def get_embeddings(self):
        provider = self.settings.llm_provider.lower()
        config = self._get_provider_config(provider)
        model = self._require(self._get_embedding_model(provider), self._get_embedding_model_name(provider))
        return self._build_embeddings_client(
            model=model,
            api_key=config.api_key,
            base_url=config.base_url,
            dimensions=self._get_embedding_dimensions(provider),
        )

    def _get_provider_config(self, provider: str) -> ProviderConfig:
        if provider == "ark":
            return ProviderConfig(
                api_key=self._require(self.settings.ark_api_key, "ARK_API_KEY"),
                api_key_name="ARK_API_KEY",
                base_url=self.settings.ark_base_url,
            )
        if provider == "glm":
            return ProviderConfig(
                api_key=self._require(self.settings.glm_api_key, "GLM_API_KEY"),
                api_key_name="GLM_API_KEY",
                base_url=self.settings.glm_base_url,
            )
        return ProviderConfig(
            api_key=self._require(self.settings.openai_api_key, "OPENAI_API_KEY"),
            api_key_name="OPENAI_API_KEY",
            base_url=self.settings.openai_base_url,
        )

    def _get_chat_model(self, provider: str) -> str:
        if provider == "ark":
            return self.settings.ark_chat_model
        if provider == "glm":
            return self.settings.glm_chat_model
        return self.settings.openai_model

    @staticmethod
    def _get_chat_model_name(provider: str) -> str:
        if provider == "ark":
            return "ARK_CHAT_MODEL"
        if provider == "glm":
            return "GLM_CHAT_MODEL"
        return "OPENAI_MODEL"

    def _get_judge_model(self, provider: str) -> str:
        if provider == "ark":
            return self.settings.faithfulness_judge_model or self.settings.ark_chat_model
        if provider == "glm":
            return self.settings.faithfulness_judge_model or self.settings.glm_chat_model
        return self.settings.faithfulness_judge_model or self.settings.openai_model

    @staticmethod
    def _get_judge_model_name(provider: str) -> str:
        if provider == "ark":
            return "FAITHFULNESS_JUDGE_MODEL or ARK_CHAT_MODEL"
        if provider == "glm":
            return "FAITHFULNESS_JUDGE_MODEL or GLM_CHAT_MODEL"
        return "FAITHFULNESS_JUDGE_MODEL or OPENAI_MODEL"

    def _get_vision_model(self, provider: str) -> str:
        if provider == "ark":
            return self.settings.ark_vision_model or self.settings.ark_chat_model
        if provider == "glm":
            return self.settings.glm_vision_model or self.settings.glm_chat_model
        return self.settings.openai_vision_model or self.settings.openai_model

    @staticmethod
    def _get_vision_model_name(provider: str) -> str:
        if provider == "ark":
            return "ARK_VISION_MODEL or ARK_CHAT_MODEL"
        if provider == "glm":
            return "GLM_VISION_MODEL or GLM_CHAT_MODEL"
        return "OPENAI_VISION_MODEL or OPENAI_MODEL"

    def _get_embedding_model(self, provider: str) -> str:
        if provider == "ark":
            return self.settings.ark_embedding_model
        if provider == "glm":
            return self.settings.glm_embedding_model
        return self.settings.openai_embedding_model

    @staticmethod
    def _get_embedding_model_name(provider: str) -> str:
        if provider == "ark":
            return "ARK_EMBEDDING_MODEL"
        if provider == "glm":
            return "GLM_EMBEDDING_MODEL"
        return "OPENAI_EMBEDDING_MODEL"

    def _get_embedding_dimensions(self, provider: str) -> int | None:
        if provider == "ark":
            return self.settings.ark_embedding_dimensions or self.settings.embedding_dimensions
        if provider == "glm":
            return self.settings.glm_embedding_dimensions or self.settings.embedding_dimensions
        return self.settings.openai_embedding_dimensions or self.settings.embedding_dimensions

    @staticmethod
    def _build_chat_client(model: str, api_key: str, base_url: str):
        kwargs = {
            "model": model,
            "api_key": api_key,
            "temperature": 0,
        }
        if base_url:
            kwargs["base_url"] = base_url
        return ChatOpenAI(**kwargs)

    @staticmethod
    def _build_embeddings_client(model: str, api_key: str, base_url: str, dimensions: int | None = None):
        kwargs = {
            "model": model,
            "api_key": api_key,
        }
        if base_url:
            kwargs["base_url"] = base_url
        if dimensions:
            kwargs["dimensions"] = dimensions
        return OpenAIEmbeddings(**kwargs)

    @staticmethod
    def _require(value: str, field_name: str) -> str:
        if not value:
            raise ValueError(f"当前服务尚未配置 {field_name}")
        return value
