from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from app.core.config import get_settings


class OpenAICompatibleProvider:
    def __init__(self) -> None:
        self.settings = get_settings()

    def get_llm(self):
        provider = self.settings.llm_provider.lower()

        if provider == 'ark':
            api_key = self._require(self.settings.ark_api_key, 'ARK_API_KEY')
            model = self._require(self.settings.ark_chat_model, 'ARK_CHAT_MODEL')
            return self._build_chat_client(model=model, api_key=api_key, base_url=self.settings.ark_base_url)

        if provider == 'glm':
            api_key = self._require(self.settings.glm_api_key, 'GLM_API_KEY')
            model = self._require(self.settings.glm_chat_model, 'GLM_CHAT_MODEL')
            return self._build_chat_client(model=model, api_key=api_key, base_url=self.settings.glm_base_url)

        api_key = self._require(self.settings.openai_api_key, 'OPENAI_API_KEY')
        model = self._require(self.settings.openai_model, 'OPENAI_MODEL')
        return self._build_chat_client(model=model, api_key=api_key, base_url=self.settings.openai_base_url)

    def get_judge_llm(self):
        provider = (self.settings.faithfulness_judge_provider or self.settings.llm_provider).lower()

        if provider == 'ark':
            api_key = self._require(self.settings.ark_api_key, 'ARK_API_KEY')
            model = self.settings.faithfulness_judge_model or self.settings.ark_chat_model
            model = self._require(model, 'FAITHFULNESS_JUDGE_MODEL or ARK_CHAT_MODEL')
            return self._build_chat_client(model=model, api_key=api_key, base_url=self.settings.ark_base_url)

        if provider == 'glm':
            api_key = self._require(self.settings.glm_api_key, 'GLM_API_KEY')
            model = self.settings.faithfulness_judge_model or self.settings.glm_chat_model
            model = self._require(model, 'FAITHFULNESS_JUDGE_MODEL or GLM_CHAT_MODEL')
            return self._build_chat_client(model=model, api_key=api_key, base_url=self.settings.glm_base_url)

        api_key = self._require(self.settings.openai_api_key, 'OPENAI_API_KEY')
        model = self.settings.faithfulness_judge_model or self.settings.openai_model
        model = self._require(model, 'FAITHFULNESS_JUDGE_MODEL or OPENAI_MODEL')
        return self._build_chat_client(model=model, api_key=api_key, base_url=self.settings.openai_base_url)

    def get_vision_llm(self):
        provider = self.settings.llm_provider.lower()

        if provider == 'ark':
            api_key = self._require(self.settings.ark_api_key, 'ARK_API_KEY')
            # OCR can reuse the chat endpoint when the selected endpoint itself supports vision.
            model = self.settings.ark_vision_model or self.settings.ark_chat_model
            model = self._require(model, 'ARK_VISION_MODEL or ARK_CHAT_MODEL')
            return self._build_chat_client(model=model, api_key=api_key, base_url=self.settings.ark_base_url)

        if provider == 'glm':
            api_key = self._require(self.settings.glm_api_key, 'GLM_API_KEY')
            # GLM also exposes OpenAI-compatible multimodal endpoints, so vision can fall back to chat.
            model = self.settings.glm_vision_model or self.settings.glm_chat_model
            model = self._require(model, 'GLM_VISION_MODEL or GLM_CHAT_MODEL')
            return self._build_chat_client(model=model, api_key=api_key, base_url=self.settings.glm_base_url)

        api_key = self._require(self.settings.openai_api_key, 'OPENAI_API_KEY')
        model = self.settings.openai_vision_model or self.settings.openai_model
        model = self._require(model, 'OPENAI_VISION_MODEL or OPENAI_MODEL')
        return self._build_chat_client(model=model, api_key=api_key, base_url=self.settings.openai_base_url)

    def get_embeddings(self):
        provider = self.settings.llm_provider.lower()

        if provider == 'ark':
            api_key = self._require(self.settings.ark_api_key, 'ARK_API_KEY')
            model = self._require(self.settings.ark_embedding_model, 'ARK_EMBEDDING_MODEL')
            return self._build_embeddings_client(
                model=model,
                api_key=api_key,
                base_url=self.settings.ark_base_url,
                dimensions=self.settings.ark_embedding_dimensions or self.settings.embedding_dimensions,
            )

        if provider == 'glm':
            api_key = self._require(self.settings.glm_api_key, 'GLM_API_KEY')
            model = self._require(self.settings.glm_embedding_model, 'GLM_EMBEDDING_MODEL')
            return self._build_embeddings_client(
                model=model,
                api_key=api_key,
                base_url=self.settings.glm_base_url,
                dimensions=self.settings.glm_embedding_dimensions or self.settings.embedding_dimensions,
            )

        api_key = self._require(self.settings.openai_api_key, 'OPENAI_API_KEY')
        model = self._require(self.settings.openai_embedding_model, 'OPENAI_EMBEDDING_MODEL')
        return self._build_embeddings_client(
            model=model,
            api_key=api_key,
            base_url=self.settings.openai_base_url,
            dimensions=self.settings.openai_embedding_dimensions or self.settings.embedding_dimensions,
        )

    @staticmethod
    def _build_chat_client(model: str, api_key: str, base_url: str):
        kwargs = {
            'model': model,
            'api_key': api_key,
            'temperature': 0,
        }
        if base_url:
            kwargs['base_url'] = base_url
        return ChatOpenAI(**kwargs)

    @staticmethod
    def _build_embeddings_client(model: str, api_key: str, base_url: str, dimensions: int | None = None):
        kwargs = {
            'model': model,
            'api_key': api_key,
        }
        if base_url:
            kwargs['base_url'] = base_url
        if dimensions:
            kwargs['dimensions'] = dimensions
        return OpenAIEmbeddings(**kwargs)

    @staticmethod
    def _require(value: str, field_name: str) -> str:
        if not value:
            raise ValueError(f'当前服务尚未配置 {field_name}')
        return value
