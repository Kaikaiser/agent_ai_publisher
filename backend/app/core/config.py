from functools import lru_cache
from pathlib import Path
from typing import List

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


PROJECT_ROOT = Path(__file__).resolve().parents[3]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / '.env',
        env_file_encoding='utf-8',
        extra='ignore',
    )

    llm_provider: str = 'openai'

    openai_api_key: str = ''
    openai_base_url: str = ''
    openai_model: str = 'gpt-4o-mini'
    openai_vision_model: str = ''
    openai_embedding_model: str = 'text-embedding-3-small'
    openai_embedding_dimensions: int | None = None

    ark_api_key: str = ''
    ark_base_url: str = 'https://ark.cn-beijing.volces.com/api/v3'
    ark_chat_model: str = ''
    ark_vision_model: str = ''
    ark_embedding_model: str = ''
    ark_embedding_dimensions: int | None = None

    glm_api_key: str = ''
    glm_base_url: str = 'https://open.bigmodel.cn/api/paas/v4/'
    glm_chat_model: str = 'glm-5'
    glm_vision_model: str = ''
    glm_embedding_model: str = ''
    glm_embedding_dimensions: int | None = None

    database_url: str = 'postgresql+psycopg://ai_assistant:ai_assistant@127.0.0.1:5432/ai_assistant'
    elasticsearch_url: str = 'http://127.0.0.1:9200'
    elasticsearch_index_name: str = 'knowledge_chunks'
    embedding_dimensions: int = 1024
    pdf_extractor: str = 'auto'
    use_pymupdf4llm: bool = True
    enable_rerank: bool = True
    zhipu_rerank_api_key: str = ''
    zhipu_rerank_base_url: str = 'https://open.bigmodel.cn/api/paas/v4'
    zhipu_rerank_model: str = 'rerank-3'
    rerank_top_n: int = 8
    enable_faithfulness_eval: bool = True
    faithfulness_judge_provider: str = ''
    faithfulness_judge_model: str = ''
    faithfulness_dataset_path: str = 'backend/data/evals/faithfulness_dataset.jsonl'
    faithfulness_report_path: str = 'backend/data/evals/faithfulness_report.json'
    hybrid_dense_top_k: int = 24
    hybrid_bm25_top_k: int = 24
    hybrid_final_top_k: int = 8
    rrf_k: int = 60
    jwt_secret_key: str = 'change-me'
    jwt_algorithm: str = 'HS256'
    jwt_expire_minutes: int = 1440
    default_admin_username: str = 'admin'
    default_admin_password: str = 'admin123456'
    langsmith_api_key: str = ''
    langsmith_tracing: bool = False
    upload_dir: str = 'data/uploads'
    backend_cors_origins: List[str] = Field(default_factory=lambda: ['http://localhost:5173'])

    @field_validator('database_url', mode='before')
    @classmethod
    def resolve_database_url(cls, value: str) -> str:
        prefix = 'sqlite:///'
        if not isinstance(value, str) or not value.startswith(prefix):
            return value

        raw_path = value[len(prefix):]
        if raw_path == ':memory:':
            return value

        db_path = Path(raw_path)
        if not db_path.is_absolute():
            db_path = (PROJECT_ROOT / db_path).resolve()
        return f'{prefix}{db_path.as_posix()}'

    @field_validator('upload_dir', 'faithfulness_dataset_path', 'faithfulness_report_path', mode='before')
    @classmethod
    def resolve_runtime_dir(cls, value: str) -> str:
        if not isinstance(value, str):
            return value

        path = Path(value)
        if not path.is_absolute():
            path = (PROJECT_ROOT / path).resolve()
        return str(path)

    @field_validator('backend_cors_origins', mode='before')
    @classmethod
    def parse_cors(cls, value):
        if isinstance(value, str):
            if value.startswith('[') and value.endswith(']'):
                inner = value[1:-1].strip()
                if not inner:
                    return []
                return [item.strip().strip('"').strip("'") for item in inner.split(',') if item.strip()]
            return [item.strip() for item in value.split(',') if item.strip()]
        return value

    @property
    def judge_model_name(self) -> str:
        return self.faithfulness_judge_model or self.glm_chat_model or self.openai_model


@lru_cache
def get_settings() -> Settings:
    return Settings()
