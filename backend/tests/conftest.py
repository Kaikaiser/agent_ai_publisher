import shutil
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from langchain_core.embeddings import Embeddings

from app.api.deps import get_embeddings, get_llm, get_vision_llm
from app.core.config import get_settings
from app.db.base import Base
from app.db.init_db import bootstrap_admin, init_db
from app.db.session import engine
from app.main import app


class FakeEmbeddings(Embeddings):
    def embed_documents(self, texts):
        return [self._vectorize(text) for text in texts]

    def embed_query(self, text):
        return self._vectorize(text)

    def _vectorize(self, text):
        return [float(len(text) % 10), float(sum(ord(char) for char in text) % 97), 1.0]


class DummyLLM:
    pass


@pytest.fixture(autouse=True)
def reset_state():
    settings = get_settings()
    vector_dir = Path(settings.vector_store_dir)
    upload_dir = Path(settings.upload_dir)

    for directory in [vector_dir, upload_dir]:
        if directory.exists():
            shutil.rmtree(directory)
        directory.mkdir(parents=True, exist_ok=True)

    Base.metadata.drop_all(bind=engine)
    init_db()
    bootstrap_admin()
    yield
    app.dependency_overrides = {}


@pytest.fixture
def client():
    app.dependency_overrides[get_embeddings] = lambda: FakeEmbeddings()
    app.dependency_overrides[get_llm] = lambda: DummyLLM()
    app.dependency_overrides[get_vision_llm] = lambda: DummyLLM()
    with TestClient(app) as test_client:
        yield test_client
