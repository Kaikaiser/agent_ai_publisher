import os
import shutil
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from langchain_core.embeddings import Embeddings

TEST_ROOT = Path(__file__).resolve().parents[1]
TEST_DB_PATH = TEST_ROOT / "data" / "test_app.db"
TEST_UPLOAD_DIR = TEST_ROOT / "data" / "test_uploads"
TEST_TMP_DIR = TEST_ROOT / "data" / "test_tmp"
TEST_DATASET_PATH = TEST_ROOT / "data" / "evals" / "faithfulness_dataset.jsonl"
TEST_REPORT_PATH = TEST_ROOT / "data" / "evals" / "faithfulness_report.json"

os.environ["DATABASE_URL"] = f"sqlite:///{TEST_DB_PATH.as_posix()}"
os.environ["ELASTICSEARCH_URL"] = ""
os.environ["REDIS_URL"] = ""
os.environ["ENABLE_RERANK"] = "false"
os.environ["USE_PYMUPDF4LLM"] = "false"
os.environ["EMBEDDING_DIMENSIONS"] = "8"
os.environ["UPLOAD_DIR"] = str(TEST_UPLOAD_DIR)
os.environ["FAITHFULNESS_DATASET_PATH"] = str(TEST_DATASET_PATH)
os.environ["FAITHFULNESS_REPORT_PATH"] = str(TEST_REPORT_PATH)

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
        dimensions = get_settings().embedding_dimensions
        base = [float((len(text) + index) % 11) for index in range(dimensions)]
        if dimensions >= 2:
            base[1] = float(sum(ord(char) for char in text) % 97)
        if dimensions >= 3:
            base[2] = 1.0
        return base


class DummyLLM:
    pass


@pytest.fixture(autouse=True)
def reset_state():
    settings = get_settings()
    upload_dir = Path(settings.upload_dir)
    report_path = Path(settings.faithfulness_report_path)

    for directory in [upload_dir, report_path.parent, TEST_TMP_DIR]:
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


@pytest.fixture
def workspace_tmp_dir():
    directory = TEST_TMP_DIR / "case"
    if directory.exists():
        shutil.rmtree(directory)
    directory.mkdir(parents=True, exist_ok=True)
    return directory
