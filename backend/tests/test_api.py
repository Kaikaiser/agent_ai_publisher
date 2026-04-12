from pathlib import Path

from langchain_core.documents import Document

from app.agents.orchestration.service import AgentOrchestrator
from app.services.image_ocr import ImageOCRService


def test_api_flow_login_import_chat_history_memory_and_projects(client, monkeypatch, workspace_tmp_dir):
    login_response = client.post("/api/auth/login", json={"username": "admin", "password": "admin123456"})
    assert login_response.status_code == 200
    token = login_response.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}

    sample_file = Path(workspace_tmp_dir) / "sample.txt"
    sample_file.write_text("Fractions are numbers that represent parts of a whole.", encoding="utf-8")

    import_response = client.post(
        "/api/knowledge/import",
        headers=headers,
        files={"file": ("sample.txt", sample_file.read_bytes(), "text/plain")},
        data={"book_title": "Primary Math", "doc_type": "textbook", "allowed_role": "user"},
    )
    assert import_response.status_code == 200
    assert import_response.json()["status"] == "completed"

    def fake_run(self, question, search_func):
        documents = [
            Document(
                page_content="Fractions are numbers that represent parts of a whole.",
                metadata={
                    "filename": "sample.txt",
                    "book_title": "Primary Math",
                    "doc_type": "textbook",
                    "location": "full-text",
                    "document_id": 1,
                },
            )
        ]
        return {
            "answer": "A fraction represents part of a whole.",
            "documents": documents,
            "grounded": True,
        }

    monkeypatch.setattr(AgentOrchestrator, "run", fake_run)

    chat_response = client.post(
        "/api/chat/ask",
        headers=headers,
        json={
            "question": "Please answer briefly. We are revising Primary Math and need unified terminology.",
            "book_title": "Primary Math",
            "doc_type": "textbook",
        },
    )
    assert chat_response.status_code == 200
    chat_payload = chat_response.json()
    assert chat_payload["grounded"] is True
    assert chat_payload["sources"]
    assert chat_payload["sources"][0]["preview"] == "Fractions are numbers that represent parts of a whole."
    assert chat_payload["session_id"] > 0
    assert chat_payload["book_id"] > 0
    assert chat_payload["project_id"] > 0
    assert chat_payload["decision_mode"] == "strict_knowledge"
    assert chat_payload["intent_type"] == "fact_qa"

    history_response = client.get("/api/conversations", headers=headers)
    assert history_response.status_code == 200
    history_items = history_response.json()["items"]
    assert len(history_items) == 1
    assert history_items[0]["book_titles"] == ["Primary Math"]
    assert history_items[0]["source_count"] == 1
    assert history_items[0]["session_id"] == chat_payload["session_id"]
    assert history_items[0]["book_id"] == chat_payload["book_id"]
    assert history_items[0]["project_id"] == chat_payload["project_id"]

    project_history = client.get(
        f"/api/conversations?project_id={chat_payload['project_id']}",
        headers=headers,
    )
    assert project_history.status_code == 200
    assert len(project_history.json()["items"]) == 1

    user_memories_response = client.get("/api/memories?scope=user", headers=headers)
    assert user_memories_response.status_code == 200
    user_memories = user_memories_response.json()["items"]
    assert any(item["memory_type"] == "preference" for item in user_memories)
    assert any(item["memory_type"] == "task" for item in user_memories)

    session_memories_response = client.get(
        f"/api/memories?scope=session&session_id={chat_payload['session_id']}&book_id={chat_payload['book_id']}",
        headers=headers,
    )
    assert session_memories_response.status_code == 200
    session_memories = session_memories_response.json()["items"]
    assert any(item["scope"] == "session" and item["session_id"] == chat_payload["session_id"] for item in session_memories)

    book_memories_response = client.get(
        f"/api/memories?scope=book&book_id={chat_payload['book_id']}",
        headers=headers,
    )
    assert book_memories_response.status_code == 200
    book_memories = book_memories_response.json()["items"]
    assert any(item["scope"] == "book" and item["book_id"] == chat_payload["book_id"] for item in book_memories)

    project_memories_response = client.get(
        f"/api/memories?scope=project&project_id={chat_payload['project_id']}",
        headers=headers,
    )
    assert project_memories_response.status_code == 200
    project_memories = project_memories_response.json()["items"]
    assert any(item["scope"] == "project" and item["project_id"] == chat_payload["project_id"] for item in project_memories)

    projects_response = client.get("/api/projects", headers=headers)
    assert projects_response.status_code == 200
    projects = projects_response.json()["items"]
    assert any(
        item["id"] == chat_payload["project_id"]
        and item["book_title"] == "Primary Math"
        and item["decision_mode"] == "strict_knowledge"
        for item in projects
    )

    decision_logs_response = client.get(f"/api/decision-logs?book_id={chat_payload['book_id']}", headers=headers)
    assert decision_logs_response.status_code == 200
    decision_logs = decision_logs_response.json()["items"]
    assert len(decision_logs) == 1
    assert decision_logs[0]["book_id"] == chat_payload["book_id"]
    assert decision_logs[0]["decision_mode"] == "strict_knowledge"
    assert "knowledge_retrieval" in decision_logs[0]["selected_tools"]

    follow_up_response = client.post(
        "/api/chat/ask",
        headers=headers,
        json={
            "question": "Continue with the same book.",
            "session_id": chat_payload["session_id"],
            "book_title": "Primary Math",
            "doc_type": "textbook",
        },
    )
    assert follow_up_response.status_code == 200
    assert follow_up_response.json()["session_id"] == chat_payload["session_id"]
    assert follow_up_response.json()["book_id"] == chat_payload["book_id"]
    assert follow_up_response.json()["project_id"] == chat_payload["project_id"]


def test_api_image_recognition_flow(client, monkeypatch):
    login_response = client.post("/api/auth/login", json={"username": "admin", "password": "admin123456"})
    assert login_response.status_code == 200
    token = login_response.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}

    monkeypatch.setattr(
        ImageOCRService,
        "extract_text",
        lambda self, image_bytes, mime_type: "Question extracted from the image: 1 + 1 = ?",
    )

    image_response = client.post(
        "/api/chat/recognize-image",
        headers=headers,
        files={"file": ("question.png", b"fake-image-bytes", "image/png")},
    )
    assert image_response.status_code == 200
    payload = image_response.json()
    assert payload["recognized_text"] == "Question extracted from the image: 1 + 1 = ?"


def test_api_image_ask_flow(client, monkeypatch):
    login_response = client.post("/api/auth/login", json={"username": "admin", "password": "admin123456"})
    assert login_response.status_code == 200
    token = login_response.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}

    monkeypatch.setattr(
        ImageOCRService,
        "extract_text",
        lambda self, image_bytes, mime_type: "Question extracted from the image: 1 + 1 = ?",
    )

    def fake_run(self, question, search_func):
        documents = [
            Document(
                page_content="1 + 1 = 2",
                metadata={
                    "filename": "math.txt",
                    "book_title": "Primary Math",
                    "doc_type": "exercise",
                    "location": "full-text",
                    "document_id": 2,
                },
            )
        ]
        return {"answer": "1 + 1 = 2.", "documents": documents, "grounded": True}

    monkeypatch.setattr(AgentOrchestrator, "run", fake_run)

    image_response = client.post(
        "/api/chat/ask-image",
        headers=headers,
        files={"file": ("question.png", b"fake-image-bytes", "image/png")},
        data={"book_title": "Primary Math", "doc_type": "exercise"},
    )
    assert image_response.status_code == 200
    payload = image_response.json()
    assert payload["recognized_text"] == "Question extracted from the image: 1 + 1 = ?"
    assert payload["answer"] == "1 + 1 = 2."
    assert payload["book_id"] > 0
    assert payload["project_id"] > 0
    assert payload["sources"]
    assert payload["session_id"] > 0


def test_api_knowledge_management_and_history_filters(client, monkeypatch, workspace_tmp_dir):
    login_response = client.post("/api/auth/login", json={"username": "admin", "password": "admin123456"})
    assert login_response.status_code == 200
    token = login_response.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}

    sample_file = Path(workspace_tmp_dir) / "sample.txt"
    sample_file.write_text("Fractions are numbers that represent parts of a whole.", encoding="utf-8")

    import_response = client.post(
        "/api/knowledge/import",
        headers=headers,
        files={"file": ("sample.txt", sample_file.read_bytes(), "text/plain")},
        data={"book_title": "Primary Math", "doc_type": "textbook", "allowed_role": "user"},
    )
    assert import_response.status_code == 200

    documents_response = client.get("/api/knowledge/documents", headers=headers)
    assert documents_response.status_code == 200
    items = documents_response.json()["items"]
    assert len(items) == 1
    assert items[0]["exists_on_disk"] is True

    def fake_run(self, question, search_func):
        documents = [
            Document(
                page_content="Fractions are numbers that represent parts of a whole.",
                metadata={
                    "filename": "sample.txt",
                    "book_title": "Primary Math",
                    "doc_type": "textbook",
                    "location": "full-text",
                    "document_id": items[0]["id"],
                },
            )
        ]
        return {
            "answer": "A fraction represents part of a whole.",
            "documents": documents,
            "grounded": True,
        }

    monkeypatch.setattr(AgentOrchestrator, "run", fake_run)

    chat_response = client.post(
        "/api/chat/ask",
        headers=headers,
        json={"question": "What is a fraction?", "book_title": "Primary Math", "doc_type": "textbook"},
    )
    assert chat_response.status_code == 200
    session_id = chat_response.json()["session_id"]
    book_id = chat_response.json()["book_id"]
    project_id = chat_response.json()["project_id"]

    filtered_history = client.get(
        f"/api/conversations?book_title=Primary Math&grounded=true&session_id={session_id}&book_id={book_id}&project_id={project_id}",
        headers=headers,
    )
    assert filtered_history.status_code == 200
    assert len(filtered_history.json()["items"]) == 1

    reindex_response = client.post("/api/knowledge/reindex", headers=headers)
    assert reindex_response.status_code == 200
    assert reindex_response.json()["documents_indexed"] == 1

    delete_response = client.delete(f"/api/knowledge/documents/{items[0]['id']}", headers=headers)
    assert delete_response.status_code == 200
    assert delete_response.json()["documents_indexed"] == 0

    documents_after_delete = client.get("/api/knowledge/documents", headers=headers)
    assert documents_after_delete.status_code == 200
    assert documents_after_delete.json()["items"] == []


def test_api_clarification_flow_can_resume_same_session(client, monkeypatch):
    login_response = client.post("/api/auth/login", json={"username": "admin", "password": "admin123456"})
    assert login_response.status_code == 200
    token = login_response.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}

    seen_questions = []

    def fake_run(self, question, search_func):
        seen_questions.append(question)
        if len(seen_questions) == 1:
            return {
                "answer": "Please clarify the specific chapter, character, or topic you want to ask about in this book.",
                "documents": [],
                "grounded": False,
                "clarification_needed": True,
                "decision_trace": [],
                "execution_trace": [],
            }

        documents = [
            Document(
                page_content="Fractions are numbers that represent parts of a whole.",
                metadata={
                    "filename": "sample.txt",
                    "book_title": "Primary Math",
                    "doc_type": "textbook",
                    "location": "full-text",
                    "document_id": 1,
                },
            )
        ]
        return {
            "answer": "A fraction represents part of a whole.",
            "documents": documents,
            "grounded": True,
            "clarification_needed": False,
            "decision_trace": [],
            "execution_trace": [],
        }

    monkeypatch.setattr(AgentOrchestrator, "run", fake_run)

    first_response = client.post(
        "/api/chat/ask",
        headers=headers,
        json={"question": "What about this?", "book_title": "Primary Math", "doc_type": "textbook"},
    )
    assert first_response.status_code == 200
    first_payload = first_response.json()
    assert first_payload["clarification_needed"] is True
    assert first_payload["clarification_slot"] == "topic"
    assert first_payload["grounded"] is False

    second_response = client.post(
        "/api/chat/ask",
        headers=headers,
        json={
            "question": "Chapter 1 fractions.",
            "session_id": first_payload["session_id"],
            "book_title": "Primary Math",
            "doc_type": "textbook",
        },
    )
    assert second_response.status_code == 200
    second_payload = second_response.json()
    assert second_payload["clarification_needed"] is False
    assert second_payload["resumed_from_clarification"] is True
    assert second_payload["grounded"] is True
    assert "Original question:" in seen_questions[1]
    assert "Clarification slot:" in seen_questions[1]
    assert "User clarification:" in seen_questions[1]
