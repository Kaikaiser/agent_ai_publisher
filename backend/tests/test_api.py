from langchain_core.documents import Document

from app.agents.orchestration.service import AgentOrchestrator
from app.services.image_ocr import ImageOCRService


def test_api_flow_login_import_chat_history(client, monkeypatch, tmp_path):
    login_response = client.post('/api/auth/login', json={'username': 'admin', 'password': 'admin123456'})
    assert login_response.status_code == 200
    token = login_response.json()['access_token']
    headers = {'Authorization': f'Bearer {token}'}

    sample_file = tmp_path / 'sample.txt'
    sample_file.write_text('这是一本小学数学教辅，适合三年级。', encoding='utf-8')

    import_response = client.post(
        '/api/knowledge/import',
        headers=headers,
        files={'file': ('sample.txt', sample_file.read_bytes(), 'text/plain')},
        data={'book_title': '小学数学', 'doc_type': 'textbook', 'allowed_role': 'user'},
    )
    assert import_response.status_code == 200
    assert import_response.json()['status'] == 'completed'

    def fake_run(self, question, search_func):
        documents = [Document(page_content='适合三年级。', metadata={'filename': 'sample.txt', 'book_title': '小学数学', 'doc_type': 'textbook', 'location': 'full-text', 'document_id': 1})]
        return {'answer': '根据知识库，这本教辅适合三年级。', 'documents': documents, 'grounded': True}

    monkeypatch.setattr(AgentOrchestrator, 'run', fake_run)

    chat_response = client.post('/api/chat/ask', headers=headers, json={'question': '适合几年级？', 'book_title': '小学数学'})
    assert chat_response.status_code == 200
    assert chat_response.json()['grounded'] is True
    assert chat_response.json()['sources']
    assert chat_response.json()['sources'][0]['preview'] == '适合三年级。'

    history_response = client.get('/api/conversations', headers=headers)
    assert history_response.status_code == 200
    assert len(history_response.json()['items']) == 1
    assert history_response.json()['items'][0]['book_titles'] == ['小学数学']
    assert history_response.json()['items'][0]['source_count'] == 1


def test_api_image_recognition_flow(client, monkeypatch):
    login_response = client.post('/api/auth/login', json={'username': 'admin', 'password': 'admin123456'})
    assert login_response.status_code == 200
    token = login_response.json()['access_token']
    headers = {'Authorization': f'Bearer {token}'}

    monkeypatch.setattr(ImageOCRService, 'extract_text', lambda self, image_bytes, mime_type: '已识别题目：1+1等于几？')

    image_response = client.post(
        '/api/chat/recognize-image',
        headers=headers,
        files={'file': ('question.png', b'fake-image-bytes', 'image/png')},
    )
    assert image_response.status_code == 200
    payload = image_response.json()
    assert payload['recognized_text'] == '已识别题目：1+1等于几？'


def test_api_image_ask_flow(client, monkeypatch):
    login_response = client.post('/api/auth/login', json={'username': 'admin', 'password': 'admin123456'})
    assert login_response.status_code == 200
    token = login_response.json()['access_token']
    headers = {'Authorization': f'Bearer {token}'}

    monkeypatch.setattr(ImageOCRService, 'extract_text', lambda self, image_bytes, mime_type: '已识别题目：1+1等于几？')

    def fake_run(self, question, search_func):
        documents = [Document(page_content='1+1=2。', metadata={'filename': 'math.txt', 'book_title': '基础数学', 'doc_type': 'exercise', 'location': 'full-text', 'document_id': 2})]
        return {'answer': '根据识别结果，答案是 2。', 'documents': documents, 'grounded': True}

    monkeypatch.setattr(AgentOrchestrator, 'run', fake_run)

    image_response = client.post(
        '/api/chat/ask-image',
        headers=headers,
        files={'file': ('question.png', b'fake-image-bytes', 'image/png')},
        data={'book_title': '基础数学', 'doc_type': 'exercise'},
    )
    assert image_response.status_code == 200
    payload = image_response.json()
    assert payload['recognized_text'] == '已识别题目：1+1等于几？'
    assert payload['answer'] == '根据识别结果，答案是 2。'
    assert payload['sources']


def test_api_knowledge_management_and_history_filters(client, monkeypatch, tmp_path):
    login_response = client.post('/api/auth/login', json={'username': 'admin', 'password': 'admin123456'})
    assert login_response.status_code == 200
    token = login_response.json()['access_token']
    headers = {'Authorization': f'Bearer {token}'}

    sample_file = tmp_path / 'sample.txt'
    sample_file.write_text('这是一本小学数学教辅，适合三年级。', encoding='utf-8')

    import_response = client.post(
        '/api/knowledge/import',
        headers=headers,
        files={'file': ('sample.txt', sample_file.read_bytes(), 'text/plain')},
        data={'book_title': '小学数学', 'doc_type': 'textbook', 'allowed_role': 'user'},
    )
    assert import_response.status_code == 200

    documents_response = client.get('/api/knowledge/documents', headers=headers)
    assert documents_response.status_code == 200
    items = documents_response.json()['items']
    assert len(items) == 1
    assert items[0]['exists_on_disk'] is True

    def fake_run(self, question, search_func):
        documents = [Document(page_content='适合三年级。', metadata={'filename': 'sample.txt', 'book_title': '小学数学', 'doc_type': 'textbook', 'location': 'full-text', 'document_id': items[0]['id']})]
        return {'answer': '根据知识库，这本教辅适合三年级。', 'documents': documents, 'grounded': True}

    monkeypatch.setattr(AgentOrchestrator, 'run', fake_run)

    chat_response = client.post('/api/chat/ask', headers=headers, json={'question': '适合几年级？', 'book_title': '小学数学'})
    assert chat_response.status_code == 200

    filtered_history = client.get('/api/conversations?book_title=小学数学&grounded=true', headers=headers)
    assert filtered_history.status_code == 200
    assert len(filtered_history.json()['items']) == 1

    reindex_response = client.post('/api/knowledge/reindex', headers=headers)
    assert reindex_response.status_code == 200
    assert reindex_response.json()['documents_indexed'] == 1

    delete_response = client.delete(f"/api/knowledge/documents/{items[0]['id']}", headers=headers)
    assert delete_response.status_code == 200
    assert delete_response.json()['documents_indexed'] == 0

    documents_after_delete = client.get('/api/knowledge/documents', headers=headers)
    assert documents_after_delete.status_code == 200
    assert documents_after_delete.json()['items'] == []
