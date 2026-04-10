from langchain_core.documents import Document

from app.agents.orchestration.service import AgentOrchestrator
from app.mcp.runtime import InProcessMcpToolClient


class FakeResult:
    def __init__(self, content: str):
        self.content = content


class FakeLLM:
    def invoke(self, message):
        return FakeResult('根据知识库，这是一本三年级教材。')


def test_mcp_tool_client_retrieves_documents():
    def search_func(query: str):
        return [
            Document(
                page_content='适合三年级。',
                metadata={
                    'filename': 'sample.txt',
                    'book_title': '小学数学',
                    'doc_type': 'textbook',
                    'location': 'full-text',
                },
            )
        ]

    client = InProcessMcpToolClient(search_func)
    documents = client.retrieve('适合几年级？')

    assert len(documents) == 1
    assert documents[0].page_content == '适合三年级。'
    assert documents[0].metadata['book_title'] == '小学数学'


def test_agent_orchestrator_uses_mcp_calculator_for_math_only_questions():
    orchestrator = AgentOrchestrator(FakeLLM())

    result = orchestrator.run('2 + 3 * 4', lambda query: [])

    assert result['answer'] == '计算结果：14'
    assert result['documents'] == []
    assert result['grounded'] is False
