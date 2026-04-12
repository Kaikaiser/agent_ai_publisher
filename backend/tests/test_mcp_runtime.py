from langchain_core.documents import Document

from app.agents.orchestration.service import AgentOrchestrator
from app.mcp.runtime import InProcessMcpToolClient


class FakeResult:
    def __init__(self, content: str):
        self.content = content


class FakeLLM:
    def invoke(self, message):
        return FakeResult("根据知识库，这是一本三年级教材。")


class CapturingLLM:
    def __init__(self):
        self.last_message = None

    def invoke(self, message):
        self.last_message = message
        return FakeResult("ok")


def test_mcp_tool_client_retrieves_documents():
    def search_func(query: str):
        return [
            Document(
                page_content="适合三年级。",
                metadata={
                    "filename": "sample.txt",
                    "book_title": "小学数学",
                    "doc_type": "textbook",
                    "location": "full-text",
                },
            )
        ]

    client = InProcessMcpToolClient(search_func)
    documents = client.retrieve("适合几年级？")

    assert len(documents) == 1
    assert documents[0].page_content == "适合三年级。"
    assert documents[0].metadata["book_title"] == "小学数学"


def test_agent_orchestrator_uses_mcp_calculator_for_math_only_questions():
    orchestrator = AgentOrchestrator(FakeLLM())

    result = orchestrator.run("2 + 3 * 4", lambda query: [])

    assert result["answer"] == "计算结果：14"
    assert result["documents"] == []
    assert result["grounded"] is False


def test_agent_orchestrator_includes_memory_context_when_available():
    llm = CapturingLLM()
    orchestrator = AgentOrchestrator(
        llm,
        memory_search_func=lambda query: [
            type(
                "MemorySnippetStub",
                (),
                {
                    "id": 1,
                    "scope": "user",
                    "memory_type": "preference",
                    "summary": "用户偏好简洁回答",
                    "content": "回答要简洁",
                },
            )()
        ],
    )

    result = orchestrator.run(
        "适合几年级？",
        lambda query: [
            Document(
                page_content="适合三年级。",
                metadata={"filename": "sample.txt", "book_title": "小学数学", "doc_type": "textbook", "location": "page-1"},
            )
        ],
    )

    assert result["grounded"] is True
    flattened = "\n".join(getattr(message, "content", "") for message in llm.last_message if hasattr(message, "content"))
    assert "用户偏好简洁回答" in flattened


def test_agent_orchestrator_uses_observation_to_choose_second_action():
    llm = CapturingLLM()
    orchestrator = AgentOrchestrator(
        llm,
        memory_search_func=lambda query: [
            type(
                "MemorySnippetStub",
                (),
                {
                    "id": 1,
                    "scope": "book",
                    "memory_type": "rule",
                    "summary": "赫敏口吻",
                    "content": "保持聪明且克制的表达风格。",
                },
            )()
        ],
    )
    orchestrator.response_policy = {
        "intent_type": "character_chat",
        "route_name": "knowledge_plus_persona",
        "selected_tools": ["memory_retrieval", "knowledge_retrieval"],
        "fallback_policy": "conservative_answer",
        "memory_scopes": ["session", "book", "project", "user"],
        "clarification_needed": False,
        "decision_mode": "immersive_character",
        "max_steps": 2,
    }

    result = orchestrator.run(
        "Pretend you are Hermione and explain potion class.",
        lambda query: [
            Document(
                page_content="Potions class is taught in the dungeon classroom.",
                metadata={"filename": "hp.txt", "book_title": "Harry Potter", "doc_type": "novel", "location": "chapter-8"},
            )
        ],
    )

    assert result["grounded"] is True
    assert len(result["decision_trace"]) >= 2
    assert len(result["execution_trace"]) == 2
    assert "memory_retrieval" in result["execution_trace"][0]
    assert "knowledge_retrieval" in result["execution_trace"][1]
    flattened = "\n".join(getattr(message, "content", "") for message in llm.last_message if hasattr(message, "content"))
    assert "Thought trace" in flattened
    assert "next_action=knowledge_retrieval" in flattened


def test_agent_orchestrator_retries_with_rewritten_query_once():
    llm = CapturingLLM()
    orchestrator = AgentOrchestrator(llm)
    orchestrator.response_policy = {
        "intent_type": "fact_qa",
        "route_name": "knowledge_answer",
        "selected_tools": ["knowledge_retrieval"],
        "fallback_policy": "conservative_answer",
        "memory_scopes": ["session", "book", "project", "user"],
        "clarification_needed": False,
        "decision_mode": "strict_knowledge",
        "allow_query_retry": True,
        "max_steps": 2,
    }

    seen_queries = []

    def search_func(query: str):
        seen_queries.append(query)
        if len(seen_queries) == 1:
            return []
        return [
            Document(
                page_content="Fractions are numbers that represent parts of a whole.",
                metadata={"filename": "sample.txt", "book_title": "Primary Math", "doc_type": "textbook", "location": "page-2"},
            )
        ]

    result = orchestrator.run("Please briefly explain this book fraction concept.", search_func)

    assert result["grounded"] is True
    assert len(seen_queries) == 2
    assert seen_queries[0] == "Please briefly explain this book fraction concept."
    assert seen_queries[1] == "explain fraction concept."
    assert "retry_knowledge_retrieval" in result["execution_trace"][1]


def test_agent_orchestrator_marks_weak_evidence_and_clarifies():
    llm = CapturingLLM()
    orchestrator = AgentOrchestrator(llm)
    orchestrator.response_policy = {
        "intent_type": "fact_qa",
        "route_name": "knowledge_answer",
        "selected_tools": ["knowledge_retrieval"],
        "fallback_policy": "conservative_answer",
        "memory_scopes": ["session", "book", "project", "user"],
        "clarification_needed": True,
        "clarification_prompt": "Please clarify the exact topic in this book.",
        "decision_mode": "strict_knowledge",
        "allow_query_retry": False,
        "max_steps": 2,
    }

    result = orchestrator.run(
        "What about this?",
        lambda query: [
            Document(
                page_content="Short clue.",
                metadata={"filename": "sample.txt", "book_title": "Primary Math", "doc_type": "textbook", "location": "page-1"},
            )
        ],
    )

    assert result["grounded"] is False
    assert result["clarification_needed"] is True
    assert result["evidence_quality"] == "weak"
