import re

from langchain_core.prompts import ChatPromptTemplate

from app.mcp import InProcessMcpToolClient

SYSTEM_PROMPT = """
你是出版社 AI 助手，服务对象主要是编辑和教辅运营人员。
优先使用知识库检索结果回答与图书、教材、教辅内容相关的问题。
如果知识库证据不足，要明确说明“知识库中未找到充分依据”。
如果用户的问题明显是计算表达式，可以直接给出计算结果。
回答尽量简洁，并优先基于检索结果给出引用线索。
""".strip()


class AgentOrchestrator:
    def __init__(self, llm) -> None:
        self.llm = llm

    def run(self, question: str, search_func) -> dict:
        mcp_client = InProcessMcpToolClient(search_func)
        documents = mcp_client.retrieve(question)
        calculator_result = self._maybe_calculate(question, mcp_client)

        if calculator_result is not None and not documents:
            return {
                "answer": f"计算结果：{calculator_result}",
                "documents": [],
                "grounded": False,
            }

        if not documents:
            return {
                "answer": "知识库中未找到充分依据。",
                "documents": [],
                "grounded": False,
            }

        context = "\n\n".join(
            [
                f"来源：{item.metadata.get('filename', '')} | {item.metadata.get('book_title', '')} | {item.metadata.get('doc_type', '')} | {item.metadata.get('location', 'unknown')}\n内容：{item.page_content}"
                for item in documents
            ]
        )
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", SYSTEM_PROMPT),
                ("human", "问题：{question}\n\n知识库上下文：\n{context}"),
            ]
        )
        message = prompt.format_messages(question=question, context=context)
        result = self.llm.invoke(message)
        answer = getattr(result, "content", str(result))
        return {"answer": answer, "documents": documents, "grounded": True}

    def _maybe_calculate(self, question: str, mcp_client: InProcessMcpToolClient):
        expression = question.strip()
        if re.fullmatch(r"[0-9\s\+\-\*/\(\)\.]+", expression):
            try:
                return mcp_client.calculate(expression)
            except ValueError:
                return None
        return None
