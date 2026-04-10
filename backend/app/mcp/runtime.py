from __future__ import annotations

from collections.abc import Callable

import anyio
import httpx
from langchain_core.documents import Document
from pydantic import BaseModel, Field

from app.knowledge.vector_store import serialize_documents
from app.mcp.bootstrap import ensure_mcp_vendor_path
from app.tools.calculator import evaluate_expression

ensure_mcp_vendor_path()

from mcp.client.session import ClientSession
from mcp.client.streamable_http import streamable_http_client
from mcp.server.fastmcp import FastMCP


MCP_BASE_URL = "http://127.0.0.1:8000"
MCP_PATH = "/mcp"


class RetrievedDocumentPayload(BaseModel):
    content: str
    filename: str = ""
    book_title: str = ""
    doc_type: str = ""
    location: str = "unknown"

    @classmethod
    def from_document(cls, document: Document) -> "RetrievedDocumentPayload":
        return cls(
            content=document.page_content,
            filename=document.metadata.get("filename", ""),
            book_title=document.metadata.get("book_title", ""),
            doc_type=document.metadata.get("doc_type", ""),
            location=document.metadata.get("location", "unknown"),
        )

    def to_document(self) -> Document:
        return Document(
            page_content=self.content,
            metadata={
                "filename": self.filename,
                "book_title": self.book_title,
                "doc_type": self.doc_type,
                "location": self.location,
            },
        )


class RetrievalResultPayload(BaseModel):
    serialized_documents: str = ""
    documents: list[RetrievedDocumentPayload] = Field(default_factory=list)


class CalculatorResultPayload(BaseModel):
    result: str


def build_mcp_tool_server(search_func: Callable[[str], list[Document]]) -> FastMCP:
    server = FastMCP(
        name="publisher-ai-tools",
        instructions="Internal MCP tool server for retrieval and calculator tools.",
        host="127.0.0.1",
        port=8000,
        streamable_http_path=MCP_PATH,
        log_level="CRITICAL",
    )

    @server.tool(
        name="knowledge_retriever",
        description="Searches the publishing knowledge base for relevant passages.",
        structured_output=True,
    )
    def knowledge_retriever(query: str) -> RetrievalResultPayload:
        documents = search_func(query)
        return RetrievalResultPayload(
            serialized_documents=serialize_documents(documents),
            documents=[RetrievedDocumentPayload.from_document(item) for item in documents],
        )

    @server.tool(
        name="calculator",
        description="Evaluates basic arithmetic expressions.",
        structured_output=True,
    )
    def calculator(expression: str) -> CalculatorResultPayload:
        return CalculatorResultPayload(result=evaluate_expression(expression))

    return server


class InProcessMcpToolClient:
    def __init__(self, search_func: Callable[[str], list[Document]]) -> None:
        self._search_func = search_func

    def retrieve(self, query: str) -> list[Document]:
        payload = anyio.run(self._call_tool, "knowledge_retriever", {"query": query}, RetrievalResultPayload)
        return [item.to_document() for item in payload.documents]

    def calculate(self, expression: str) -> str:
        payload = anyio.run(self._call_tool, "calculator", {"expression": expression}, CalculatorResultPayload)
        return payload.result

    async def _call_tool(self, name: str, arguments: dict, result_model: type[BaseModel]) -> BaseModel:
        app = build_mcp_tool_server(self._search_func).streamable_http_app()
        async with app.router.lifespan_context(app):
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(transport=transport, base_url=MCP_BASE_URL) as http_client:
                async with streamable_http_client(f"{MCP_BASE_URL}{MCP_PATH}", http_client=http_client) as (read, write, _):
                    async with ClientSession(read, write) as session:
                        await session.initialize()
                        result = await session.call_tool(name, arguments)

        if result.isError:
            message = "\n".join(item.text for item in result.content if hasattr(item, "text")) or f"MCP tool {name} failed"
            raise ValueError(message)

        if result.structuredContent is None:
            raise ValueError(f"MCP tool {name} returned no structured content")

        return result_model.model_validate(result.structuredContent)
