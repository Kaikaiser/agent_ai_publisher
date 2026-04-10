from typing import Dict, List, Optional, Tuple

from langchain.tools import StructuredTool
from langchain_core.documents import Document
from pydantic import BaseModel

from app.knowledge.vector_store import serialize_documents
from app.tools.calculator import evaluate_expression


class RetrievalInput(BaseModel):
    query: str


class CalculatorInput(BaseModel):
    expression: str


class RetrievalToolFactory:
    def __init__(self, search_func):
        self.search_func = search_func
        self.last_documents: List[Document] = []

    def build(self) -> StructuredTool:
        def run(query: str) -> str:
            documents = self.search_func(query)
            self.last_documents = documents
            return serialize_documents(documents)

        return StructuredTool.from_function(
            func=run,
            name="knowledge_retriever",
            description="Searches the publishing knowledge base for relevant passages.",
            args_schema=RetrievalInput,
        )


class ToolRegistry:
    @staticmethod
    def build_tools(search_func) -> Tuple[List[StructuredTool], RetrievalToolFactory]:
        retrieval_factory = RetrievalToolFactory(search_func)
        tools = [
            retrieval_factory.build(),
            StructuredTool.from_function(
                func=evaluate_expression,
                name="calculator",
                description="Evaluates basic arithmetic expressions.",
                args_schema=CalculatorInput,
            ),
        ]
        return tools, retrieval_factory
