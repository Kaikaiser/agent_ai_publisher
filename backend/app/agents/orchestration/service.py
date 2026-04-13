from __future__ import annotations

import re
from collections.abc import Callable

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate

from app.mcp import InProcessMcpToolClient
from app.services.decision import DecisionService, ObservationSummary, ReactionDecision, ThoughtStep
from app.services.memory import MemorySnippet

SYSTEM_PROMPT = """
You are a publishing-domain assistant. Use memory only as background context, user preference,
or project guidance. Treat retrieved knowledge context as the source of truth for factual book
content. If memory conflicts with retrieved knowledge, prefer retrieved knowledge.
""".strip()


class AgentOrchestrator:
    def __init__(self, llm, memory_search_func: Callable[[str], list[MemorySnippet]] | None = None) -> None:
        self.llm = llm
        self.memory_search_func = memory_search_func
        self.response_policy: dict | None = None
        self.decision_service = DecisionService()

    def run(self, question: str, search_func) -> dict:
        mcp_client = InProcessMcpToolClient(search_func)
        execution = self._execute_controlled_react(question, mcp_client)

        if execution["calculator_result"] is not None and not execution["documents"]:
            return {
                "answer": f"计算结果：{execution['calculator_result']}",
                "documents": [],
                "grounded": False,
                "clarification_needed": False,
                "evidence_quality": "none",
                "decision_trace": execution["decision_trace"],
                "execution_trace": execution["execution_trace"],
                "agent_steps": execution["agent_steps"],
            }

        if execution["should_refuse"]:
            return {
                "answer": "I cannot answer this reliably from the current book context.",
                "documents": [],
                "grounded": False,
                "clarification_needed": False,
                "evidence_quality": execution["evidence_quality"],
                "decision_trace": execution["decision_trace"],
                "execution_trace": execution["execution_trace"],
                "agent_steps": execution["agent_steps"],
            }

        if execution["should_clarify"]:
            return {
                "answer": self._policy_value(
                    "clarification_prompt",
                    "Please clarify the specific chapter, character, or topic you want to ask about in this book.",
                ),
                "documents": [],
                "grounded": False,
                "clarification_needed": True,
                "evidence_quality": execution["evidence_quality"],
                "decision_trace": execution["decision_trace"],
                "execution_trace": execution["execution_trace"],
                "agent_steps": execution["agent_steps"],
            }

        if not execution["documents"]:
            return {
                "answer": "No relevant knowledge was retrieved for this question.",
                "documents": [],
                "grounded": False,
                "clarification_needed": False,
                "evidence_quality": execution["evidence_quality"],
                "decision_trace": execution["decision_trace"],
                "execution_trace": execution["execution_trace"],
                "agent_steps": execution["agent_steps"],
            }

        memory_context = self._format_memory_context(execution["memories"])
        policy_context = self._format_policy_context(self.response_policy)
        thought_context = self._format_thought_context(execution["decision_trace"])
        execution_context = self._format_execution_context(execution["execution_trace"])
        knowledge_context = "\n\n".join(
            (
                f"Source: {item.metadata.get('filename', '')} | {item.metadata.get('book_title', '')} | "
                f"{item.metadata.get('doc_type', '')} | {item.metadata.get('location', 'unknown')}\n"
                f"Content: {item.page_content}"
            )
            for item in execution["documents"]
        )
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", SYSTEM_PROMPT),
                (
                    "human",
                    "Question:\n{question}\n\nResponse policy:\n{policy_context}\n\nThought trace:\n{thought_context}\n\nExecution trace:\n{execution_context}\n\nRelevant memory:\n{memory_context}\n\nRetrieved knowledge:\n{knowledge_context}",
                ),
            ]
        )
        message = prompt.format_messages(
            question=question,
            policy_context=policy_context,
            thought_context=thought_context,
            execution_context=execution_context,
            memory_context=memory_context,
            knowledge_context=knowledge_context,
        )
        result = self.llm.invoke(message)
        answer = getattr(result, "content", str(result))
        return {
            "answer": answer,
            "documents": execution["documents"],
            "grounded": True,
            "clarification_needed": False,
            "evidence_quality": execution["evidence_quality"],
            "decision_trace": execution["decision_trace"],
            "execution_trace": execution["execution_trace"],
            "agent_steps": execution["agent_steps"],
        }

    def _execute_controlled_react(self, question: str, mcp_client: InProcessMcpToolClient) -> dict:
        plan = self.decision_service.runtime_plan(self.response_policy)
        calculator_result = self._maybe_calculate(question, mcp_client)
        memories: list[MemorySnippet] = []
        documents: list[Document] = []
        decision_trace: list[ThoughtStep] = [plan.initial_thought]
        execution_trace: list[str] = []
        step_logs: list[dict] = []
        executed_actions: list[str] = []
        next_action = plan.initial_thought.next_action
        reaction: ReactionDecision | None = None
        current_query = question
        evidence_quality = "none"

        if calculator_result is not None and "knowledge_retrieval" not in plan.selected_tools:
            execution_trace.append("action=calculate observation=success")
            return {
                "calculator_result": calculator_result,
                "memories": [],
                "documents": [],
                "decision_trace": decision_trace,
                "execution_trace": execution_trace,
                "agent_steps": step_logs,
                "evidence_quality": "none",
                "should_clarify": False,
                "should_refuse": False,
            }

        for _ in range(max(plan.max_steps, 1)):
            if next_action not in {"knowledge_retrieval", "memory_retrieval", "calculate", "retry_knowledge_retrieval"}:
                break

            observation, action_payload = self._execute_action(
                action=next_action,
                question=current_query,
                mcp_client=mcp_client,
                calculator_result=calculator_result,
            )
            executed_actions.append(next_action)
            execution_trace.append(self._format_observation_trace(observation))

            if next_action in {"knowledge_retrieval", "retry_knowledge_retrieval"}:
                documents = action_payload
                evidence_quality = self._assess_evidence_quality(documents)
            elif next_action == "memory_retrieval":
                memories = action_payload

            if next_action == "calculate" and calculator_result is not None:
                break

            if next_action in {"knowledge_retrieval", "retry_knowledge_retrieval"}:
                observation.knowledge_hits = len(documents)
                observation.evidence_quality = evidence_quality
            elif next_action == "memory_retrieval":
                observation.memory_hits = len(memories)

            reaction = self.decision_service.model_reassess(
                self.llm,
                plan,
                observation,
                question=question,
                executed_actions=executed_actions,
                decision_trace=decision_trace,
                execution_trace=execution_trace,
                total_knowledge_hits=len(documents),
                total_memory_hits=len(memories),
                evidence_quality=evidence_quality,
            )
            step_logs.append(
                {
                    "step_index": len(executed_actions),
                    "executed_action": next_action,
                    "used_query": observation.used_query or current_query,
                    "knowledge_hits": observation.knowledge_hits,
                    "memory_hits": observation.memory_hits,
                    "evidence_quality": observation.evidence_quality,
                    "proposed_next_action": reaction.proposed_action or reaction.thought.next_action,
                    "chosen_next_action": reaction.thought.next_action,
                    "decision_source": reaction.decision_source,
                    "guard_reason": reaction.guard_reason,
                    "should_answer": reaction.should_answer,
                    "should_clarify": reaction.should_clarify,
                    "should_refuse": reaction.should_refuse,
                    "thought_reason": reaction.thought.reason,
                }
            )
            decision_trace.append(reaction.thought)

            if reaction.should_answer or reaction.should_clarify or reaction.should_refuse:
                break

            next_action = reaction.thought.next_action
            if next_action == "retry_knowledge_retrieval":
                current_query = self.decision_service.rewrite_query(current_query)

        return {
            "calculator_result": calculator_result,
            "memories": memories,
            "documents": documents,
            "decision_trace": decision_trace,
            "execution_trace": execution_trace,
            "agent_steps": step_logs,
            "evidence_quality": evidence_quality,
            "should_clarify": bool(reaction.should_clarify) if reaction is not None else plan.clarification_needed,
            "should_refuse": bool(reaction.should_refuse) if reaction is not None else False,
        }

    def _execute_action(
        self,
        *,
        action: str,
        question: str,
        mcp_client: InProcessMcpToolClient,
        calculator_result,
    ) -> tuple[ObservationSummary, list[Document] | list[MemorySnippet] | None]:
        if action in {"knowledge_retrieval", "retry_knowledge_retrieval"}:
            documents = mcp_client.retrieve(question)
            return ObservationSummary(executed_action=action, knowledge_hits=len(documents), used_query=question), documents
        if action == "memory_retrieval":
            memories = self.memory_search_func(question) if self.memory_search_func else []
            return ObservationSummary(executed_action=action, memory_hits=len(memories)), memories
        if action == "calculate":
            return ObservationSummary(
                executed_action=action,
                calculator_available=calculator_result is not None,
            ), None
        return ObservationSummary(executed_action=action), None

    @staticmethod
    def _format_memory_context(memories: list[MemorySnippet]) -> str:
        if not memories:
            return "None"
        return "\n".join(f"- [{item.scope}/{item.memory_type}] {item.summary}: {item.content}" for item in memories)

    @staticmethod
    def _format_policy_context(policy: dict | None) -> str:
        if not policy:
            return "None"
        return "\n".join(f"- {key}: {value}" for key, value in policy.items())

    @staticmethod
    def _format_thought_context(trace: list[ThoughtStep]) -> str:
        if not trace:
            return "No thought trace"
        return "\n".join(
            f"- intent={item.intent}; evidence={item.evidence_status}; next_action={item.next_action}; risk={item.risk}; reason={item.reason}"
            for item in trace
        )

    @staticmethod
    def _format_execution_context(trace: list[str]) -> str:
        if not trace:
            return "No tool execution"
        return "\n".join(f"- {item}" for item in trace)

    @staticmethod
    def _format_observation_trace(observation: ObservationSummary) -> str:
        if observation.executed_action in {"knowledge_retrieval", "retry_knowledge_retrieval"}:
            return (
                f"action={observation.executed_action} "
                f"observation=hits:{observation.knowledge_hits} quality={observation.evidence_quality} query={observation.used_query}"
            )
        if observation.executed_action == "memory_retrieval":
            return f"action=memory_retrieval observation=hits:{observation.memory_hits}"
        if observation.executed_action == "calculate":
            status = "success" if observation.calculator_available else "skipped"
            return f"action=calculate observation={status}"
        return f"action={observation.executed_action} observation=done"

    def _maybe_calculate(self, question: str, mcp_client: InProcessMcpToolClient):
        expression = question.strip()
        if re.fullmatch(r"[0-9\s\+\-\*/\(\)\.]+", expression):
            try:
                return mcp_client.calculate(expression)
            except ValueError:
                return None
        return None

    @staticmethod
    def _assess_evidence_quality(documents: list[Document]) -> str:
        if not documents:
            return "none"

        total_chars = sum(len(item.page_content.strip()) for item in documents)
        distinct_sources = {
            (
                item.metadata.get("document_id"),
                item.metadata.get("filename", ""),
                item.metadata.get("location", ""),
            )
            for item in documents
        }
        if len(documents) >= 3 or (len(distinct_sources) >= 2 and total_chars >= 180):
            return "strong"
        if len(documents) >= 2 or total_chars >= 120:
            return "medium"
        return "weak"

    def _policy_value(self, key: str, default):
        if not self.response_policy:
            return default
        return self.response_policy.get(key, default)
