from __future__ import annotations

import json
from dataclasses import dataclass
from json import JSONDecodeError

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, ValidationError


@dataclass
class ThoughtStep:
    intent: str
    need_context: list[str]
    evidence_status: str
    next_action: str
    risk: str
    reason: str


@dataclass
class ObservationSummary:
    executed_action: str
    knowledge_hits: int = 0
    memory_hits: int = 0
    calculator_available: bool = False
    used_query: str = ""
    evidence_quality: str = "none"


@dataclass
class ReactionDecision:
    thought: ThoughtStep
    should_answer: bool
    should_clarify: bool
    should_refuse: bool
    decision_source: str = "rule"
    proposed_action: str | None = None
    guard_reason: str = ""


@dataclass
class DecisionPlan:
    intent_type: str
    route_name: str
    decision_mode: str
    fallback_policy: str
    selected_tools: list[str]
    memory_scopes: list[str]
    clarification_needed: bool
    clarification_slot: str | None
    clarification_prompt: str | None
    allow_query_retry: bool
    note: str
    initial_thought: ThoughtStep
    max_steps: int = 3


class ModelReactionPayload(BaseModel):
    next_action: str
    should_answer: bool = False
    should_clarify: bool = False
    should_refuse: bool = False
    reason: str = ""


class DecisionService:
    def plan(
        self,
        question: str,
        *,
        decision_mode: str,
        fallback_policy: str,
        citation_policy: str,
        allow_roleplay: bool,
        has_image_input: bool,
    ) -> DecisionPlan:
        lowered = question.lower()
        intent_type = self._classify_intent(lowered, has_image_input)
        route_name = self._choose_route(intent_type, decision_mode)
        selected_tools = self._choose_tools(intent_type, route_name, has_image_input)
        memory_scopes = ["session", "book", "project", "user"]
        clarification_needed = self._needs_clarification(question, intent_type)
        clarification_slot, clarification_prompt = self._build_clarification_hint(
            question=question,
            intent_type=intent_type,
            route_name=route_name,
            decision_mode=decision_mode,
            clarification_needed=clarification_needed,
        )
        note_parts = [f"citation_policy={citation_policy}"]
        if allow_roleplay:
            note_parts.append("roleplay_enabled")
        initial_thought = self._build_initial_thought(
            intent_type=intent_type,
            route_name=route_name,
            decision_mode=decision_mode,
            selected_tools=selected_tools,
        )
        return DecisionPlan(
            intent_type=intent_type,
            route_name=route_name,
            decision_mode=decision_mode,
            fallback_policy=fallback_policy,
            selected_tools=selected_tools,
            memory_scopes=memory_scopes,
            clarification_needed=clarification_needed,
            clarification_slot=clarification_slot,
            clarification_prompt=clarification_prompt,
            allow_query_retry=True,
            note=", ".join(note_parts),
            initial_thought=initial_thought,
        )

    def runtime_plan(self, policy: dict | None) -> DecisionPlan:
        policy = policy or {}
        route_name = policy.get("route_name", "knowledge_answer")
        selected_tools = list(policy.get("selected_tools", ["knowledge_retrieval", "memory_retrieval"]))
        intent_type = policy.get("intent_type", "fact_qa")
        decision_mode = policy.get("decision_mode", "strict_knowledge")
        fallback_policy = policy.get("fallback_policy", "conservative_answer")
        memory_scopes = list(policy.get("memory_scopes", ["session", "book", "project", "user"]))
        clarification_needed = bool(policy.get("clarification_needed", False))
        clarification_slot = policy.get("clarification_slot")
        clarification_prompt = policy.get("clarification_prompt")
        allow_query_retry = bool(policy.get("allow_query_retry", True))
        note = str(policy.get("note", ""))
        initial_thought = self._build_initial_thought(
            intent_type=intent_type,
            route_name=route_name,
            decision_mode=decision_mode,
            selected_tools=selected_tools,
        )
        return DecisionPlan(
            intent_type=intent_type,
            route_name=route_name,
            decision_mode=decision_mode,
            fallback_policy=fallback_policy,
            selected_tools=selected_tools,
            memory_scopes=memory_scopes,
            clarification_needed=clarification_needed,
            clarification_slot=clarification_slot,
            clarification_prompt=clarification_prompt,
            allow_query_retry=allow_query_retry,
            note=note,
            initial_thought=initial_thought,
            max_steps=int(policy.get("max_steps", 3)),
        )

    def reassess(
        self,
        plan: DecisionPlan,
        observation: ObservationSummary,
        *,
        executed_actions: list[str],
    ) -> ReactionDecision:
        remaining_actions = [
            tool
            for tool in plan.selected_tools
            if tool in {"knowledge_retrieval", "memory_retrieval", "calculate"} and tool not in executed_actions
        ]

        if (
            observation.executed_action == "knowledge_retrieval"
            and observation.evidence_quality == "weak"
            and "memory_retrieval" not in remaining_actions
        ):
            if plan.clarification_needed:
                return ReactionDecision(
                    thought=ThoughtStep(
                        intent=plan.intent_type,
                        need_context=self._context_needs(plan.route_name),
                        evidence_status="weak",
                        next_action="clarify_user",
                        risk=self._risk_level(plan.decision_mode),
                        reason="Retrieved evidence is too weak for a reliable grounded answer.",
                    ),
                    should_answer=False,
                    should_clarify=True,
                    should_refuse=False,
                )
            if plan.fallback_policy == "conservative_answer":
                return ReactionDecision(
                    thought=ThoughtStep(
                        intent=plan.intent_type,
                        need_context=self._context_needs(plan.route_name),
                        evidence_status="weak",
                        next_action="final_answer",
                        risk=self._risk_level(plan.decision_mode),
                        reason="Retrieved evidence is weak, so return a conservative answer instead of a confident grounded one.",
                    ),
                    should_answer=True,
                    should_clarify=False,
                    should_refuse=False,
                )

        if (
            observation.executed_action == "knowledge_retrieval"
            and observation.knowledge_hits > 0
            and plan.route_name in {"knowledge_answer", "knowledge_summary"}
            and "memory_retrieval" in remaining_actions
        ):
            return ReactionDecision(
                thought=ThoughtStep(
                    intent=plan.intent_type,
                    need_context=self._context_needs(plan.route_name),
                    evidence_status="sufficient",
                    next_action="memory_retrieval",
                    risk=self._risk_level(plan.decision_mode),
                    reason="After the grounded retrieval succeeds, load memory once more for book-level style and user context.",
                ),
                should_answer=False,
                should_clarify=False,
                should_refuse=False,
            )

        if observation.knowledge_hits > 0:
            return ReactionDecision(
                thought=ThoughtStep(
                    intent=plan.intent_type,
                    need_context=self._context_needs(plan.route_name),
                    evidence_status="sufficient",
                    next_action="final_answer",
                    risk=self._risk_level(plan.decision_mode),
                    reason="Retrieved knowledge is sufficient for a grounded answer.",
                ),
                should_answer=True,
                should_clarify=False,
                should_refuse=False,
            )

        if observation.executed_action == "memory_retrieval" and "knowledge_retrieval" in remaining_actions:
            return ReactionDecision(
                thought=ThoughtStep(
                    intent=plan.intent_type,
                    need_context=self._context_needs(plan.route_name),
                    evidence_status="memory_only",
                    next_action="knowledge_retrieval",
                    risk=self._risk_level(plan.decision_mode),
                    reason="Memory helps with style and context, but factual answers still need knowledge retrieval.",
                ),
                should_answer=False,
                should_clarify=False,
                should_refuse=False,
            )

        if observation.executed_action == "knowledge_retrieval" and observation.knowledge_hits == 0:
            if plan.allow_query_retry and "knowledge_retrieval" not in remaining_actions and observation.used_query:
                rewritten = self.rewrite_query(observation.used_query)
                if rewritten != observation.used_query:
                    return ReactionDecision(
                        thought=ThoughtStep(
                            intent=plan.intent_type,
                            need_context=self._context_needs(plan.route_name),
                            evidence_status="insufficient",
                            next_action="retry_knowledge_retrieval",
                            risk=self._risk_level(plan.decision_mode),
                            reason="The first retrieval missed, so retry once with a normalized book-scoped query.",
                        ),
                        should_answer=False,
                        should_clarify=False,
                        should_refuse=False,
                    )
            if plan.route_name == "knowledge_plus_persona" and "memory_retrieval" in remaining_actions:
                return ReactionDecision(
                    thought=ThoughtStep(
                        intent=plan.intent_type,
                        need_context=self._context_needs(plan.route_name),
                        evidence_status="insufficient",
                        next_action="memory_retrieval",
                        risk=self._risk_level(plan.decision_mode),
                        reason="Knowledge retrieval missed, so retrieve book-style memory before the fallback decision.",
                    ),
                    should_answer=False,
                    should_clarify=False,
                    should_refuse=False,
                )
            if plan.clarification_needed:
                return ReactionDecision(
                    thought=ThoughtStep(
                        intent=plan.intent_type,
                        need_context=self._context_needs(plan.route_name),
                        evidence_status="insufficient",
                        next_action="clarify_user",
                        risk=self._risk_level(plan.decision_mode),
                        reason="The question is underspecified and retrieved evidence is empty.",
                    ),
                    should_answer=False,
                    should_clarify=True,
                    should_refuse=False,
                )
            if plan.fallback_policy == "conservative_answer":
                return ReactionDecision(
                    thought=ThoughtStep(
                        intent=plan.intent_type,
                        need_context=self._context_needs(plan.route_name),
                        evidence_status="insufficient",
                        next_action="final_answer",
                        risk=self._risk_level(plan.decision_mode),
                        reason="No direct evidence was found, so return a conservative fallback response.",
                    ),
                    should_answer=True,
                    should_clarify=False,
                    should_refuse=False,
                )
            return ReactionDecision(
                thought=ThoughtStep(
                    intent=plan.intent_type,
                    need_context=self._context_needs(plan.route_name),
                    evidence_status="insufficient",
                    next_action="refuse_answer",
                    risk=self._risk_level(plan.decision_mode),
                    reason="No direct evidence was found and the policy does not allow a conservative fallback.",
                ),
                should_answer=False,
                should_clarify=False,
                should_refuse=True,
            )

        if plan.clarification_needed and observation.knowledge_hits == 0 and observation.memory_hits == 0:
            return ReactionDecision(
                thought=ThoughtStep(
                    intent=plan.intent_type,
                    need_context=self._context_needs(plan.route_name),
                    evidence_status="insufficient",
                    next_action="clarify_user",
                    risk=self._risk_level(plan.decision_mode),
                    reason="Neither knowledge nor memory provided enough context.",
                ),
                should_answer=False,
                should_clarify=True,
                should_refuse=False,
            )

        return ReactionDecision(
            thought=ThoughtStep(
                intent=plan.intent_type,
                need_context=self._context_needs(plan.route_name),
                evidence_status="sufficient" if observation.memory_hits > 0 else "limited",
                next_action="final_answer",
                risk=self._risk_level(plan.decision_mode),
                reason="The current context is enough to produce the final response.",
            ),
            should_answer=True,
            should_clarify=False,
            should_refuse=False,
        )

    @staticmethod
    def _classify_intent(lowered_question: str, has_image_input: bool) -> str:
        if has_image_input:
            return "image_qa"
        if any(token in lowered_question for token in ("sum up", "summarize", "summary", "总结", "概括")):
            return "summary"
        if any(token in lowered_question for token in ("why", "explain", "how does", "解释", "为什么")):
            return "learning_explain"
        if any(token in lowered_question for token in ("character", "roleplay", "pretend", "扮演", "你是")):
            return "character_chat"
        if any(token in lowered_question for token in ("plot", "story", "剧情", "故事")):
            return "plot_chat"
        return "fact_qa"

    @staticmethod
    def _choose_route(intent_type: str, decision_mode: str) -> str:
        if intent_type == "image_qa":
            return "ocr_then_answer"
        if intent_type in {"character_chat", "plot_chat"} and decision_mode == "immersive_character":
            return "knowledge_plus_persona"
        if intent_type == "summary":
            return "knowledge_summary"
        return "knowledge_answer"

    @staticmethod
    def _choose_tools(intent_type: str, route_name: str, has_image_input: bool) -> list[str]:
        tools: list[str] = []
        if has_image_input:
            tools.append("ocr")
        if route_name in {"knowledge_answer", "knowledge_plus_persona", "knowledge_summary", "ocr_then_answer"}:
            tools.extend(["knowledge_retrieval", "memory_retrieval"])
        if intent_type == "fact_qa":
            tools.append("citation_policy")
        return tools

    @staticmethod
    def _needs_clarification(question: str, intent_type: str) -> bool:
        normalized = question.strip()
        if len(normalized) < 6:
            return True
        if intent_type == "fact_qa" and "?" not in normalized and "？" not in normalized and len(normalized.split()) <= 2:
            return True
        return False

    def _build_clarification_hint(
        self,
        *,
        question: str,
        intent_type: str,
        route_name: str,
        decision_mode: str,
        clarification_needed: bool,
    ) -> tuple[str | None, str | None]:
        if not clarification_needed:
            return None, None

        lowered = question.lower()
        if intent_type in {"character_chat", "plot_chat"} or decision_mode == "immersive_character":
            return "character_or_plot", "Please clarify which character, scene, or plot point in this book you want to ask about."
        if route_name == "knowledge_summary":
            return "chapter_or_topic", "Please clarify which chapter, section, or topic in this book you want summarized."
        if any(token in lowered for token in ("chapter", "section", "page", "第", "章", "节", "页")):
            return "chapter_or_section", "Please clarify the exact chapter, section, or page you want to ask about."
        if any(token in lowered for token in ("what", "why", "how", "什么", "为什么", "怎么")):
            return "topic", "Please clarify the exact topic or knowledge point in this book you want to ask about."
        return "topic", "Please clarify the specific chapter, character, or topic you want to ask about in this book."

    def rewrite_query(self, question: str) -> str:
        normalized = " ".join(question.replace("\n", " ").split())
        if not normalized:
            return normalized
        lowered = normalized.lower()
        stop_tokens = {
            "please",
            "briefly",
            "continue",
            "same",
            "book",
            "the",
            "a",
            "an",
            "this",
            "that",
        }
        pieces = [token for token in normalized.split() if token.lower().strip(".,?!") not in stop_tokens]
        rewritten = " ".join(pieces).strip()
        return rewritten or normalized

    def _build_initial_thought(
        self,
        *,
        intent_type: str,
        route_name: str,
        decision_mode: str,
        selected_tools: list[str],
    ) -> ThoughtStep:
        context_needs = self._context_needs(route_name)
        next_action = self._initial_action(route_name, selected_tools)
        return ThoughtStep(
            intent=intent_type,
            need_context=context_needs,
            evidence_status="unknown",
            next_action=next_action,
            risk=self._risk_level(decision_mode),
            reason="Start from the route-specific retrieval order and keep the answer inside the current book scope.",
        )

    @staticmethod
    def _initial_action(route_name: str, selected_tools: list[str]) -> str:
        allowed = [tool for tool in selected_tools if tool in {"knowledge_retrieval", "memory_retrieval", "calculate"}]
        if route_name == "knowledge_plus_persona" and "memory_retrieval" in allowed:
            return "memory_retrieval"
        if "knowledge_retrieval" in allowed:
            return "knowledge_retrieval"
        if "memory_retrieval" in allowed:
            return "memory_retrieval"
        if "calculate" in allowed:
            return "calculate"
        return "final_answer"

    @staticmethod
    def _context_needs(route_name: str) -> list[str]:
        if route_name == "knowledge_plus_persona":
            return ["book_memory", "knowledge"]
        if route_name == "knowledge_summary":
            return ["knowledge", "book_memory"]
        if route_name == "ocr_then_answer":
            return ["ocr_text", "knowledge"]
        return ["knowledge", "memory"]

    @staticmethod
    def _risk_level(decision_mode: str) -> str:
        if decision_mode == "strict_knowledge":
            return "must_be_grounded"
        if decision_mode == "immersive_character":
            return "stay_in_character_without_breaking_book_facts"
        return "grounded_with_style"

    def model_reassess(
        self,
        llm,
        plan: DecisionPlan,
        observation: ObservationSummary,
        *,
        question: str,
        executed_actions: list[str],
        decision_trace: list[ThoughtStep],
        execution_trace: list[str],
        total_knowledge_hits: int,
        total_memory_hits: int,
        evidence_quality: str,
    ) -> ReactionDecision:
        fallback = self.reassess(plan, observation, executed_actions=executed_actions)
        available_actions = self._available_actions(plan, observation, executed_actions)
        if not available_actions:
            return fallback

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    (
                        "You are the action selector for a publishing-domain assistant. "
                        "Choose exactly one next action from the allowed list. "
                        "Stay inside the current book/project scope. "
                        "Knowledge retrieval is the factual source of truth. "
                        "Memory is only for preference, style, and project context. "
                        "Return strict JSON only with keys: next_action, should_answer, should_clarify, should_refuse, reason."
                    ),
                ),
                (
                    "human",
                    (
                        "Question: {question}\n"
                        "Route: {route_name}\n"
                        "Intent: {intent_type}\n"
                        "Decision mode: {decision_mode}\n"
                        "Fallback policy: {fallback_policy}\n"
                        "Clarification needed: {clarification_needed}\n"
                        "Executed actions: {executed_actions}\n"
                        "Available actions: {available_actions}\n"
                        "Cumulative context:\n{cumulative_context}\n"
                        "Current observation:\n{observation_context}\n"
                        "Thought trace:\n{thought_context}\n"
                        "Execution trace:\n{execution_context}\n\n"
                        "Rules:\n"
                        "- Use final_answer only when grounded evidence is already sufficient.\n"
                        "- If evidence is weak and more retrieval actions remain, prefer retrieval over final_answer.\n"
                        "- If the question is underspecified and current evidence is insufficient, choose clarify_user.\n"
                        "- If no reliable answer can be produced under policy and no better retrieval action remains, choose refuse_answer.\n"
                        "- Do not invent actions outside the allowed list.\n"
                        "- Set only one of should_answer, should_clarify, should_refuse to true.\n"
                        "- For non-terminal actions, all three booleans must be false."
                    ),
                ),
            ]
        )
        messages = prompt.format_messages(
            question=question,
            route_name=plan.route_name,
            intent_type=plan.intent_type,
            decision_mode=plan.decision_mode,
            fallback_policy=plan.fallback_policy,
            clarification_needed=str(plan.clarification_needed).lower(),
            executed_actions=", ".join(executed_actions) if executed_actions else "none",
            available_actions=", ".join(available_actions),
            cumulative_context=self._format_cumulative_context(
                plan,
                executed_actions=executed_actions,
                total_knowledge_hits=total_knowledge_hits,
                total_memory_hits=total_memory_hits,
                evidence_quality=evidence_quality,
            ),
            observation_context=self._format_observation_context(observation),
            thought_context=self._format_thought_trace(decision_trace),
            execution_context=self._format_execution_trace(execution_trace),
        )

        try:
            result = llm.invoke(messages)
        except Exception:
            return self._decorate_reaction(
                fallback,
                decision_source="rule_fallback",
                proposed_action=None,
                guard_reason="model_invoke_failed",
            )
        try:
            raw_content = getattr(result, "content", str(result))
            payload = self._parse_model_reaction(raw_content)
            proposed = self._payload_to_reaction(payload, plan, available_actions)
            return self._guard_reaction(
                plan,
                observation,
                proposed,
                available_actions=available_actions,
                fallback=fallback,
                executed_actions=executed_actions,
                total_knowledge_hits=total_knowledge_hits,
                total_memory_hits=total_memory_hits,
                evidence_quality=evidence_quality,
            )
        except (ValueError, ValidationError, JSONDecodeError):
            return self._decorate_reaction(
                fallback,
                decision_source="rule_fallback",
                proposed_action=None,
                guard_reason="model_output_invalid",
            )

    def _payload_to_reaction(
        self,
        payload: ModelReactionPayload,
        plan: DecisionPlan,
        available_actions: list[str],
    ) -> ReactionDecision:
        if payload.next_action not in available_actions:
            raise ValueError("Model selected an action outside the allowed set")

        terminal_flags = [payload.should_answer, payload.should_clarify, payload.should_refuse]
        if sum(bool(flag) for flag in terminal_flags) > 1:
            raise ValueError("Model selected multiple terminal states")

        next_action = payload.next_action
        should_answer = payload.should_answer
        should_clarify = payload.should_clarify
        should_refuse = payload.should_refuse

        if next_action == "final_answer":
            should_answer = True
            should_clarify = False
            should_refuse = False
        elif next_action == "clarify_user":
            should_answer = False
            should_clarify = True
            should_refuse = False
        elif next_action == "refuse_answer":
            should_answer = False
            should_clarify = False
            should_refuse = True
        elif should_answer or should_clarify or should_refuse:
            raise ValueError("Non-terminal action cannot set terminal flags")

        return ReactionDecision(
            thought=ThoughtStep(
                intent=plan.intent_type,
                need_context=self._context_needs(plan.route_name),
                evidence_status=self._model_evidence_status(next_action),
                next_action=next_action,
                risk=self._risk_level(plan.decision_mode),
                reason=payload.reason.strip() or "Model selected the next action from the current observation.",
            ),
            should_answer=should_answer,
            should_clarify=should_clarify,
            should_refuse=should_refuse,
            decision_source="model",
            proposed_action=next_action,
        )

    def _available_actions(
        self,
        plan: DecisionPlan,
        observation: ObservationSummary,
        executed_actions: list[str],
    ) -> list[str]:
        remaining_actions = [
            tool
            for tool in plan.selected_tools
            if tool in {"knowledge_retrieval", "memory_retrieval", "calculate"} and tool not in executed_actions
        ]
        available_actions = list(remaining_actions)
        if (
            observation.executed_action == "knowledge_retrieval"
            and observation.knowledge_hits == 0
            and plan.allow_query_retry
            and observation.used_query
            and "knowledge_retrieval" not in remaining_actions
        ):
            rewritten = self.rewrite_query(observation.used_query)
            if rewritten != observation.used_query:
                available_actions.append("retry_knowledge_retrieval")

        available_actions.extend(["final_answer", "clarify_user", "refuse_answer"])
        deduped: list[str] = []
        for action in available_actions:
            if action not in deduped:
                deduped.append(action)
        return deduped

    def _guard_reaction(
        self,
        plan: DecisionPlan,
        observation: ObservationSummary,
        proposed: ReactionDecision,
        *,
        available_actions: list[str],
        fallback: ReactionDecision,
        executed_actions: list[str],
        total_knowledge_hits: int,
        total_memory_hits: int,
        evidence_quality: str,
    ) -> ReactionDecision:
        next_action = proposed.thought.next_action
        has_grounded_knowledge = total_knowledge_hits > 0
        has_memory_context = total_memory_hits > 0
        has_remaining_retrieval = any(
            action in available_actions for action in {"knowledge_retrieval", "memory_retrieval", "retry_knowledge_retrieval"}
        )

        if next_action == "final_answer":
            if not has_grounded_knowledge and plan.route_name in {
                "knowledge_answer",
                "knowledge_summary",
                "knowledge_plus_persona",
                "ocr_then_answer",
            }:
                return self._redirect_reaction(
                    plan,
                    available_actions=available_actions,
                    fallback=fallback,
                    proposed_action=proposed.thought.next_action,
                    preferred=self._preferred_followup_action(
                        plan,
                        observation,
                        available_actions=available_actions,
                        executed_actions=executed_actions,
                        total_knowledge_hits=total_knowledge_hits,
                        total_memory_hits=total_memory_hits,
                        evidence_quality=evidence_quality,
                    ),
                    reason="Final answer blocked because no grounded knowledge has been retrieved yet.",
                )

            if evidence_quality == "weak" and has_remaining_retrieval:
                return self._redirect_reaction(
                    plan,
                    available_actions=available_actions,
                    fallback=fallback,
                    proposed_action=proposed.thought.next_action,
                    preferred=self._preferred_followup_action(
                        plan,
                        observation,
                        available_actions=available_actions,
                        executed_actions=executed_actions,
                        total_knowledge_hits=total_knowledge_hits,
                        total_memory_hits=total_memory_hits,
                        evidence_quality=evidence_quality,
                    ),
                    reason="Final answer blocked because evidence is weak and more retrieval is still available.",
                )

            if plan.route_name == "knowledge_plus_persona" and not has_memory_context and "memory_retrieval" in available_actions:
                return self._redirect_reaction(
                    plan,
                    available_actions=available_actions,
                    fallback=fallback,
                    proposed_action=proposed.thought.next_action,
                    preferred="memory_retrieval",
                    reason="Final answer blocked because persona mode still needs memory context.",
                )

        if next_action == "refuse_answer" and has_grounded_knowledge:
            return self._redirect_reaction(
                plan,
                available_actions=available_actions,
                fallback=fallback,
                proposed_action=proposed.thought.next_action,
                preferred="final_answer",
                reason="Refusal blocked because grounded knowledge is already available.",
            )

        if next_action == "clarify_user" and has_grounded_knowledge and evidence_quality in {"medium", "strong"}:
            return self._redirect_reaction(
                plan,
                available_actions=available_actions,
                fallback=fallback,
                proposed_action=proposed.thought.next_action,
                preferred="final_answer",
                reason="Clarification skipped because grounded evidence is already sufficient.",
            )

        return self._decorate_reaction(
            proposed,
            decision_source="model",
            proposed_action=proposed.proposed_action or proposed.thought.next_action,
            guard_reason="",
        )

    def _redirect_reaction(
        self,
        plan: DecisionPlan,
        *,
        available_actions: list[str],
        fallback: ReactionDecision,
        proposed_action: str | None,
        preferred: str | None,
        reason: str,
    ) -> ReactionDecision:
        if preferred and preferred in available_actions:
            redirected = self._build_guard_reaction(plan, next_action=preferred, reason=reason)
            return self._decorate_reaction(
                redirected,
                decision_source="guard",
                proposed_action=proposed_action,
                guard_reason=reason,
            )
        return self._decorate_reaction(
            fallback,
            decision_source="rule_fallback",
            proposed_action=proposed_action,
            guard_reason=reason,
        )

    def _preferred_followup_action(
        self,
        plan: DecisionPlan,
        observation: ObservationSummary,
        *,
        available_actions: list[str],
        executed_actions: list[str],
        total_knowledge_hits: int,
        total_memory_hits: int,
        evidence_quality: str,
    ) -> str | None:
        del executed_actions
        if (
            observation.executed_action == "knowledge_retrieval"
            and total_knowledge_hits == 0
            and "retry_knowledge_retrieval" in available_actions
        ):
            return "retry_knowledge_retrieval"
        if plan.route_name == "knowledge_plus_persona" and total_memory_hits == 0 and "memory_retrieval" in available_actions:
            return "memory_retrieval"
        if evidence_quality == "weak" and "memory_retrieval" in available_actions:
            return "memory_retrieval"
        if total_knowledge_hits == 0 and "knowledge_retrieval" in available_actions:
            return "knowledge_retrieval"
        if plan.clarification_needed and "clarify_user" in available_actions:
            return "clarify_user"
        if total_knowledge_hits > 0 and evidence_quality in {"medium", "strong"} and "final_answer" in available_actions:
            return "final_answer"
        return None

    def _build_guard_reaction(self, plan: DecisionPlan, *, next_action: str, reason: str) -> ReactionDecision:
        return ReactionDecision(
            thought=ThoughtStep(
                intent=plan.intent_type,
                need_context=self._context_needs(plan.route_name),
                evidence_status=self._model_evidence_status(next_action),
                next_action=next_action,
                risk=self._risk_level(plan.decision_mode),
                reason=reason,
            ),
            should_answer=next_action == "final_answer",
            should_clarify=next_action == "clarify_user",
            should_refuse=next_action == "refuse_answer",
            decision_source="guard",
            proposed_action=next_action,
            guard_reason=reason,
        )

    @staticmethod
    def _decorate_reaction(
        reaction: ReactionDecision,
        *,
        decision_source: str,
        proposed_action: str | None,
        guard_reason: str,
    ) -> ReactionDecision:
        return ReactionDecision(
            thought=reaction.thought,
            should_answer=reaction.should_answer,
            should_clarify=reaction.should_clarify,
            should_refuse=reaction.should_refuse,
            decision_source=decision_source,
            proposed_action=proposed_action,
            guard_reason=guard_reason,
        )

    @staticmethod
    def _format_observation_context(observation: ObservationSummary) -> str:
        return (
            f"executed_action={observation.executed_action}\n"
            f"knowledge_hits={observation.knowledge_hits}\n"
            f"memory_hits={observation.memory_hits}\n"
            f"calculator_available={str(observation.calculator_available).lower()}\n"
            f"used_query={observation.used_query or 'none'}\n"
            f"evidence_quality={observation.evidence_quality}"
        )

    def _format_cumulative_context(
        self,
        plan: DecisionPlan,
        *,
        executed_actions: list[str],
        total_knowledge_hits: int,
        total_memory_hits: int,
        evidence_quality: str,
    ) -> str:
        return (
            f"route_name={plan.route_name}\n"
            f"grounded_knowledge_available={str(total_knowledge_hits > 0).lower()}\n"
            f"memory_context_available={str(total_memory_hits > 0).lower()}\n"
            f"total_knowledge_hits={total_knowledge_hits}\n"
            f"total_memory_hits={total_memory_hits}\n"
            f"evidence_quality={evidence_quality}\n"
            f"executed_actions={', '.join(executed_actions) if executed_actions else 'none'}"
        )

    @staticmethod
    def _format_thought_trace(trace: list[ThoughtStep]) -> str:
        if not trace:
            return "none"
        return "\n".join(
            f"- intent={item.intent}; evidence={item.evidence_status}; next_action={item.next_action}; risk={item.risk}; reason={item.reason}"
            for item in trace
        )

    @staticmethod
    def _format_execution_trace(trace: list[str]) -> str:
        if not trace:
            return "none"
        return "\n".join(f"- {item}" for item in trace)

    @staticmethod
    def _parse_model_reaction(content) -> ModelReactionPayload:
        text = content if isinstance(content, str) else str(content)
        text = text.strip()
        if text.startswith("```"):
            lines = [line for line in text.splitlines() if not line.strip().startswith("```")]
            text = "\n".join(lines).strip()
        if "{" in text and "}" in text:
            text = text[text.find("{") : text.rfind("}") + 1]
        payload = json.loads(text)
        return ModelReactionPayload.model_validate(payload)

    @staticmethod
    def _model_evidence_status(next_action: str) -> str:
        if next_action == "final_answer":
            return "sufficient"
        if next_action == "clarify_user":
            return "insufficient"
        if next_action == "refuse_answer":
            return "insufficient"
        return "in_progress"
