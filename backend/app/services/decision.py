from __future__ import annotations

from dataclasses import dataclass


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
    max_steps: int = 2


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
            max_steps=int(policy.get("max_steps", 2)),
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
