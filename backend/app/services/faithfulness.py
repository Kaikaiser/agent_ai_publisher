from __future__ import annotations

import json
from pathlib import Path
from statistics import mean

from langchain_core.messages import HumanMessage, SystemMessage

from app.core.config import get_settings


JUDGE_PROMPT = """
你是一个离线评测裁判。你的任务是判断“答案”是否被“检索上下文”充分支持。

只依据给定上下文判断，不要使用外部知识。
如果答案包含上下文中没有支持的信息，应降低分数。

请只返回 JSON：
{
  "score": 0 到 1 之间的小数,
  "faithful": true 或 false,
  "reason": "一句简短中文说明"
}
""".strip()


class FaithfulnessEvaluator:
    def __init__(self, judge_llm) -> None:
        self.judge_llm = judge_llm
        self.settings = get_settings()

    def evaluate_file(self, dataset_path: str | None = None, report_path: str | None = None) -> dict:
        dataset_file = Path(dataset_path or self.settings.faithfulness_dataset_path)
        report_file = Path(report_path or self.settings.faithfulness_report_path)
        samples = self._load_dataset(dataset_file)
        results = [self._evaluate_sample(sample, index + 1) for index, sample in enumerate(samples)]

        aggregate = {
            "sample_count": len(results),
            "mean_score": round(mean([item["score"] for item in results]), 4) if results else 0.0,
            "faithful_rate": round(mean([1.0 if item["faithful"] else 0.0 for item in results]), 4) if results else 0.0,
        }
        report = {
            "dataset_path": str(dataset_file),
            "judge_provider": self.settings.faithfulness_judge_provider or self.settings.llm_provider,
            "judge_model": self.settings.judge_model_name,
            "aggregate": aggregate,
            "results": results,
        }

        report_file.parent.mkdir(parents=True, exist_ok=True)
        report_file.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        return report

    def _load_dataset(self, dataset_file: Path) -> list[dict]:
        if not dataset_file.exists():
            raise ValueError(f"Faithfulness dataset not found: {dataset_file}")

        samples = []
        for line_number, raw_line in enumerate(dataset_file.read_text(encoding="utf-8").splitlines(), start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at line {line_number}: {exc}") from exc
            samples.append(item)
        return samples

    def _evaluate_sample(self, sample: dict, index: int) -> dict:
        question = str(sample.get("question", "")).strip()
        answer = str(sample.get("answer", "")).strip()
        contexts = self._normalize_contexts(sample)
        if not question or not answer:
            raise ValueError(f"Dataset sample #{index} is missing question or answer.")

        prompt = (
            f"问题：{question}\n\n"
            f"答案：{answer}\n\n"
            f"检索上下文：\n{contexts}\n\n"
            "按要求输出 JSON。"
        )
        response = self.judge_llm.invoke(
            [
                SystemMessage(content=JUDGE_PROMPT),
                HumanMessage(content=prompt),
            ]
        )
        content = getattr(response, "content", response)
        parsed = self._parse_judge_payload(content)
        return {
            "index": index,
            "question": question,
            "answer": answer,
            "score": parsed["score"],
            "faithful": parsed["faithful"],
            "reason": parsed["reason"],
            "contexts": contexts,
            "metadata": sample.get("metadata", {}),
        }

    @staticmethod
    def _normalize_contexts(sample: dict) -> str:
        raw_contexts = sample.get("contexts")
        if raw_contexts is None:
            raw_contexts = sample.get("retrieved_contexts")
        if raw_contexts is None:
            raw_contexts = sample.get("sources")

        if isinstance(raw_contexts, str):
            return raw_contexts.strip()
        if isinstance(raw_contexts, list):
            parts = []
            for item in raw_contexts:
                if isinstance(item, str):
                    parts.append(item.strip())
                elif isinstance(item, dict):
                    parts.append(str(item.get("content") or item.get("text") or "").strip())
            return "\n\n".join(part for part in parts if part)
        return ""

    @staticmethod
    def _parse_judge_payload(content) -> dict:
        if isinstance(content, list):
            content = "\n".join(item.get("text", "") if isinstance(item, dict) else str(item) for item in content)
        text = str(content).strip()
        if "```" in text:
            segments = [segment.strip() for segment in text.split("```") if segment.strip()]
            for segment in segments:
                if segment.startswith("{") and segment.endswith("}"):
                    text = segment
                    break
                if "\n" in segment:
                    maybe_json = segment.split("\n", 1)[1].strip()
                    if maybe_json.startswith("{") and maybe_json.endswith("}"):
                        text = maybe_json
                        break
        parsed = json.loads(text)
        score = max(0.0, min(1.0, float(parsed.get("score", 0.0))))
        faithful = bool(parsed.get("faithful", score >= 0.8))
        reason = str(parsed.get("reason", "")).strip()
        return {"score": score, "faithful": faithful, "reason": reason}
