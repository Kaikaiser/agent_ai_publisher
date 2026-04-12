import json

from app.services.faithfulness import FaithfulnessEvaluator


class FakeJudgeResponse:
    def __init__(self, content):
        self.content = content


class FakeJudgeLLM:
    def invoke(self, messages):
        return FakeJudgeResponse('{"score": 1.0, "faithful": true, "reason": "答案完全由上下文支持"}')


def test_faithfulness_evaluator_writes_report(workspace_tmp_dir):
    dataset_path = workspace_tmp_dir / "faithfulness.jsonl"
    report_path = workspace_tmp_dir / "faithfulness_report.json"
    dataset_path.write_text(
        json.dumps(
            {
                "question": "这本教材适合几年级？",
                "answer": "适合三年级。",
                "contexts": ["本教材适合三年级学生使用。"],
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    report = FaithfulnessEvaluator(FakeJudgeLLM()).evaluate_file(str(dataset_path), str(report_path))

    assert report["aggregate"]["sample_count"] == 1
    assert report["aggregate"]["mean_score"] == 1.0
    assert report["results"][0]["faithful"] is True
    assert report_path.exists()
