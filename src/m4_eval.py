from __future__ import annotations

"""Module 4: RAGAS Evaluation — 4 metrics + failure analysis."""

import os, sys, json, math
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import TEST_SET_PATH

METRICS = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]

# Diagnostic Tree: metric thấp nhất → (nguyên nhân, hướng xử lý)
DIAGNOSTIC_TREE = {
    "faithfulness": ("LLM hallucinating", "Thắt chặt system prompt, giảm temperature về 0"),
    "context_recall": ("Missing relevant chunks", "Cải thiện chunking hoặc thêm BM25/keyword"),
    "context_precision": ("Too many irrelevant chunks", "Thêm reranking (cross-encoder) hoặc lọc metadata"),
    "answer_relevancy": ("Answer doesn't match question", "Viết lại prompt để trả lời trực tiếp hơn"),
}


@dataclass
class EvalResult:
    question: str
    answer: str
    contexts: list[str]
    ground_truth: str
    faithfulness: float
    answer_relevancy: float
    context_precision: float
    context_recall: float


def load_test_set(path: str = TEST_SET_PATH) -> list[dict]:
    """Load test set from JSON. (Đã implement sẵn)"""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _safe(value) -> float:
    """RAGAS có thể trả NaN khi một câu không tính được → coi như 0."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        return 0.0
    return 0.0 if math.isnan(v) else v


def evaluate_ragas(questions: list[str], answers: list[str],
                   contexts: list[list[str]], ground_truths: list[str]) -> dict:
    """Run RAGAS evaluation.

    Cần OPENAI_API_KEY (RAGAS mặc định dùng gpt-4o-mini làm judge + OpenAI embeddings).
    Không có key / lỗi → in cảnh báo và trả về 0.0 cho cả 4 metric.
    """
    zeros = {m: 0.0 for m in METRICS}
    zeros["per_question"] = []
    try:
        from ragas import evaluate
        from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
        from datasets import Dataset

        dataset = Dataset.from_dict({
            "question": questions, "answer": answers,
            "contexts": contexts, "ground_truth": ground_truths,
        })
        result = evaluate(dataset, metrics=[faithfulness, answer_relevancy,
                                            context_precision, context_recall])
        df = result.to_pandas()
        per_question = [
            EvalResult(
                question=row["question"], answer=row["answer"],
                contexts=list(row["contexts"]), ground_truth=row["ground_truth"],
                faithfulness=_safe(row.get("faithfulness")),
                answer_relevancy=_safe(row.get("answer_relevancy")),
                context_precision=_safe(row.get("context_precision")),
                context_recall=_safe(row.get("context_recall")),
            )
            for _, row in df.iterrows()
        ]
        aggregate = {m: _safe(df[m].mean()) for m in METRICS}
        return {**aggregate, "per_question": per_question}
    except Exception as e:
        print(f"  ⚠️  RAGAS evaluation failed: {e}")
        return zeros


def failure_analysis(eval_results: list[EvalResult], bottom_n: int = 10) -> list[dict]:
    """Analyze bottom-N worst questions using Diagnostic Tree."""
    analyzed = []
    for r in eval_results:
        scores = {m: getattr(r, m) for m in METRICS}
        avg = sum(scores.values()) / len(METRICS)
        worst_metric = min(scores, key=scores.get)
        diagnosis, suggested_fix = DIAGNOSTIC_TREE[worst_metric]
        analyzed.append({
            "question": r.question,
            "answer": r.answer,
            "ground_truth": r.ground_truth,
            "contexts": r.contexts,
            "avg_score": round(avg, 4),
            "worst_metric": worst_metric,
            "score": round(scores[worst_metric], 4),
            "diagnosis": diagnosis,
            "suggested_fix": suggested_fix,
        })
    analyzed.sort(key=lambda x: x["avg_score"])
    return analyzed[:bottom_n]


def save_report(results: dict, failures: list[dict], path: str = "reports/ragas_report.json"):
    """Save evaluation report to JSON. (Đã implement sẵn)"""
    parent_dir = os.path.dirname(path)
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)
    report = {
        "aggregate": {k: v for k, v in results.items() if k != "per_question"},
        "num_questions": len(results.get("per_question", [])),
        "failures": failures,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"Report saved to {path}")


if __name__ == "__main__":
    test_set = load_test_set()
    print(f"Loaded {len(test_set)} test questions")
    print("Run pipeline.py first to generate answers, then call evaluate_ragas().")
