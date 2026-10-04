from __future__ import annotations

"""
Module 5: Enrichment Pipeline
==============================
Làm giàu chunks TRƯỚC khi embed: Summarize, HyQA, Contextual Prepend, Auto Metadata.

Test: pytest tests/test_m5.py
"""

import os, sys, json, re
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import OPENAI_API_KEY

LLM_MODEL = "gpt-4o-mini"
ENRICH_WORKERS = 8  # số call song song khi chạy combined mode


@dataclass
class EnrichedChunk:
    """Chunk đã được làm giàu."""
    original_text: str
    enriched_text: str
    summary: str
    hypothesis_questions: list[str]
    auto_metadata: dict
    method: str  # "contextual", "summary", "hyqa", "full"


_client = None


def _chat(system: str, user: str, max_tokens: int, json_mode: bool = False) -> str | None:
    """Gọi gpt-4o-mini. Trả về None nếu không có API key hoặc call lỗi (để caller fallback)."""
    global _client
    if not OPENAI_API_KEY:
        return None
    try:
        from openai import OpenAI
        if _client is None:
            _client = OpenAI()
        kwargs = {"response_format": {"type": "json_object"}} if json_mode else {}
        resp = _client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            max_tokens=max_tokens,
            temperature=0,
            **kwargs,
        )
        return (resp.choices[0].message.content or "").strip()
    except Exception as e:
        print(f"  ⚠️  OpenAI call failed: {e}")
        return None


# ─── Technique 1: Chunk Summarization ────────────────────


def summarize_chunk(text: str) -> str:
    """
    Tạo summary ngắn cho chunk.
    Embed summary thay vì (hoặc cùng với) raw chunk → giảm noise.
    """
    out = _chat("Tóm tắt đoạn văn sau trong 2-3 câu ngắn gọn bằng tiếng Việt.", text, 150)
    if out:
        return out
    # Extractive fallback (không cần API): lấy 2 câu đầu
    sentences = [s.strip() for s in text.replace("\n", " ").split(". ") if s.strip()]
    return ". ".join(sentences[:2]) + "." if sentences else text


# ─── Technique 2: Hypothesis Question-Answer (HyQA) ─────


def generate_hypothesis_questions(text: str, n_questions: int = 3) -> list[str]:
    """
    Generate câu hỏi mà chunk có thể trả lời.
    Index cả questions lẫn chunk → query match tốt hơn (bridge vocabulary gap).
    """
    out = _chat(f"Dựa trên đoạn văn, tạo {n_questions} câu hỏi mà đoạn văn có thể trả lời. "
                "Trả về mỗi câu hỏi trên 1 dòng.", text, 200)
    if out:
        questions = [q.strip().lstrip("0123456789.-) ") for q in out.split("\n") if q.strip()]
        return [q for q in questions if q][:n_questions]
    # Extractive fallback: biến các câu dài thành câu hỏi
    sentences = [s.strip() for s in re.split(r"[.!?\n]", text) if len(s.strip()) > 10]
    return [f"{s.rstrip('.')}?" for s in sentences[:n_questions]]


# ─── Technique 3: Contextual Prepend (Anthropic style) ──


def contextual_prepend(text: str, document_title: str = "") -> str:
    """
    Prepend context giải thích chunk nằm ở đâu trong document.
    Anthropic benchmark: giảm 49% retrieval failure (alone).
    """
    out = _chat("Viết 1 câu ngắn mô tả đoạn văn này nằm ở đâu trong tài liệu và nói về chủ đề gì. "
                "Chỉ trả về 1 câu.",
                f"Tài liệu: {document_title}\n\nĐoạn văn:\n{text}", 80)
    if out:
        return f"{out}\n\n{text}"
    # Simple fallback: ghép tên tài liệu làm tiêu đề
    prefix = f"Trích từ {document_title}. " if document_title else ""
    return f"{prefix}{text}"


# ─── Technique 4: Auto Metadata Extraction ──────────────


def extract_metadata(text: str) -> dict:
    """
    LLM extract metadata tự động: topic, entities, date_range, category.
    """
    out = _chat('Trích xuất metadata từ đoạn văn. Trả về JSON: '
                '{"topic": "...", "entities": ["..."], "category": "policy|hr|it|finance", "language": "vi|en"}',
                text, 150, json_mode=True)
    if out:
        try:
            data = json.loads(out)
            if isinstance(data, dict):
                return data
        except json.JSONDecodeError as e:
            print(f"  ⚠️  Metadata JSON không hợp lệ: {e}")
    return {"topic": "general", "entities": [], "category": "unknown", "language": "vi"}


# ─── Combined Single-Call Mode ───────────────────────────

COMBINED_SYSTEM_PROMPT = """Phân tích đoạn văn và trả về JSON:
{
  "summary": "tóm tắt 2-3 câu",
  "questions": ["câu hỏi 1", "câu hỏi 2", "câu hỏi 3"],
  "context": "1 câu mô tả đoạn văn nằm ở đâu trong tài liệu",
  "metadata": {"topic": "...", "entities": ["..."], "category": "policy|hr|it|finance", "language": "vi|en"}
}"""


def _enrich_single_call(text: str, source: str) -> dict:
    """Single LLM call to get summary + questions + context + metadata.

    ⚠️ Cost optimization: 1 API call thay vì 4 calls riêng lẻ.
    """
    out = _chat(COMBINED_SYSTEM_PROMPT, f"Tài liệu: {source}\n\nĐoạn văn:\n{text}", 400, json_mode=True)
    if out:
        try:
            data = json.loads(out)
            return {
                "summary": str(data.get("summary", "")),
                "questions": [str(q) for q in data.get("questions", [])][:3],
                "context": str(data.get("context", "")),
                "metadata": dict(data.get("metadata", {})),
            }
        except (json.JSONDecodeError, TypeError, ValueError) as e:
            print(f"  ⚠️  Enrichment JSON không hợp lệ: {e}")
    # Fallback (không có API key): không có summary/questions, chỉ gắn tên tài liệu làm context
    return {
        "summary": "",
        "questions": [],
        "context": f"Trích từ {source}." if source else "",
        "metadata": {},
    }


# ─── Full Enrichment Pipeline ────────────────────────────


def _build_enriched_text(text: str, context: str, questions: list[str]) -> str:
    """Text dùng để index: context + nội dung gốc + câu hỏi giả định (HyQA)."""
    parts = [context, text]
    if questions:
        parts.append("Câu hỏi liên quan: " + " | ".join(questions))
    return "\n\n".join(p for p in parts if p)


def enrich_chunks(
    chunks: list[dict],
    methods: list[str] | None = None,
) -> list[EnrichedChunk]:
    """
    Chạy enrichment pipeline trên danh sách chunks. (Đã implement sẵn — dùng functions ở trên)

    Có 2 chế độ:
    - methods cụ thể (["summary"], ["contextual"]...): gọi từng function riêng (tốt cho học/debug)
    - methods=["combined"] hoặc None: 1 API call duy nhất cho tất cả (tốt cho production)

    Args:
        chunks: List of {"text": str, "metadata": dict}
        methods: Default None → combined mode (1 call/chunk).
                 Options: "summary", "hyqa", "contextual", "metadata", "combined"
    """
    if methods is None:
        methods = ["combined"]

    use_combined = "combined" in methods
    method_label = "+".join(methods)

    def _combined(chunk: dict) -> EnrichedChunk:
        text = chunk["text"]
        source = chunk.get("metadata", {}).get("source", "")
        result = _enrich_single_call(text, source)
        questions = result.get("questions", [])
        return EnrichedChunk(
            original_text=text,
            enriched_text=_build_enriched_text(text, result.get("context", ""), questions),
            summary=result.get("summary", ""),
            hypothesis_questions=questions,
            auto_metadata={**chunk.get("metadata", {}), **result.get("metadata", {})},
            method=method_label,
        )

    def _individual(chunk: dict) -> EnrichedChunk:
        text = chunk["text"]
        source = chunk.get("metadata", {}).get("source", "")
        summary = summarize_chunk(text) if "summary" in methods else ""
        questions = generate_hypothesis_questions(text) if "hyqa" in methods else []
        base = contextual_prepend(text, source) if "contextual" in methods else text
        auto_meta = extract_metadata(text) if "metadata" in methods else {}
        return EnrichedChunk(
            original_text=text,
            enriched_text=_build_enriched_text(base, "", questions),
            summary=summary,
            hypothesis_questions=questions,
            auto_metadata={**chunk.get("metadata", {}), **auto_meta},
            method=method_label,
        )

    if use_combined:
        # Các call độc lập nhau → chạy song song, executor.map giữ nguyên thứ tự chunk
        with ThreadPoolExecutor(max_workers=ENRICH_WORKERS) as pool:
            enriched = []
            for i, item in enumerate(pool.map(_combined, chunks), start=1):
                enriched.append(item)
                if i % 10 == 0 or i == len(chunks):
                    print(f"  Enriched {i}/{len(chunks)} chunks...", flush=True)
    else:
        enriched = []
        for i, chunk in enumerate(chunks, start=1):
            enriched.append(_individual(chunk))
            if i % 10 == 0 or i == len(chunks):
                print(f"  Enriched {i}/{len(chunks)} chunks...", flush=True)

    return enriched


# ─── Main ────────────────────────────────────────────────

if __name__ == "__main__":
    sample = "Nhân viên chính thức được nghỉ phép năm 12 ngày làm việc mỗi năm. Số ngày nghỉ phép tăng thêm 1 ngày cho mỗi 5 năm thâm niên công tác."

    print("=== Enrichment Pipeline Demo ===\n")
    print(f"Original: {sample}\n")

    s = summarize_chunk(sample)
    print(f"Summary: {s}\n")

    qs = generate_hypothesis_questions(sample)
    print(f"HyQA questions: {qs}\n")

    ctx = contextual_prepend(sample, "Sổ tay nhân viên VinUni 2024")
    print(f"Contextual: {ctx}\n")

    meta = extract_metadata(sample)
    print(f"Auto metadata: {meta}")
