from __future__ import annotations

"""Production RAG Pipeline — Ghép toàn bộ M1+M2+M3+M4+M5."""

import os, sys, time, json
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.m1_chunking import load_documents, chunk_hierarchical
from src.m2_search import HybridSearch
from src.m3_rerank import CrossEncoderReranker
from src.m4_eval import load_test_set, evaluate_ragas, failure_analysis, save_report
from src.m5_enrichment import enrich_chunks
from config import RERANK_TOP_K, OPENAI_API_KEY

LATENCY_REPORT_PATH = "reports/latency_breakdown.json"


def build_pipeline():
    """Build production RAG pipeline.

    Returns:
        search: HybridSearch đã index child chunks (M2)
        reranker: CrossEncoderReranker (M3)
        parent_texts: {parent_key: text} — để gửi parent (context đầy đủ) cho LLM
        build_ms: {step: ms} — thời gian từng bước khi build
    """
    print("=" * 60)
    print("PRODUCTION RAG PIPELINE")
    print("=" * 60, flush=True)
    build_ms: dict[str, float] = {}

    # Step 1: Load & Chunk (M1) — hierarchical: index child, giữ parent để trả về
    t0 = time.perf_counter()
    print("\n[1/4] Chunking documents...", flush=True)
    docs = load_documents()
    all_chunks = []
    parent_texts: dict[str, str] = {}
    for doc in docs:
        source = doc["metadata"]["source"]
        parents, children = chunk_hierarchical(doc["text"], metadata=doc["metadata"])
        # parent_id phải duy nhất toàn corpus (mỗi file đều bắt đầu từ parent_0)
        for p in parents:
            parent_texts[f"{source}::{p.metadata['parent_id']}"] = p.text
        for child in children:
            all_chunks.append({"text": child.text,
                               "metadata": {**child.metadata, "parent_id": f"{source}::{child.parent_id}"}})
    build_ms["chunking"] = (time.perf_counter() - t0) * 1000
    print(f"  ✓ {len(all_chunks)} child chunks / {len(parent_texts)} parents "
          f"from {len(docs)} documents ({build_ms['chunking']/1000:.1f}s)", flush=True)

    # Step 2: Enrichment (M5)
    t0 = time.perf_counter()
    mode = "1 API call/chunk" if OPENAI_API_KEY else "không có API key → fallback"
    print(f"\n[2/4] Enriching {len(all_chunks)} chunks (M5, {mode})...", flush=True)
    enriched = enrich_chunks(all_chunks)
    all_chunks = [{"text": e.enriched_text, "metadata": e.auto_metadata} for e in enriched]
    build_ms["enrichment"] = (time.perf_counter() - t0) * 1000
    print(f"  ✓ Enriched {len(enriched)} chunks ({build_ms['enrichment']/1000:.1f}s)", flush=True)

    # Step 3: Index (M2)
    t0 = time.perf_counter()
    print(f"\n[3/4] Indexing {len(all_chunks)} chunks (BM25 + Dense)...", flush=True)
    search = HybridSearch()
    search.index(all_chunks)
    build_ms["indexing"] = (time.perf_counter() - t0) * 1000
    print(f"  ✓ Indexed ({build_ms['indexing']/1000:.1f}s)", flush=True)

    # Step 4: Reranker (M3)
    t0 = time.perf_counter()
    print("\n[4/4] Loading reranker...", flush=True)
    reranker = CrossEncoderReranker()
    reranker._load_model()  # load trước để không tính vào latency query
    build_ms["reranker_load"] = (time.perf_counter() - t0) * 1000
    print(f"  ✓ Reranker ready ({build_ms['reranker_load']/1000:.1f}s)", flush=True)

    return search, reranker, parent_texts, build_ms


def run_query(query: str, search: HybridSearch, reranker: CrossEncoderReranker,
              parent_texts: dict[str, str]) -> tuple[str, list[str], dict[str, float]]:
    """Run single query through pipeline. Trả về (answer, contexts, latency_ms)."""
    timings: dict[str, float] = {}

    t0 = time.perf_counter()
    results = search.search(query)
    timings["search"] = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    docs = [{"text": r.text, "score": r.score, "metadata": r.metadata} for r in results]
    reranked = reranker.rerank(query, docs, top_k=RERANK_TOP_K)
    timings["rerank"] = (time.perf_counter() - t0) * 1000

    # Retrieve child → trả về parent (giữ thứ tự, bỏ trùng)
    contexts: list[str] = []
    for r in reranked:
        text = parent_texts.get(r.metadata.get("parent_id"), r.text)
        if text not in contexts:
            contexts.append(text)
    if not contexts:
        contexts = [r.text for r in results[:RERANK_TOP_K]]

    t0 = time.perf_counter()
    if OPENAI_API_KEY and contexts:
        try:
            from openai import OpenAI
            client = OpenAI()
            context_str = "\n\n".join(contexts)
            resp = client.chat.completions.create(model="gpt-4o-mini", temperature=0, messages=[
                {"role": "system", "content": "Trả lời CHỈ dựa trên context. Nếu không có → nói 'Không tìm thấy.'"},
                {"role": "user", "content": f"Context:\n{context_str}\n\nCâu hỏi: {query}"},
            ])
            answer = resp.choices[0].message.content
        except Exception as e:
            print(f"  ⚠️  LLM generation failed: {e}", flush=True)
            answer = contexts[0]
    else:
        answer = contexts[0] if contexts else "Không tìm thấy thông tin."
    timings["generate"] = (time.perf_counter() - t0) * 1000
    timings["total"] = sum(timings.values())
    return answer, contexts, timings


def _summarize_latency(per_query: list[dict[str, float]]) -> dict[str, dict[str, float]]:
    summary = {}
    for step in per_query[0]:
        values = sorted(q[step] for q in per_query)
        p95_idx = min(len(values) - 1, round(0.95 * (len(values) - 1)))
        summary[step] = {
            "avg_ms": round(sum(values) / len(values), 1),
            "p95_ms": round(values[p95_idx], 1),
            "max_ms": round(values[-1], 1),
        }
    return summary


def evaluate_pipeline(search: HybridSearch, reranker: CrossEncoderReranker,
                      parent_texts: dict[str, str], build_ms: dict[str, float] | None = None):
    """Run evaluation on test set."""
    test_set = load_test_set()
    print(f"\n[Eval] Running {len(test_set)} queries...", flush=True)
    questions, answers, all_contexts, ground_truths = [], [], [], []
    per_query_ms = []

    for i, item in enumerate(test_set):
        answer, contexts, timings = run_query(item["question"], search, reranker, parent_texts)
        per_query_ms.append(timings)
        questions.append(item["question"])
        answers.append(answer)
        all_contexts.append(contexts)
        ground_truths.append(item["ground_truth"])
        print(f"  [{i+1}/{len(test_set)}] {item['question'][:50]}...", flush=True)

    # Latency breakdown (bonus): thời gian build + từng bước của query
    latency = {"build_ms": {k: round(v, 1) for k, v in (build_ms or {}).items()},
               "query_ms": _summarize_latency(per_query_ms)}
    os.makedirs(os.path.dirname(LATENCY_REPORT_PATH), exist_ok=True)
    with open(LATENCY_REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(latency, f, ensure_ascii=False, indent=2)
    print("\nLATENCY BREAKDOWN (ms)")
    for step, s in latency["query_ms"].items():
        print(f"  {step:<10} avg={s['avg_ms']:>9.1f}  p95={s['p95_ms']:>9.1f}  max={s['max_ms']:>9.1f}")
    print(f"  Report saved to {LATENCY_REPORT_PATH}")

    t0 = time.time()
    print(f"\n[Eval] Running RAGAS (4 metrics × {len(test_set)} questions)...", flush=True)
    results = evaluate_ragas(questions, answers, all_contexts, ground_truths)
    print(f"  ✓ RAGAS done ({time.time()-t0:.1f}s)", flush=True)

    print("\n" + "=" * 60)
    print("PRODUCTION RAG SCORES")
    print("=" * 60)
    for m in ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]:
        s = results.get(m, 0)
        print(f"  {'✓' if s >= 0.75 else '✗'} {m}: {s:.4f}")

    failures = failure_analysis(results.get("per_question", []), bottom_n=5)
    save_report(results, failures)
    return results


if __name__ == "__main__":
    start = time.time()
    search, reranker, parent_texts, build_ms = build_pipeline()
    evaluate_pipeline(search, reranker, parent_texts, build_ms)
    print(f"\nTotal: {time.time() - start:.1f}s")
