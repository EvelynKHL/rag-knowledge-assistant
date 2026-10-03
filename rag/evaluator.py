import json
from pathlib import Path

def evaluate_queries(retriever):
    path = Path("data/evaluation/questions.json")

    if not path.exists():
        return {
            "status": "missing_evaluation_file",
            "message": "Create data/evaluation/questions.json first."
        }

    questions = json.loads(path.read_text(encoding="utf-8"))

    total = len(questions)
    retrieval_hits = 0
    source_hits = 0

    details = []

    for item in questions:
        q = item["question"]
        expected_keywords = [x.lower() for x in item.get("keywords", [])]
        expected_source = item.get("source")

        results = retriever.search(q, k=5)
        combined = " ".join(r["text"].lower() for r in results)

        keyword_hit = (
            all(k in combined for k in expected_keywords)
            if expected_keywords else bool(results)
        )

        source_hit = (
            any(r["source"] == expected_source for r in results)
            if expected_source else None
        )

        retrieval_hits += int(keyword_hit)
        if source_hit is not None:
            source_hits += int(source_hit)

        details.append({
            "question": q,
            "retrieved": len(results),
            "keyword_hit": keyword_hit,
            "source_hit": source_hit,
            "top_source": results[0]["source"] if results else None,
        })

    return {
        "queries": total,
        "retrieval_keyword_success_rate": round(
            retrieval_hits / total * 100, 2
        ) if total else 0,
        "source_success_rate": round(
            source_hits / total * 100, 2
        ) if total and any("source" in x for x in questions) else None,
        "details": details,
    }
