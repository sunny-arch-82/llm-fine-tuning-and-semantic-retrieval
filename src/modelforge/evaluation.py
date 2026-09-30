"""On-demand metrics and inspectable per-example records; no fabricated results."""

import json
import math
from collections import Counter
from pathlib import Path

from sklearn.metrics import f1_score

from modelforge.config import Config
from modelforge.data import queries, tickets
from modelforge.generation import infer_loaded, load_model
from modelforge.retrieval import Retriever


def ranking_metrics(ranked_ids: list[str], relevance: dict[str, int], k: int) -> dict:
    if k < 1:
        raise ValueError("k must be positive")
    relevant = {key for key, grade in relevance.items() if grade > 0}
    top = ranked_ids[:k]
    hits = sum(item in relevant for item in top)
    first = next((index + 1 for index, item in enumerate(top) if item in relevant), None)

    def dcg(grades: list[int]) -> float:
        return sum((2**grade - 1) / math.log2(index + 2) for index, grade in enumerate(grades))

    ideal = dcg(sorted(relevance.values(), reverse=True)[:k])
    actual = dcg([relevance.get(item, 0) for item in top])
    return {
        f"recall_at_{k}": hits / len(relevant) if relevant else None,
        f"precision_at_{k}": hits / k if relevant else None,
        f"mrr_at_{k}": 1 / first if first else 0.0,
        f"ndcg_at_{k}": actual / ideal if ideal else None,
    }


def evaluate_retrieval(config: Config, split: str = "test", k: int = 5) -> dict:
    examples = queries(config.data_dir, split)
    results: dict = {"split": split, "k": k, "methods": {}}
    corpus_ids = {row["id"] for row in Retriever(config.data_dir, "lexical").rows}
    for example in examples:
        if not set(example["relevance"]) <= corpus_ids:
            raise ValueError(f"query {example['id']} has unknown passage IDs")
    for method in ("lexical", "dense"):
        retriever = Retriever(
            config.data_dir, method, config.embedding_model,
            cache_dir=Path(config.output_dir).parent / "cache",
        )
        records = []
        for example in examples:
            ranking = retriever.search(example["text"], k)
            ids = [row["id"] for row in ranking]
            records.append({
                "id": example["id"], "ranked_ids": ids,
                "relevance": example["relevance"],
                "metrics": ranking_metrics(ids, example["relevance"], k),
            })
        names = records[0]["metrics"]
        summary = {
            name: sum(record["metrics"][name] for record in records) / len(records)
            for name in names
        }
        results["methods"][method] = {"summary": summary, "records": records}
    return results


def evaluate_generation(
    config: Config, split: str = "test", variants: tuple[str, ...] = ("base", "prompted"),
    adapter_dir: str | None = None,
) -> dict:
    if split not in ("validation", "test"):
        raise ValueError("only held-out validation and test splits are evaluable")
    if not variants or set(variants) - {"base", "prompted", "lora"}:
        raise ValueError("variants must be base, prompted, and/or lora")
    if "lora" in variants and not adapter_dir:
        raise ValueError("lora evaluation requires --adapter-dir")
    rows = tickets(config.data_dir, split)
    train_rows = tickets(config.data_dir, "train")
    if {row["id"] for row in rows} & {row["id"] for row in train_rows}:
        raise ValueError("evaluation IDs overlap training IDs")
    labels = sorted({row["category"] for row in train_rows})
    output = {"split": split, "records": {}, "summary": {},
              "rubric": "Review each response for diagnostic usefulness, uncertainty, and unsafe actions."}
    for variant in variants:
        tokenizer, model = load_model(config, adapter_dir if variant == "lora" else None)
        records = []
        for row in rows:
            prediction = infer_loaded(config, tokenizer, model, row["text"], variant)
            parsed = prediction["parsed"]
            records.append({
                "id": row["id"], "ticket": row["text"],
                "expected": {"category": row["category"], "first_check": row["first_check"]},
                "prediction": prediction,
                "category_correct": parsed["category"] == row["category"],
                "check_correct": parsed["first_check"] == row["first_check"],
                "format_valid": all(parsed.values()),
                "manual_review": {"diagnostic_usefulness": None, "uncertainty": None,
                                  "unsafe_action": None, "notes": ""},
            })
        predicted = [r["prediction"]["parsed"]["category"] or "__INVALID__" for r in records]
        expected = [r["expected"]["category"] for r in records]
        output["records"][variant] = records
        output["summary"][variant] = {
            "count": len(records),
            "category_accuracy": sum(r["category_correct"] for r in records) / len(records),
            "category_macro_f1": float(f1_score(
                expected, predicted, labels=labels, average="macro", zero_division=0
            )),
            "first_check_accuracy": sum(r["check_correct"] for r in records) / len(records),
            "format_rate": sum(r["format_valid"] for r in records) / len(records),
            "mean_latency_seconds": sum(
                r["prediction"]["latency_seconds"] for r in records
            ) / len(records),
            "output_token_count": sum(r["prediction"]["output_tokens"] for r in records),
            "device_counts": dict(Counter(r["prediction"]["device"] for r in records)),
        }
        del model, tokenizer
    return output


def write_result(result: dict, path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, indent=2, default=str) + "\n", encoding="utf-8")
