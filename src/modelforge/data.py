"""Read strictly separated, original synthetic records."""

import json
from pathlib import Path


def read_jsonl(path: str | Path) -> list[dict]:
    with Path(path).open(encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    if not rows:
        raise ValueError(f"empty dataset: {path}")
    return rows


def tickets(data_dir: str | Path, split: str) -> list[dict]:
    if split not in ("train", "validation", "test"):
        raise ValueError(f"unknown split: {split}")
    return read_jsonl(Path(data_dir) / f"tickets.{split}.jsonl")


def passages(data_dir: str | Path) -> list[dict]:
    return read_jsonl(Path(data_dir) / "passages.jsonl")


def queries(data_dir: str | Path, split: str) -> list[dict]:
    if split not in ("validation", "test"):
        raise ValueError(f"unknown retrieval split: {split}")
    return read_jsonl(Path(data_dir) / f"queries.{split}.jsonl")
