"""Single configuration contract for retrieval, inference, and training."""

from dataclasses import dataclass, fields
from pathlib import Path
from typing import Literal

import yaml


@dataclass(frozen=True)
class Config:
    seed: int = 20260929
    model_name: str = "HuggingFaceTB/SmolLM2-135M-Instruct"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    data_dir: str = "data/sample"
    output_dir: str = "outputs/lora-demo"
    max_length: int = 256
    max_new_tokens: int = 80
    batch_size: int = 2
    gradient_accumulation_steps: int = 2
    learning_rate: float = 2e-4
    epochs: int = 1
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    quantization: Literal["none", "4bit"] = "none"

    def __post_init__(self) -> None:
        if self.quantization not in ("none", "4bit"):
            raise ValueError("quantization must be none or 4bit")
        if any(getattr(self, key) < 1 for key in (
            "max_length", "max_new_tokens", "batch_size", "gradient_accumulation_steps",
            "epochs", "lora_r", "lora_alpha",
        )):
            raise ValueError("lengths, counts, epochs and LoRA dimensions must be positive")
        if self.learning_rate <= 0 or not 0 <= self.lora_dropout < 1:
            raise ValueError("learning_rate must be positive and dropout in [0, 1)")


def load_config(path: str | Path = "configs/default.yaml") -> Config:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("configuration must be a YAML mapping")
    allowed = {field.name for field in fields(Config)}
    unknown = raw.keys() - allowed
    if unknown:
        raise ValueError(f"unknown configuration fields: {sorted(unknown)}")
    return Config(**raw)
