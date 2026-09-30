"""Completion-only supervised tuning with PEFT; validation never supplies optimizer data."""

import hashlib
import json
import random
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import Trainer, TrainingArguments

from modelforge.config import Config
from modelforge.data import tickets
from modelforge.generation import answer_for, load_model, prompt_for


def encode_example(tokenizer, row: dict, max_length: int) -> dict:
    prefix = tokenizer(prompt_for(tokenizer, row["text"]), add_special_tokens=False)["input_ids"]
    completion = tokenizer(answer_for(row) + tokenizer.eos_token, add_special_tokens=False)[
        "input_ids"
    ]
    if len(prefix) + len(completion) > max_length:
        raise ValueError(
            f"ticket {row['id']} has {len(prefix) + len(completion)} tokens; "
            f"raise max_length rather than discard an answer or instruction"
        )
    return {
        "input_ids": prefix + completion,
        "attention_mask": [1] * (len(prefix) + len(completion)),
        "labels": [-100] * len(prefix) + completion,
    }


class CompletionCollator:
    def __init__(self, pad_id: int):
        self.pad_id = pad_id

    def __call__(self, features: list[dict]) -> dict:
        longest = max(len(item["input_ids"]) for item in features)
        return {
            "input_ids": torch.tensor([
                item["input_ids"] + [self.pad_id] * (longest - len(item["input_ids"]))
                for item in features
            ], dtype=torch.long),
            "attention_mask": torch.tensor([
                item["attention_mask"] + [0] * (longest - len(item["attention_mask"]))
                for item in features
            ], dtype=torch.long),
            "labels": torch.tensor([
                item["labels"] + [-100] * (longest - len(item["labels"]))
                for item in features
            ], dtype=torch.long),
        }


def fingerprint(data_dir: str) -> str:
    digest = hashlib.sha256()
    for name in ("tickets.train.jsonl", "tickets.validation.jsonl"):
        digest.update(Path(data_dir, name).read_bytes())
    return digest.hexdigest()


def train(config: Config) -> dict:
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
    train_rows = tickets(config.data_dir, "train")
    validation_rows = tickets(config.data_dir, "validation")
    if {row["id"] for row in train_rows} & {row["id"] for row in validation_rows}:
        raise ValueError("training and validation ticket IDs overlap")

    tokenizer, model = load_model(config)
    model.train()
    model.config.use_cache = False
    if config.quantization == "4bit":
        model = prepare_model_for_kbit_training(model)
    lora_config = LoraConfig(
        r=config.lora_r, lora_alpha=config.lora_alpha, lora_dropout=config.lora_dropout,
        bias="none", target_modules="all-linear", task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    total = sum(parameter.numel() for parameter in model.parameters())
    train_data = Dataset.from_list([
        encode_example(tokenizer, row, config.max_length) for row in train_rows
    ])
    validation_data = Dataset.from_list([
        encode_example(tokenizer, row, config.max_length) for row in validation_rows
    ])
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    args = TrainingArguments(
        output_dir=str(output_dir / "checkpoints"),
        seed=config.seed, data_seed=config.seed,
        num_train_epochs=config.epochs,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        eval_strategy="epoch", save_strategy="epoch", save_total_limit=2,
        logging_steps=5, report_to="none", optim="adamw_torch",
        fp16=config.quantization == "4bit" and not torch.cuda.is_bf16_supported(),
        bf16=config.quantization == "4bit" and torch.cuda.is_bf16_supported(),
    )
    trainer = Trainer(
        model=model, args=args, train_dataset=train_data, eval_dataset=validation_data,
        data_collator=CompletionCollator(tokenizer.pad_token_id),
    )
    train_result = trainer.train()
    validation_result = trainer.evaluate()
    adapter_dir = output_dir / "adapter"
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    manifest = {
        "config": asdict(config), "dataset_sha256": fingerprint(config.data_dir),
        "train_examples": len(train_rows), "validation_examples": len(validation_rows),
        "trainable_parameters": trainable, "all_parameters": total,
        "train_metrics": train_result.metrics, "validation_metrics": validation_result,
        "adapter_dir": str(adapter_dir),
    }
    (output_dir / "run.json").write_text(json.dumps(manifest, indent=2, default=str) + "\n")
    return manifest
