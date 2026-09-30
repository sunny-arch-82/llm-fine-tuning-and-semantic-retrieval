"""Shared training/inference prompt and causal-LM loading."""

import re
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from modelforge.config import Config
from modelforge.data import tickets


INSTRUCTION = """Classify the operations ticket using one category/check pair:
    DB_POOL: INSPECT_CONNECTION_HOLDERS
    UPSTREAM_TIMEOUT: COMPARE_DEPENDENCY_LATENCY
    RESTART_LOOP: INSPECT_CONTAINER_TERMINATION
    TLS_CERT: INSPECT_CERTIFICATE_CHAIN
    QUEUE_BACKLOG: COMPARE_ARRIVAL_DRAIN_RATE
    DISK_FULL: INSPECT_FILESYSTEM_USAGE
    DNS_FAILURE: INSPECT_NAME_RESOLUTION
    RATE_LIMIT: INSPECT_QUOTA_HEADERS

    Return exactly three lines:
        CATEGORY: chosen category
        FIRST_CHECK: corresponding check
        RATIONALE: one short explanation of why this check is useful.
        Do not repeat the ticket or claim the cause is proven."""


def answer_for(row: dict) -> str:
    return (
        f"CATEGORY: {row['category']}\nFIRST_CHECK: {row['first_check']}\n"
        f"RATIONALE: {row['reason']}"
    )


def prompt_for(tokenizer, text: str, example: dict | None = None) -> str:
    content = INSTRUCTION
    if example is not None:
        content += f"\nExample ticket: {example['text']}\nExample answer:\n{answer_for(example)}"
    content += f"\nTicket: {text}\nAnswer:"
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": content}], tokenize=False, add_generation_prompt=True
        )
    return content + "\n"


def parse_answer(text: str) -> dict:
    def field(name: str) -> str | None:
        match = re.search(rf"(?im)^\s*{name}:\s*([^\r\n]+)", text)
        return match.group(1).strip() if match else None

    return {"category": field("CATEGORY"), "first_check": field("FIRST_CHECK"),
            "rationale": field("RATIONALE")}


def load_model(config: Config, adapter_dir: str | Path | None = None):
    if config.quantization == "4bit" and not torch.cuda.is_available():
        raise RuntimeError("4-bit bitsandbytes requires a compatible CUDA GPU")
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.eos_token_id is None:
        raise ValueError("base model must supply an EOS token")
    kwargs = {}
    if config.quantization == "4bit":
        kwargs["device_map"] = {"": 0}
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        )
    model = AutoModelForCausalLM.from_pretrained(config.model_name, **kwargs)
    if adapter_dir is not None:
        from peft import PeftConfig, PeftModel

        adapter = PeftConfig.from_pretrained(str(adapter_dir))
        if adapter.base_model_name_or_path != config.model_name:
            raise ValueError("adapter base model differs from configured model")
        model = PeftModel.from_pretrained(model, str(adapter_dir))
    if config.quantization == "none":
        device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        model.to(device)
    model.eval()
    return tokenizer, model


def generate(
    config: Config,
    text: str,
    variant: str = "base",
    adapter_dir: str | Path | None = None,
) -> dict:
    if variant not in {"base", "prompted", "lora"}:
        raise ValueError("variant must be base, prompted or lora")
    if variant == "lora" and adapter_dir is None:
        raise ValueError("lora variant requires --adapter-dir")
    tokenizer, model = load_model(config, adapter_dir if variant == "lora" else None)
    return infer_loaded(config, tokenizer, model, text, variant)


def infer_loaded(config: Config, tokenizer, model, text: str, variant: str) -> dict:
    """Run a single deterministic decode with an already loaded model."""
    example = tickets(config.data_dir, "train")[0] if variant == "prompted" else None
    prompt = prompt_for(tokenizer, text, example)
    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    if inputs["input_ids"].shape[1] > config.max_length:
        raise ValueError("prompt exceeds max_length; increase it rather than silently truncating")
    device = model.device
    inputs = {key: value.to(device) for key, value in inputs.items()}
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    start = time.perf_counter()
    with torch.inference_mode():
        generated = model.generate(
            **inputs, max_new_tokens=config.max_new_tokens,
            do_sample=False, pad_token_id=tokenizer.eos_token_id,
        )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    seconds = time.perf_counter() - start
    new_ids = generated[0, inputs["input_ids"].shape[1]:]
    output = tokenizer.decode(new_ids, skip_special_tokens=True).strip()
    return {
        "variant": variant, "model": config.model_name, "input": text,
        "output": output, "parsed": parse_answer(output),
        "input_tokens": int(inputs["input_ids"].shape[1]),
        "output_tokens": int(new_ids.numel()),
        "latency_seconds": seconds,
        "tokens_per_second": float(new_ids.numel() / seconds) if seconds else None,
        "cuda_peak_memory_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
        "device": str(device),
    }
