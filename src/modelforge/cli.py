"""Small command-line surface; large downloads/training happen only when invoked."""

import argparse
import json
from pathlib import Path

from modelforge.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(prog="modelforge", description="NLP/model adaptation lab")
    parser.add_argument("--config", default="configs/default.yaml", help="YAML configuration")
    command = parser.add_subparsers(dest="command", required=True)
    search = command.add_parser("search", help="rank original synthetic troubleshooting passages")
    search.add_argument("--query", required=True)
    search.add_argument("--method", choices=("lexical", "dense"), default="lexical")
    search.add_argument("--top-k", type=int, default=5)
    infer = command.add_parser("infer", help="run a local generative model")
    infer.add_argument("--ticket", required=True)
    infer.add_argument("--variant", choices=("base", "prompted", "lora"), default="base")
    infer.add_argument("--adapter-dir")
    command.add_parser("train", help="fine-tune and save a LoRA adapter")
    token = command.add_parser("tokens", help="inspect tokenizer lengths and truncation risk")
    token.add_argument("--text", required=True)
    retrieval = command.add_parser("evaluate-retrieval")
    retrieval.add_argument("--split", choices=("validation", "test"), default="test")
    retrieval.add_argument("--k", type=int, default=5)
    retrieval.add_argument("--output", default="outputs/retrieval.json")
    generation = command.add_parser("evaluate-generation")
    generation.add_argument("--split", choices=("validation", "test"), default="test")
    generation.add_argument("--variants", nargs="+", choices=("base", "prompted", "lora"),
                            default=["base", "prompted"])
    generation.add_argument("--adapter-dir")
    generation.add_argument("--output", default="outputs/generation.json")
    args = parser.parse_args()
    config = load_config(args.config)
    if args.command == "search":
        from modelforge.retrieval import Retriever

        result = Retriever(
            config.data_dir, args.method, config.embedding_model,
            cache_dir=Path(config.output_dir).parent / "cache",
        ).search(args.query, args.top_k)
    elif args.command == "infer":
        from modelforge.generation import generate

        result = generate(config, args.ticket, args.variant, args.adapter_dir)
    elif args.command == "train":
        from modelforge.training import train

        result = train(config)
    elif args.command == "tokens":
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        ids = tokenizer(args.text, add_special_tokens=False)["input_ids"]
        result = {"model": config.model_name, "tokens": tokenizer.convert_ids_to_tokens(ids),
                  "count": len(ids), "max_length": config.max_length,
                  "would_truncate_as_full_input": len(ids) > config.max_length,
                  "note": "Training counts full prompt plus answer; it rejects overflow."}
    elif args.command == "evaluate-retrieval":
        from modelforge.evaluation import evaluate_retrieval, write_result

        result = evaluate_retrieval(config, args.split, args.k)
        write_result(result, args.output)
    else:
        from modelforge.evaluation import evaluate_generation, write_result

        result = evaluate_generation(config, args.split, tuple(args.variants), args.adapter_dir)
        write_result(result, args.output)
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
