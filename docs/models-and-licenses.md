# Models, provenance and license boundaries

| Asset | Purpose | Upstream license/reference | Packaged? |
| --- | --- | --- | --- |
| Original fictional tickets/guides | Domain examples and labels | Project MIT license; authored in `scripts/build_synthetic_data.py` | Yes |
| [SmolLM2-135M-Instruct](https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct) | Small generative model | Model card indicates Apache-2.0 | No |
| [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) | Pretrained dense embeddings | Model card indicates Apache-2.0 | No |

The default model identifiers are in `configs/default.yaml`. Hugging Face Hub downloads happen on the user's first model/tokenizer command, not during install. No external dataset, proprietary ticket, model weight, or adapter is shipped. Your trained adapter belongs in ignored `outputs/`; check the base model license and your data rights before distribution.

Direct libraries are PyTorch, Transformers, Datasets, Sentence Transformers, PEFT, Accelerate, scikit-learn, NumPy and PyYAML; bitsandbytes is an optional extra for compatible CUDA. Their versions and transitive dependencies are resolved in `uv.lock`, and each upstream package retains its own license. Before publishing a redistributable binary or shipping dependencies, inspect the exact locked versions and their license files. The repository's MIT grant applies only to the original code and synthetic data.

Implementation references: [Hugging Face PEFT LoRA](https://huggingface.co/docs/peft/main/package_reference/lora), [PEFT quantization](https://huggingface.co/docs/peft/developer_guides/quantization), and [Transformers bitsandbytes](https://huggingface.co/docs/transformers/en/quantization/bitsandbytes). Optional 4-bit NF4 adapters follow PEFT's quantized-base training guidance; compatibility has not been tested in this delivery.
