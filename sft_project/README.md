# Project 1 — SFT Customer Support Assistant

## Objective
Fine-tune `Qwen/Qwen2.5-0.5B-Instruct` on a small synthetic customer-support instruction dataset using LoRA. The dataset is intentionally transparent and safe to regenerate.

## Run

```bash
pip install -r ../requirements.txt
python train_sft.py --epochs 2 --output_dir ./outputs
python evaluate_sft.py --adapter_dir ./outputs/final_adapter
```

For a quick smoke test:
`python train_sft.py --max_train_samples 80 --max_eval_samples 20 --epochs 1`

## What the code does
- Builds train/evaluation examples from a deterministic synthetic dataset.
- Applies the model's chat template.
- Masks prompt tokens so loss is calculated on assistant responses only.
- Trains LoRA adapters; attempts 4-bit loading on CUDA if bitsandbytes is usable.
- Saves adapter, tokenizer, training log, and evaluation predictions.

## Metrics
The training script logs loss and elapsed time. The evaluation script reports exact-match and token-overlap F1 as lightweight diagnostics. These are not substitutes for human evaluation, factuality checks, or domain-specific safety tests.

## Limitations
Synthetic examples do not establish production quality. Replace them with reviewed, permissioned customer-support conversations, remove personal data, add refusal/escalation examples, and create a held-out evaluation set before deployment.
