import argparse, json, os, time
import torch
from data import build_datasets
from transformers import (AutoTokenizer, AutoModelForCausalLM, TrainingArguments,
                          Trainer, DataCollatorForSeq2Seq, BitsAndBytesConfig)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_name", default="Qwen/Qwen2.5-0.5B-Instruct")
    p.add_argument("--output_dir", default="./outputs")
    p.add_argument("--epochs", type=float, default=2)
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--grad_accum", type=int, default=8)
    p.add_argument("--max_length", type=int, default=384)
    p.add_argument("--max_train_samples", type=int, default=0)
    p.add_argument("--max_eval_samples", type=int, default=0)
    p.add_argument("--learning_rate", type=float, default=2e-4)
    return p.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    train_ds, eval_ds = build_datasets()
    if args.max_train_samples:
        train_ds = train_ds.select(range(min(args.max_train_samples, len(train_ds))))
    if args.max_eval_samples:
        eval_ds = eval_ds.select(range(min(args.max_eval_samples, len(eval_ds))))

    def tokenize(batch):
        texts = [tokenizer.apply_chat_template(m, tokenize=False, add_generation_prompt=False)
                 for m in batch["messages"]]
        model_inputs = tokenizer(texts, max_length=args.max_length, truncation=True)
        labels = []
        for msgs, ids in zip(batch["messages"], model_inputs["input_ids"]):
            # Mask everything through the assistant header; supervise assistant response only.
            prompt = tokenizer.apply_chat_template(msgs[:2], tokenize=True, add_generation_prompt=True)
            prefix_len = min(len(prompt), len(ids))
            labels.append([-100] * prefix_len + ids[prefix_len:])
        model_inputs["labels"] = labels
        return model_inputs

    train_tok = train_ds.map(tokenize, batched=True, remove_columns=train_ds.column_names)
    eval_tok = eval_ds.map(tokenize, batched=True, remove_columns=eval_ds.column_names)

    quant_config = None
    use_4bit = torch.cuda.is_available()
    if use_4bit:
        try:
            import bitsandbytes  # noqa
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True
            )
        except Exception as e:
            print("4-bit unavailable; using regular model:", repr(e))
            use_4bit = False

    load_kwargs = {"torch_dtype": torch.float16 if torch.cuda.is_available() else torch.float32}
    if use_4bit and quant_config is not None:
        load_kwargs["quantization_config"] = quant_config
        load_kwargs["device_map"] = "auto"
    model = AutoModelForCausalLM.from_pretrained(args.model_name, **load_kwargs)
    if use_4bit:
        model = prepare_model_for_kbit_training(model)
    model.config.use_cache = False
    lora = LoraConfig(
        r=8, lora_alpha=16, lora_dropout=0.05, bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )
    model = get_peft_model(model, lora)
    model.enable_input_require_grads()
    model.print_trainable_parameters()
    trainable = [p for p in model.parameters() if p.requires_grad]
    print("Trainable tensors:", len(trainable))
    print("Gradients enabled:", any(p.requires_grad for p in trainable))

    import inspect
    supported = inspect.signature(TrainingArguments).parameters
    training_kwargs = {
        "output_dir": args.output_dir,
        "num_train_epochs": args.epochs,
        "per_device_train_batch_size": args.batch_size,
        "per_device_eval_batch_size": args.batch_size,
        "gradient_accumulation_steps": args.grad_accum,
        "learning_rate": args.learning_rate,
        "logging_steps": 10,
        "save_strategy": "epoch",
        "report_to": "none",
        "fp16": torch.cuda.is_available(),
        "gradient_checkpointing": False,
        "warmup_steps": 5,
    }
    if "eval_strategy" in supported:
        training_kwargs["eval_strategy"] = "epoch"
    elif "evaluation_strategy" in supported:
        training_kwargs["evaluation_strategy"] = "epoch"
    training_kwargs = {k: v for k, v in training_kwargs.items()
                       if k in supported}
    training_args = TrainingArguments(**training_kwargs)

    trainer = Trainer(
        model=model, args=training_args,
        train_dataset=train_tok, eval_dataset=eval_tok,
        data_collator=DataCollatorForSeq2Seq(tokenizer, pad_to_multiple_of=8)
    )
    started = time.time()
    result = trainer.train()
    elapsed = time.time() - started
    final_dir = os.path.join(args.output_dir, "final_adapter")
    trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)
    metrics = dict(result.metrics)
    metrics["elapsed_seconds"] = elapsed
    with open(os.path.join(args.output_dir, "train_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps(metrics, indent=2))
    print("Adapter saved to", final_dir)

if __name__ == "__main__":
    main()
