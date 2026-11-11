import argparse, json, os, re
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from data import ROWS

def normalize(s):
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", "", s.lower())).strip()

def token_f1(pred, gold):
    p, g = normalize(pred).split(), normalize(gold).split()
    if not p or not g: return 0.0
    common = {}
    for w in p: common[w] = common.get(w, 0) + 1
    overlap = 0
    counts = {}
    for w in g: counts[w] = counts.get(w, 0) + 1
    overlap = sum(min(v, counts.get(k, 0)) for k, v in common.items())
    if overlap == 0: return 0.0
    precision, recall = overlap / len(p), overlap / len(g)
    return 2 * precision * recall / (precision + recall)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter_dir", required=True)
    ap.add_argument("--base_model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--max_new_tokens", type=int, default=100)
    ap.add_argument("--limit", type=int, default=10)
    args = ap.parse_args()
    tok = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    base = AutoModelForCausalLM.from_pretrained(
        args.base_model, torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None
    )
    model = PeftModel.from_pretrained(base, args.adapter_dir)
    model.eval()
    scores, rows = [], []
    for q, gold in ROWS[:args.limit]:
        messages = [
            {"role": "system", "content": "You are a concise, helpful customer support assistant. Do not invent account details."},
            {"role": "user", "content": q}
        ]
        prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tok(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False,
                                 pad_token_id=tok.eos_token_id)
        pred = tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        score = token_f1(pred, gold)
        scores.append(score)
        rows.append({"question": q, "reference": gold, "prediction": pred, "token_f1": score})
        print("\nQ:", q, "\nPrediction:", pred, "\nReference:", gold, "\nToken-F1:", round(score, 3))
    report = {"examples": len(rows), "mean_token_f1": sum(scores)/max(1,len(scores)),
              "exact_match": sum(normalize(x["prediction"]) == normalize(x["reference"]) for x in rows)/max(1,len(rows)),
              "predictions": rows}
    os.makedirs(os.path.dirname(os.path.abspath(args.adapter_dir)), exist_ok=True)
    out_path = os.path.join(os.path.dirname(os.path.abspath(args.adapter_dir)), "evaluation_report.json")
    with open(out_path, "w") as f: json.dump(report, f, indent=2)
    print("\nEvaluation:", json.dumps({k:v for k,v in report.items() if k != "predictions"}, indent=2))
    print("Saved:", out_path)

if __name__ == "__main__":
    main()
