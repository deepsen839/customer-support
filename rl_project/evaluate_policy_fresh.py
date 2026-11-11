import argparse, json, os
import numpy as np
import torch
from ppo import ActorCritic
from inventory_env import InventoryEnv

def episode(env, policy, seed, baseline=False):
    obs = env.reset(seed=seed)
    totals = dict(cost=0.0, lost_sales=0.0, demand=0.0,
                  sold=0.0, ordered=0.0)
    done = False
    while not done:
        if baseline:
            action = int(np.clip(round(20 - env.inventory), 0, env.max_order))
        else:
            with torch.no_grad():
                state = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
                logits, _ = policy(state)
                action = int(torch.argmax(logits, dim=-1).item())
        obs, _, done, info = env.step(action)
        totals["cost"] += info["cost"]
        totals["lost_sales"] += info["lost_sales"]
        totals["demand"] += info["demand"]
        totals["sold"] += info["sold"]
        totals["ordered"] += info["order"]
    totals["fill_rate_percent"] = (
        100.0 * totals["sold"] / totals["demand"]
        if totals["demand"] else 100.0
    )
    return totals

def summarize(rows):
    return {k: {"mean": float(np.mean([r[k] for r in rows])),
                "std": float(np.std([r[k] for r in rows], ddof=1))}
            for k in rows[0]}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--episodes", type=int, default=500)
    ap.add_argument("--seed_start", type=int, default=20000)
    ap.add_argument("--output_dir", default="/kaggle/working/rl_output_penalty8")
    args = ap.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    ckpt = torch.load(args.model_path, map_location="cpu", weights_only=False)
    policy = ActorCritic(ckpt["obs_dim"], ckpt["action_dim"])
    policy.load_state_dict(ckpt["model_state_dict"])
    policy.eval()

    ppo_rows, base_rows = [], []
    for seed in range(args.seed_start, args.seed_start + args.episodes):
        ppo_rows.append(episode(InventoryEnv(lost_sales_penalty=8.0), policy, seed))
        base_rows.append(episode(InventoryEnv(lost_sales_penalty=8.0), None, seed, True))

    ppo, base = summarize(ppo_rows), summarize(base_rows)
    improvement = 100.0 * (base["cost"]["mean"] - ppo["cost"]["mean"]) / base["cost"]["mean"]
    report = {
        "episodes": args.episodes, "seed_start": args.seed_start,
        "lost_sales_penalty": 8.0, "ppo": ppo, "baseline": base,
        "cost_improvement_percent": float(improvement),
        "interpretation": "Compare cost and fill rate together; lower cost alone does not imply better service."
    }
    out = os.path.join(args.output_dir, "evaluation_report_fresh.json")
    with open(out, "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))
    print("Saved:", out)

if __name__ == "__main__":
    main()
