import argparse, json
import numpy as np
import torch
from ppo import ActorCritic
from inventory_env import InventoryEnv

def run_episode(env, policy, seed, baseline=False):
    obs = env.reset(seed=seed)
    total_cost, lost, ordered = 0.0, 0.0, 0.0
    done = False
    while not done:
        if baseline:
            # Simple reorder-up-to policy; target stock is approximately 20 units.
            action = int(np.clip(round(20 - env.inventory), 0, env.max_order))
        else:
            with torch.no_grad():
                logits, _ = policy(torch.tensor(obs, dtype=torch.float32).unsqueeze(0))
                action = int(torch.argmax(logits, dim=-1).item())
        obs, reward, done, info = env.step(action)
        total_cost += info["cost"]; lost += info["lost_sales"]; ordered += info["order"]
    return {"cost": total_cost, "lost_sales": lost, "units_ordered": ordered}

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", required=True)
    p.add_argument("--episodes", type=int, default=100)
    args = p.parse_args()
    ckpt = torch.load(args.model_path, map_location="cpu")
    policy = ActorCritic(ckpt["obs_dim"], ckpt["action_dim"])
    policy.load_state_dict(ckpt["model_state_dict"]); policy.eval()
    ppo_results, base_results = [], []
    for seed in range(1000, 1000 + args.episodes):
        ppo_results.append(run_episode(InventoryEnv(seed=seed), policy, seed, False))
        base_results.append(run_episode(InventoryEnv(seed=seed), None, seed, True))
    def summary(rows):
        return {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}
    report = {"episodes": args.episodes, "ppo": summary(ppo_results),
              "reorder_point_baseline": summary(base_results)}
    print(json.dumps(report, indent=2))
    with open("evaluation_report.json", "w") as f: json.dump(report, f, indent=2)
if __name__ == "__main__": main()
