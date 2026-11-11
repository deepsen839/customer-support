import argparse, os, csv, random
import numpy as np
import torch
from ppo import ActorCritic
from inventory_env import InventoryEnv
import matplotlib.pyplot as plt

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--total_timesteps", type=int, default=100000)
    p.add_argument("--rollout_steps", type=int, default=1024)
    p.add_argument("--update_epochs", type=int, default=8)
    p.add_argument("--minibatch_size", type=int, default=256)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--gae_lambda", type=float, default=0.95)
    p.add_argument("--clip_coef", type=float, default=0.2)
    p.add_argument("--output_dir", default="./outputs")
    args = p.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    random.seed(7); np.random.seed(7); torch.manual_seed(7)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = InventoryEnv(seed=7)
    net = ActorCritic(env.obs_dim, env.action_dim).to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr, eps=1e-5)
    obs = env.reset()
    global_step, episode_return, episode_cost = 0, 0.0, 0.0
    returns, episode_costs = [], []
    logs = []

    while global_step < args.total_timesteps:
        n = min(args.rollout_steps, args.total_timesteps - global_step)
        ob = np.zeros((n, env.obs_dim), dtype=np.float32)
        ac = np.zeros(n, dtype=np.int64)
        lp = np.zeros(n, dtype=np.float32)
        rew = np.zeros(n, dtype=np.float32)
        done_arr = np.zeros(n, dtype=np.float32)
        val = np.zeros(n, dtype=np.float32)

        for t in range(n):
            ob[t] = obs
            with torch.no_grad():
                a, logp, v = net.act(torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0))
            action = int(a.item())
            next_obs, reward, done, info = env.step(action)
            ac[t], lp[t], rew[t], val[t] = action, logp.item(), reward, v.item()
            episode_return += reward
            episode_cost += info["cost"]
            global_step += 1
            obs = next_obs
            done_arr[t] = float(done)
            if done:
                returns.append(episode_return); episode_costs.append(episode_cost)
                logs.append({"episode": len(returns), "return": episode_return, "cost": episode_cost})
                print(f"step={global_step} episode={len(returns)} return={episode_return:.1f} cost={episode_cost:.1f}")
                obs = env.reset()
                episode_return, episode_cost = 0.0, 0.0

        with torch.no_grad():
            _, next_value = net(torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0))
            next_value = next_value.item()
        advantages = np.zeros(n, dtype=np.float32)
        lastgaelam = 0.0
        for t in reversed(range(n)):
            if t == n - 1:
                next_nonterminal = 1.0 - done_arr[t]
                nextv = next_value
            else:
                next_nonterminal = 1.0 - done_arr[t]
                nextv = val[t+1]
            delta = rew[t] + args.gamma * nextv * next_nonterminal - val[t]
            lastgaelam = delta + args.gamma * args.gae_lambda * next_nonterminal * lastgaelam
            advantages[t] = lastgaelam
        returns_t = advantages + val

        b_obs = torch.tensor(ob, device=device)
        b_actions = torch.tensor(ac, device=device)
        b_logprobs = torch.tensor(lp, device=device)
        b_adv = torch.tensor(advantages, device=device)
        b_returns = torch.tensor(returns_t, device=device)
        b_adv = (b_adv - b_adv.mean()) / (b_adv.std() + 1e-8)

        inds = np.arange(n)
        for _ in range(args.update_epochs):
            np.random.shuffle(inds)
            for start in range(0, n, args.minibatch_size):
                mb = inds[start:start+args.minibatch_size]
                newlp, entropy, newv = net.evaluate_actions(b_obs[mb], b_actions[mb])
                logratio = newlp - b_logprobs[mb]
                ratio = logratio.exp()
                pg1 = -b_adv[mb] * ratio
                pg2 = -b_adv[mb] * torch.clamp(ratio, 1-args.clip_coef, 1+args.clip_coef)
                policy_loss = torch.max(pg1, pg2).mean()
                value_loss = 0.5 * ((newv - b_returns[mb]) ** 2).mean()
                entropy_loss = entropy.mean()
                loss = policy_loss + 0.5 * value_loss - 0.01 * entropy_loss
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 0.5)
                optimizer.step()

    model_path = os.path.join(args.output_dir, "ppo_inventory.pt")
    torch.save({"model_state_dict": net.state_dict(), "obs_dim": env.obs_dim,
                "action_dim": env.action_dim, "max_order": env.max_order}, model_path)
    with open(os.path.join(args.output_dir, "episode_metrics.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["episode", "return", "cost"])
        writer.writeheader(); writer.writerows(logs)
    if returns:
        plt.figure(figsize=(9,4)); plt.plot(returns, alpha=.35, label="Episode return")
        if len(returns) >= 10:
            smooth = np.convolve(returns, np.ones(10)/10, mode="valid")
            plt.plot(range(9, len(returns)), smooth, label="10-episode mean")
        plt.xlabel("Episode"); plt.ylabel("Return (higher is better)")
        plt.title("PPO Inventory Agent Training"); plt.legend(); plt.tight_layout()
        plt.savefig(os.path.join(args.output_dir, "learning_curve.png"), dpi=150)
    print("Saved model:", model_path, "device:", device)

if __name__ == "__main__": main()
