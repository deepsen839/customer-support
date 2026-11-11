# Project 2 — PPO Inventory Replenishment Agent

## Objective
Train a reinforcement-learning agent to choose daily replenishment quantities in a stochastic inventory environment.

## State
Current inventory, previous demand, and day-of-year seasonality features.

## Action
A discrete replenishment quantity from 0 to `max_order`.

## Reward
Negative sum of holding cost, lost-sales penalty, and order cost. The agent is rewarded for balancing stock availability against excess inventory and purchasing cost.

## Run

```bash
pip install -r requirements.txt
python train_ppo.py --total_timesteps 100000 --output_dir ./outputs
python evaluate_policy.py --model_path ./outputs/ppo_inventory.pt --episodes 100
```

Training writes episodic rewards, a learning curve, and a PyTorch checkpoint. Evaluation compares PPO with a simple reorder-point baseline on the same held-out seed set.

## Limitations
The simulator uses synthetic demand and simplified costs. It is a demonstration of PPO implementation, not a validated supply-chain optimizer. Calibrate the environment with historical data and compare against operational baselines before deployment.
