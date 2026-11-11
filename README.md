# SFT + Reinforcement Learning Projects (Kaggle GPU)

This repository contains two independently runnable, educational projects:

1. **SFT Customer Support Assistant** — supervised fine-tuning of a small instruction model with LoRA.
2. **RL Inventory Replenishment Agent** — PPO agent trained in a custom inventory simulator.

Both projects include training, evaluation, saved artifacts, and Kaggle instructions. The SFT data is synthetic and deliberately small so the pipeline is reproducible; replace it with permissioned, representative data before making real-world claims or deployment.

## Kaggle setup

1. Create a Kaggle notebook and enable **GPU** in Notebook settings.
2. Enable Internet if your Kaggle account/notebook permits it. The SFT model is downloaded from Hugging Face.
3. Upload/extract this repository as a Kaggle Dataset, or clone it in a notebook with Internet enabled.
4. In a notebook cell, install dependencies:

```bash
%cd /kaggle/working/sft_rl_kaggle_projects
!pip install -q -r requirements.txt
```

If you uploaded the zip as a Kaggle Dataset, unzip it under `/kaggle/working` and `cd` into the extracted folder.

### Run SFT

```bash
%cd /kaggle/working/sft_rl_kaggle_projects/sft_project
!python train_sft.py --output_dir /kaggle/working/sft_output --epochs 2
!python evaluate_sft.py --adapter_dir /kaggle/working/sft_output/final_adapter
```

Default base model: `Qwen/Qwen2.5-0.5B-Instruct`. For a smaller/faster smoke test use `--max_train_samples 80 --max_eval_samples 20 --epochs 1`. The script attempts 4-bit quantization when CUDA + bitsandbytes are available; otherwise it falls back to ordinary LoRA loading. Kaggle GPU memory and package compatibility vary. If 4-bit loading fails, try restarting the session and using the fallback noted in `sft_project/README.md`.

### Run RL

```bash
%cd /kaggle/working/sft_rl_kaggle_projects/rl_project
!pip install -q -r requirements.txt
!python train_ppo.py --total_timesteps 100000 --output_dir /kaggle/working/rl_output
!python evaluate_policy.py --model_path /kaggle/working/rl_output/ppo_inventory.pt --episodes 100
```

The RL project is self-contained PyTorch PPO and does not require Gymnasium or Stable-Baselines3.

## Important implementation notes

- **SFT** means supervised fine-tuning on input/target examples. This project uses LoRA adapters; it is not RLHF.
- **RL** uses PPO with a custom environment, explicit state/action/reward design, and held-out evaluation episodes. It is not an LLM alignment project.
- Training metrics are produced by running the code. Do not claim a particular score or improvement until you have run it and inspected the outputs.
- The simulator is synthetic. Inventory policies should be validated against real demand, lead-time, cost, and service-level data before operational use.

## Repository structure

```text
sft_rl_kaggle_projects/
├── requirements.txt
├── sft_project/
│   ├── README.md
│   ├── data.py
│   ├── train_sft.py
│   └── evaluate_sft.py
└── rl_project/
    ├── README.md
    ├── inventory_env.py
    ├── ppo.py
    ├── train_ppo.py
    └── evaluate_policy.py
```


## Streamlit deployment

The repository now includes `streamlit_app.py`, `requirements_streamlit.txt`, and `DEPLOY_STREAMLIT.md`. Follow `DEPLOY_STREAMLIT.md` to merge the SFT adapter on Kaggle, host the merged model on Hugging Face, and deploy the PPO dashboard on Streamlit Community Cloud.
