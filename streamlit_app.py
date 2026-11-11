import os
import sys
from pathlib import Path

import pandas as pd
import streamlit as st
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parent
RL_PROJECT_DIR = ROOT / "rl_project"
if str(RL_PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(RL_PROJECT_DIR))

DEFAULT_HF_MODEL_ID = "Deepsen8391/qwen25-synthetic-customer-support"
HF_MODEL_ID = os.getenv("HF_MODEL_ID", DEFAULT_HF_MODEL_ID)


def get_setting(name: str, default: str = "") -> str:
    try:
        value = st.secrets.get(name, default)
    except Exception:
        value = os.getenv(name, default)
    return str(value) if value is not None else default


HF_TOKEN = get_setting("HF_TOKEN")
SFT_MODEL_PATH = get_setting("SFT_MODEL_PATH", HF_MODEL_ID)
RL_MODEL_PATH = Path(get_setting(
    "RL_MODEL_PATH", "rl_output_penalty8/ppo_inventory.pt"
))
if not RL_MODEL_PATH.is_absolute():
    RL_MODEL_PATH = ROOT / RL_MODEL_PATH

PPO_IMPORT_ERROR = None
try:
    from ppo import ActorCritic
    from inventory_env import InventoryEnv
except Exception as exc:
    ActorCritic = None
    InventoryEnv = None
    PPO_IMPORT_ERROR = exc

st.set_page_config(
    page_title="SFT + PPO AI Lab",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource(show_spinner=False)
def load_sft_model(model_path: str, token: str):
    """Load a merged/full Transformers model from Hugging Face."""
    auth_token = token or None
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, token=auth_token, trust_remote_code=False
    )

    if torch.cuda.is_available():
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            token=auth_token,
            dtype=torch.float16,
            device_map="auto",
            low_cpu_mem_usage=True,
            trust_remote_code=False,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            token=auth_token,
            dtype=torch.float32,
            low_cpu_mem_usage=True,
            trust_remote_code=False,
        )
        model.to("cpu")

    model.eval()
    return tokenizer, model


def generate_sft_response(user_message: str) -> str:
    tokenizer, model = load_sft_model(SFT_MODEL_PATH, HF_TOKEN)

    system_prompt = (
        "You are a careful, polite, concise customer-support assistant. "
        "Never invent company policies, prices, delivery dates, refund "
        "eligibility, account details, contact information, or transaction "
        "status. If details are missing, ask for them. Never request "
        "passwords, OTPs, CVVs, or full payment-card numbers. For "
        "account-specific actions, direct the customer to the official "
        "authenticated app or support channel."
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message.strip()},
    ]

    # Explicit return_dict=True prevents BatchEncoding from being mistaken
    # for a tensor by model.generate().
    if getattr(tokenizer, "chat_template", None):
        encoded = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        )
        input_ids = encoded["input_ids"]
        attention_mask = encoded.get("attention_mask")
    else:
        prompt = (
            "<|im_start|>system\n" + system_prompt
            + "<|im_end|>\n<|im_start|>user\n" + user_message.strip()
            + "<|im_end|>\n<|im_start|>assistant\n"
        )
        encoded = tokenizer(prompt, return_tensors="pt")
        input_ids = encoded["input_ids"]
        attention_mask = encoded.get("attention_mask")

    input_device = model.get_input_embeddings().weight.device
    input_ids = input_ids.to(input_device)
    if attention_mask is not None:
        attention_mask = attention_mask.to(input_device)

    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=180,
            do_sample=False,
            pad_token_id=pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    new_tokens = output_ids[0, input_ids.shape[-1]:]
    answer = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    return answer or "The model returned an empty response."


@st.cache_resource(show_spinner=False)
def load_ppo(checkpoint_path: str):
    if ActorCritic is None:
        raise RuntimeError(f"Could not import PPO modules: {PPO_IMPORT_ERROR}")

    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False
    )
    if not isinstance(checkpoint, dict):
        raise TypeError("PPO checkpoint must contain a dictionary.")

    required = {"obs_dim", "action_dim", "model_state_dict"}
    missing = required.difference(checkpoint.keys())
    if missing:
        raise KeyError(
            "PPO checkpoint is missing required keys: "
            + ", ".join(sorted(missing))
        )

    policy = ActorCritic(checkpoint["obs_dim"], checkpoint["action_dim"])
    policy.load_state_dict(checkpoint["model_state_dict"])
    policy.eval()
    return policy


st.title("🤖 SFT + Reinforcement Learning AI Lab")
st.caption(
    "Customer-support assistant fine-tuned with LoRA · "
    "Inventory replenishment policy trained with PPO"
)

with st.sidebar:
    st.header("Project status")
    st.write("**SFT model:** Qwen 2.5 0.5B Instruct")
    st.write("**Hugging Face repository:**")
    st.link_button(
        "Open model repository",
        f"https://huggingface.co/{HF_MODEL_ID}",
        use_container_width=True,
    )
    st.write(
        "**PPO checkpoint:** "
        + ("Available" if RL_MODEL_PATH.is_file() else "Not found")
    )
    st.write(
        "**Compute:** "
        + ("CUDA GPU" if torch.cuda.is_available() else "CPU")
    )
    st.caption(
        "SFT weights are loaded from Hugging Face using Transformers. "
        "PPO uses the local checkpoint."
    )

tab_sft, tab_rl, tab_about = st.tabs(
    ["💬 SFT Customer Support", "📦 PPO Inventory Agent", "ℹ️ About"]
)

with tab_sft:
    st.subheader("Fine-tuned customer-support assistant")
    st.write(
        "Ask a general support question. The assistant should ask for "
        "missing information rather than inventing account details or policies."
    )
    question = st.text_area(
        "Your question",
        placeholder="How can I request a refund?",
        height=110,
        key="sft_question",
    )
    generate = st.button(
        "Generate answer", type="primary", key="sft_generate"
    )

    if generate:
        if not question.strip():
            st.warning("Please enter a question.")
        else:
            try:
                with st.spinner(
                    "Loading Qwen model (first load may take a few minutes) "
                    "and generating an answer..."
                ):
                    answer = generate_sft_response(question)
                st.markdown("### Assistant response")
                st.write(answer)
            except Exception as exc:
                st.error("SFT model inference failed.")
                st.exception(exc)
                st.info(
                    "Confirm that the repository contains a merged/full "
                    "Transformers model with config.json, tokenizer files "
                    "and model weights. If it contains only "
                    "adapter_model.safetensors, it is a LoRA adapter and "
                    "must be loaded together with its base model."
                )

    st.caption(
        "This app uses Transformers directly, not the Hugging Face "
        "Inference Providers Chat Completion API."
    )

with tab_rl:
    st.subheader("PPO inventory replenishment simulator")
    st.write(
        "Run the trained Actor-Critic policy in the project's stochastic "
        "inventory environment."
    )
    seed = st.number_input(
        "Demand seed", min_value=0, max_value=100000,
        value=42, step=1, key="ppo_seed"
    )
    run_simulation = st.button(
        "Run trained PPO policy", type="primary", key="ppo_run"
    )

    if PPO_IMPORT_ERROR is not None:
        st.error("PPO modules could not be imported.")
        st.exception(PPO_IMPORT_ERROR)

    if run_simulation:
        if not RL_MODEL_PATH.is_file():
            st.error(
                f"PPO checkpoint not found: {RL_MODEL_PATH}. "
                "Place ppo_inventory.pt at the configured path."
            )
        elif ActorCritic is None or InventoryEnv is None:
            st.error("PPO cannot run until the RL imports are fixed.")
        else:
            try:
                with st.spinner("Running inventory simulation..."):
                    policy = load_ppo(str(RL_MODEL_PATH))
                    env = InventoryEnv(seed=int(seed))
                    obs = env.reset(seed=int(seed))
                    done = False
                    daily_rows = []
                    total_reward = 0.0

                    # Matches the supplied project's legacy four-value API.
                    while not done:
                        with torch.no_grad():
                            obs_tensor = torch.as_tensor(
                                obs, dtype=torch.float32
                            ).unsqueeze(0)
                            logits, _ = policy(obs_tensor)
                            action = int(
                                torch.argmax(logits, dim=-1).item()
                            )

                        obs, reward, done, info = env.step(action)
                        total_reward += float(reward)
                        daily_rows.append({
                            "Day": len(daily_rows) + 1,
                            "Demand": info["demand"],
                            "Order": info["order"],
                            "Lost sales": info["lost_sales"],
                            "Ending inventory": env.inventory,
                            "Daily cost": info["cost"],
                        })

                    if not daily_rows:
                        st.warning(
                            "The environment returned no simulation days."
                        )
                    else:
                        df = pd.DataFrame(daily_rows)
                        total_cost = float(df["Daily cost"].sum())
                        lost_sales = float(df["Lost sales"].sum())
                        mean_cost = float(df["Daily cost"].mean())

                        c1, c2, c3 = st.columns(3)
                        c1.metric(
                            "Total simulated cost", f"{total_cost:,.2f}"
                        )
                        c2.metric(
                            "Lost sales (units)", f"{lost_sales:,.0f}"
                        )
                        c3.metric(
                            "Mean daily cost", f"{mean_cost:,.2f}"
                        )

                        st.markdown("### Inventory performance")
                        st.line_chart(
                            df.set_index("Day")[
                                ["Demand", "Order", "Ending inventory"]
                            ]
                        )
                        st.markdown("### Daily simulation results")
                        st.dataframe(
                            df, use_container_width=True, hide_index=True
                        )
                        st.success(
                            "Simulation complete · cumulative reward: "
                            f"{total_reward:,.2f}"
                        )
                        st.download_button(
                            "Download simulation CSV",
                            data=df.to_csv(index=False).encode("utf-8"),
                            file_name="ppo_inventory_simulation.csv",
                            mime="text/csv",
                        )
            except Exception as exc:
                st.error(
                    "PPO simulation failed. Check the checkpoint format, "
                    "environment API, and RL project files."
                )
                st.exception(exc)

with tab_about:
    st.subheader("Project details")
    st.markdown(
        f"""
        ### SFT customer-support assistant
        - Base model: Qwen 2.5 0.5B Instruct
        - Fine-tuning: supervised fine-tuning with LoRA
        - Dataset: synthetic customer-support conversations
        - Serving: Transformers model loading from Hugging Face
        - Model repository: [{HF_MODEL_ID}](https://huggingface.co/{HF_MODEL_ID})

        ### Reinforcement learning inventory agent
        - Algorithm: Proximal Policy Optimization (PPO)
        - Framework: PyTorch
        - Policy: Actor-Critic
        - Environment: custom stochastic inventory simulator
        - Action: discrete replenishment quantity
        - Reward: negative holding, ordering, and lost-sales costs

        ### Technology stack
        Python · PyTorch · Transformers · PEFT · Hugging Face ·
        Streamlit · NumPy · Pandas
        """
    )
    st.warning(
        "Educational experimental project. The SFT model requires further "
        "quality evaluation. Previous PPO evaluation did not outperform the "
        "baseline on cost and lost sales. Do not use these components for "
        "real customer or inventory decisions without additional validation."
    )

st.divider()
st.caption("SFT + PPO AI Lab · Experimental project")
