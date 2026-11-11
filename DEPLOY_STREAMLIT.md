# Deploy the trained SFT + RL projects with Streamlit

## Recommended deployment architecture

- Train both projects on Kaggle GPU.
- Merge the trained LoRA adapter into the Qwen base model on Kaggle.
- Upload the merged model to a Hugging Face model repository.
- Store the small PPO checkpoint (`ppo_inventory.pt`) in this GitHub repository.
- Deploy `streamlit_app.py` on Streamlit Community Cloud.

This avoids loading a multi-hundred-million-parameter language model into Streamlit Community Cloud's limited CPU memory. Streamlit calls the hosted Hugging Face model through its inference API; the PPO checkpoint runs locally in the app.

## A. Merge SFT adapter on Kaggle

After `train_sft.py` finishes, run:

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base_id = "Qwen/Qwen2.5-0.5B-Instruct"
adapter_dir = "/kaggle/working/sft_output/final_adapter"
merged_dir = "/kaggle/working/qwen-sft-merged"

base = AutoModelForCausalLM.from_pretrained(
    base_id,
    torch_dtype=torch.float16,
    device_map="auto"
)
model = PeftModel.from_pretrained(base, adapter_dir)
merged = model.merge_and_unload()
merged.save_pretrained(merged_dir, safe_serialization=True)
tokenizer = AutoTokenizer.from_pretrained(adapter_dir)
tokenizer.save_pretrained(merged_dir)
print("Merged model saved:", merged_dir)
```

If you trained without CUDA 4-bit loading, the merge process may need more GPU memory. Keep the base model and adapter IDs identical to training.

## B. Upload merged model to Hugging Face Hub

Create a model repository on Hugging Face, then in Kaggle:

```python
!pip install -q huggingface_hub
from huggingface_hub import notebook_login, HfApi

notebook_login()
api = HfApi()
api.create_repo(repo_id="YOUR_USERNAME/customer-support-qwen-sft", repo_type="model", exist_ok=True)
api.upload_folder(
    folder_path="/kaggle/working/qwen-sft-merged",
    repo_id="YOUR_USERNAME/customer-support-qwen-sft",
    repo_type="model"
)
```

Use a private repository if you do not want the model files to be public. A private model repository may require a token with read permission in the Streamlit app.

## C. Save PPO checkpoint

After training:

```python
from pathlib import Path
import shutil

shutil.copy(
    "/kaggle/working/rl_output/ppo_inventory.pt",
    "/kaggle/working/ppo_inventory.pt"
)
```

Download `ppo_inventory.pt` from Kaggle output and add it to your GitHub repository at `rl_project/ppo_inventory.pt`. It is a small PyTorch checkpoint. Do not commit credentials or access tokens.

## D. GitHub repository layout

```text
your-repo/
├── streamlit_app.py
├── requirements_streamlit.txt
├── DEPLOY_STREAMLIT.md
├── ppo.py
├── inventory_env.py
└── rl_project/
    └── ppo_inventory.pt
```

Copy `ppo.py` and `inventory_env.py` from this project's `rl_project/` folder into the repository root, as the app imports them from there.

## E. Deploy on Streamlit Community Cloud

1. Push the files above to a GitHub repository.
2. Open https://share.streamlit.io/ and choose **Create app**.
3. Select the repository, branch, and `streamlit_app.py` as the main file.
4. In Advanced settings, set the Python version to a version compatible with your dependencies (Python 3.11 is a sensible choice).
5. Set the dependency file to `requirements_streamlit.txt` if the deployment UI supports selecting it. Otherwise name it `requirements.txt` in the deployment repository.
6. Deploy the app.

## F. Configure Streamlit Secrets

In the app's **Settings → Secrets**, add:

```toml
HF_MODEL_ID = "YOUR_USERNAME/customer-support-qwen-sft"
HF_TOKEN = "hf_your_read_token"
RL_MODEL_PATH = "rl_project/ppo_inventory.pt"
```

Use a Hugging Face token with read access to the model repository. Never put the token in source code or GitHub.

The Hugging Face model must be available to the Inference API/provider for `InferenceClient.chat_completion` to work. If the model is not served by an available provider, configure a dedicated Hugging Face Inference Endpoint and use its endpoint URL as the inference client model/endpoint target; availability and pricing depend on your Hugging Face account and model.

## Troubleshooting

- **Hugging Face 401/403:** Check `HF_TOKEN`, repository access, and that the token has read permission.
- **Model not supported by an inference provider:** Create a dedicated inference endpoint or choose a model/provider supported by Hugging Face Inference.
- **Streamlit cannot import `ppo`:** Ensure `ppo.py` and `inventory_env.py` are in the same directory as `streamlit_app.py`.
- **Missing checkpoint:** Ensure `rl_project/ppo_inventory.pt` exists in GitHub and the `RL_MODEL_PATH` secret points to it.
- **App memory issue:** The app intentionally does not load Qwen weights locally. Keep LLM inference hosted remotely.
- **PyTorch checkpoint safety:** Only load the checkpoint you trained yourself or otherwise trust.
