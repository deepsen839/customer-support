from datasets import load_dataset

TRAIN_FILE = r"/kaggle/input/datasets/deepsen839/customer-support-synthetic-qwen-dataset/customer_support_synthetic_qwen/train.jsonl"
VALID_FILE = r"/kaggle/input/datasets/deepsen839/customer-support-synthetic-qwen-dataset/customer_support_synthetic_qwen/validation.jsonl"

def build_datasets():
    train_ds = load_dataset("json", data_files=TRAIN_FILE, split="train")
    eval_ds = load_dataset("json", data_files=VALID_FILE, split="train")
    return train_ds, eval_ds
