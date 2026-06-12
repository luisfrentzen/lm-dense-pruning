import gc
import random

import numpy as np
import torch


def set_random_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_torch_dtype(device: str):
    if device == "cpu":
        return torch.float32
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def move_batch_to_device(batch, device):
    if hasattr(batch, "to"):
        return batch.to(device)
    if isinstance(batch, dict):
        return {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
    if torch.is_tensor(batch):
        return batch.to(device)
    return batch


def cleanup_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def gb(x):
    if x is None:
        return None
    return round(x / 1024**3, 4)


def show_step(msg: str):
    print(f"\n{'=' * 20} {msg} {'=' * 20}\n")


def count_trainable_parameters(module):
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def log_cuda_memory(prefix=""):
    if torch.cuda.is_available():
        mem = torch.cuda.memory_allocated() / 1024 / 1024
        text = f"{prefix}Memory Requirement: {mem:.2f} MiB"
        print(text)
        return text
    return None