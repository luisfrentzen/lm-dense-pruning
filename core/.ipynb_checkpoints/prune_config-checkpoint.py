from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
import yaml


@dataclass
class PruneConfig:
    base_model: str
    model_type: str
    save_dir: str
    save_model: bool
    dataset: str

    seed: int

    pruning_ratio: float
    pruner_type: str

    channel_wise: bool
    block_wise: bool
    layer_wise: bool

    layer: int
    block_attention_layer_start: int
    block_attention_layer_end: int
    block_mlp_layer_start: int
    block_mlp_layer_end: int

    iterative_steps: int
    grouping_strategy: str
    global_pruning: bool

    taylor: str
    num_examples: int

    temperature: float
    top_p: float
    max_seq_len: int

    test_before_train: bool
    test_after_train: bool

    device: str
    eval_device: str
    torch_version: float

    @classmethod
    def from_yaml(cls, config_path="config/prune_config.yaml"):
        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f) or {}

        # device = "cuda" if torch.cuda.is_available() else "cpu"
        # eval_device = "cuda" if torch.cuda.is_available() else "cpu"
        device = "cpu"
        eval_device = "cpu"
        torch_version = float(".".join(torch.__version__.split(".")[:2]))

        return cls(
            base_model=cfg["base_model"],
            model_type=str(cfg.get("model_type", "gemma3")).lower(),
            save_dir=cfg.get("save_dir", "./pruned_models"),
            save_model=bool(cfg.get("save_model", True)),
            dataset=cfg["dataset"],
            seed=int(cfg.get("seed", 42)),
            pruning_ratio=float(cfg.get("pruning_ratio", 0.2)),
            pruner_type=str(cfg.get("pruner_type", "l2")).lower(),
            channel_wise=bool(cfg.get("channel_wise", False)),
            block_wise=bool(cfg.get("block_wise", False)),
            layer_wise=bool(cfg.get("layer_wise", False)),
            layer=int(cfg.get("layer", 24)),
            block_attention_layer_start=int(cfg.get("block_attention_layer_start", 0)),
            block_attention_layer_end=int(cfg.get("block_attention_layer_end", 0)),
            block_mlp_layer_start=int(cfg.get("block_mlp_layer_start", 0)),
            block_mlp_layer_end=int(cfg.get("block_mlp_layer_end", 0)),
            iterative_steps=int(cfg.get("iterative_steps", 10)),
            grouping_strategy=str(cfg.get("grouping_strategy", "sum")),
            global_pruning=bool(cfg.get("global_pruning", False)),
            taylor=str(cfg.get("taylor", "param_first")),
            num_examples=int(cfg.get("num_examples", 10)),
            temperature=float(cfg.get("temperature", 1.0)),
            top_p=float(cfg.get("top_p", 0.95)),
            max_seq_len=int(cfg.get("max_seq_len", 128)),
            test_before_train=bool(cfg.get("test_before_train", False)),
            test_after_train=bool(cfg.get("test_after_train", False)),
            device=device,
            eval_device=eval_device,
            torch_version=torch_version,
        )