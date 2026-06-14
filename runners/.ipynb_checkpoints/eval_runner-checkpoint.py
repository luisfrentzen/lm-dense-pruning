from __future__ import annotations

import gc
import json
import re
from pathlib import Path

import lm_eval
import numpy as np
import pandas as pd
import torch
from lm_eval.models.huggingface import HFLM

from core.eval_config import EvalConfig
from core.checkpoint_manager import CheckpointManager
from core.model_loader import ModelLoader
from core.utils import cleanup_memory, gb, show_step


class EvalRunner:
    def __init__(self, config_path: str = "eval_config.yaml"):
        self.cfg = EvalConfig.from_yaml(config_path)
        self.ckpt = CheckpointManager()
        self.loader = ModelLoader(self.cfg, self.ckpt)

    def print_cuda_mem(self, tag=""):
        if torch.cuda.is_available():
            print(
                f"{tag} allocated={torch.cuda.memory_allocated()/1024**3:.2f} GB | "
                f"reserved={torch.cuda.memory_reserved()/1024**3:.2f} GB | "
                f"peak_alloc={torch.cuda.max_memory_allocated()/1024**3:.2f} GB"
            )

    def get_gpu_mem_stats(self):
        if not torch.cuda.is_available():
            return {
                "gpu_alloc_gb": None,
                "gpu_reserved_gb": None,
                "gpu_peak_alloc_gb": None,
                "gpu_peak_reserved_gb": None,
            }

        return {
            "gpu_alloc_gb": gb(torch.cuda.memory_allocated()),
            "gpu_reserved_gb": gb(torch.cuda.memory_reserved()),
            "gpu_peak_alloc_gb": gb(torch.cuda.max_memory_allocated()),
            "gpu_peak_reserved_gb": gb(torch.cuda.max_memory_reserved()),
        }

    def get_model_size_stats(self, model):
        param_count = sum(p.numel() for p in model.parameters())
        param_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
        buffer_bytes = sum(b.numel() * b.element_size() for b in model.buffers())
        raw_total_bytes = param_bytes + buffer_bytes

        memory_footprint_bytes = None
        if hasattr(model, "get_memory_footprint"):
            try:
                memory_footprint_bytes = int(model.get_memory_footprint())
            except Exception:
                memory_footprint_bytes = None

        total_bytes = memory_footprint_bytes if memory_footprint_bytes is not None else raw_total_bytes

        return {
            "param_count": param_count,
            "param_count_m": round(param_count / 1e6, 3),
            "model_size_gb": gb(total_bytes),
            "raw_tensor_size_gb": gb(raw_total_bytes),
            "memory_footprint_gb": gb(memory_footprint_bytes) if memory_footprint_bytes is not None else None,
        }

    def parse_run_name(self, name: str):
        row = {
            "folder": name,
            "pruning_ratio": None,
            "pruner_type": None,
            "block_wise": False,
            "block_attn_range": None,
            "block_mlp_range": None,
            "channel_wise": False,
            "layer_wise": False,
            "drop_after_layer": None,
            "iterative_steps": None,
            "grouping_strategy": None,
            "global_pruning": None,
            "taylor": None,
            "num_examples": None,
        }

        parts = name.split("_")

        for p in parts:
            if re.fullmatch(r"r.+", p):
                row["pruning_ratio"] = p[1:]
            elif re.fullmatch(r"p.+", p):
                row["pruner_type"] = p[1:]
            elif p == "blk":
                row["block_wise"] = True
            elif re.fullmatch(r"a\d+-\d+", p):
                row["block_attn_range"] = p[1:]
            elif re.fullmatch(r"m\d+-\d+", p):
                row["block_mlp_range"] = p[1:]
            elif p == "ch":
                row["channel_wise"] = True
            elif p == "ly":
                row["layer_wise"] = True
            elif re.fullmatch(r"d\d+", p):
                row["drop_after_layer"] = int(p[1:])
            elif re.fullmatch(r"it\d+", p):
                row["iterative_steps"] = int(p[2:])
            elif re.fullmatch(r"g.+", p):
                row["grouping_strategy"] = p[1:]
            elif p == "glob":
                row["global_pruning"] = True
            elif p == "noglob":
                row["global_pruning"] = False
            elif re.fullmatch(r"t.+", p):
                row["taylor"] = p[1:]
            elif re.fullmatch(r"n\d+", p):
                row["num_examples"] = int(p[1:])

        return row

    def flatten_results(self, results_dict):
        flat = {}
        for task_name, metrics in results_dict.items():
            if isinstance(metrics, dict):
                for k, v in metrics.items():
                    if isinstance(v, (int, float, bool)):
                        flat[f"{task_name}.{k}"] = v
        return flat

    def to_jsonable(self, obj):
        if isinstance(obj, dict):
            return {str(k): self.to_jsonable(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [self.to_jsonable(v) for v in obj]
        if isinstance(obj, tuple):
            return [self.to_jsonable(v) for v in obj]
        if isinstance(obj, np.generic):
            return obj.item()
        if isinstance(obj, torch.Tensor):
            return obj.tolist()
        if isinstance(obj, torch.dtype):
            return str(obj)
        if callable(obj):
            return str(obj)
        return obj if isinstance(obj, (str, int, float, bool)) or obj is None else str(obj)

    def build_eval_model(self, loaded):
        return HFLM(
            pretrained=loaded["model"],
            tokenizer=loaded["tokenizer"],
            batch_size=self.cfg.batch_size,
            max_batch_size=self.cfg.max_batch_size,
        )

    def evaluate_one_model(self, model_dir: Path):
        show_step(f"Evaluating: {model_dir.name}")

        cleanup_memory()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        self.print_cuda_mem("before load")

        loaded = self.loader.load_model(model_dir)
        model = loaded["model"]
        tokenizer = loaded["tokenizer"]
        model_type = loaded["model_type"]

        self.print_cuda_mem("after load")

        size_stats = self.get_model_size_stats(model)
        mem_after_load = self.get_gpu_mem_stats()

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        lm = self.build_eval_model(loaded)

        results = lm_eval.simple_evaluate(
            model=lm,
            tasks=self.cfg.tasks,
        )

        self.print_cuda_mem("after eval")
        mem_after_eval = self.get_gpu_mem_stats()

        row = self.parse_run_name(model_dir.name)
        row["model_type"] = model_type
        row["quantization"] = self.cfg.quantization
        row["int8_threshold"] = self.cfg.int8_threshold if self.cfg.quantization == "8bit" else None
        row["int8_skip_modules"] = ",".join(self.cfg.int8_skip_modules) if self.cfg.quantization == "8bit" else None

        row.update(size_stats)
        row["gpu_alloc_after_load_gb"] = mem_after_load["gpu_alloc_gb"]
        row["gpu_reserved_after_load_gb"] = mem_after_load["gpu_reserved_gb"]
        row["gpu_peak_eval_alloc_gb"] = mem_after_eval["gpu_peak_alloc_gb"]
        row["gpu_peak_eval_reserved_gb"] = mem_after_eval["gpu_peak_reserved_gb"]

        row.update(self.flatten_results(results["results"]))

        del lm
        del model
        del tokenizer
        cleanup_memory()

        return row, results

    def show_saved_csv(self, csv_path=None, wanted_cols=None):
        csv_path = csv_path or self.cfg.output_csv
        df = pd.read_csv(csv_path)

        if wanted_cols is None:
            wanted_cols = [
                "folder",
                "model_type",
                "quantization",
                "pruning_ratio",
                "pruner_type",
                "param_count",
                "model_size_gb",
                "memory_footprint_gb",
                "gpu_reserved_after_load_gb",
                "gsm8k.exact_match,strict-match",
                "mmlu.acc,none",
            ]

        existing_cols = [c for c in wanted_cols if c in df.columns]
        return df[existing_cols]

    def run(self):
        if not self.cfg.root_dir.exists():
            raise FileNotFoundError(f"Root directory not found: {self.cfg.root_dir}")

        model_dirs = [p for p in self.cfg.root_dir.iterdir() if p.is_dir()]

        all_rows = []
        all_raw_results = {}
        failed = {}

        for model_dir in sorted(model_dirs):
            try:
                row, raw = self.evaluate_one_model(model_dir)
                all_rows.append(row)
                all_raw_results[model_dir.name] = raw
            except Exception as e:
                failed[model_dir.name] = str(e)

        df = pd.DataFrame(all_rows)

        if not df.empty and "folder" in df.columns:
            df = df.sort_values(["folder"]).reset_index(drop=True)

        Path(self.cfg.output_json).parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "tasks": self.cfg.tasks,
            "quantization": self.cfg.quantization,
            "results": self.to_jsonable(all_raw_results),
            "failed": failed,
        }

        with open(self.cfg.output_json, "w") as f:
            json.dump(payload, f, indent=2)

        df.to_csv(self.cfg.output_csv, index=False)

        return {
            "num_success": len(all_rows),
            "num_failed": len(failed),
            "output_json": self.cfg.output_json,
            "output_csv": self.cfg.output_csv,
            "failed": failed,
        }