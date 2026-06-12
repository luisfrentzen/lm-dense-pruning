from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import yaml


@dataclass
class EvalConfig:
    # paths
    root_dir: Path
    output_json: str
    output_csv: str
    offload_folder: str

    # eval settings
    tasks: List[str]
    device: str
    dtype_name: str
    dtype: torch.dtype
    batch_size: str | int
    max_batch_size: int

    # multi-gpu
    use_multi_gpu: bool
    gpus: Optional[int]
    max_memory_per_gpu: Optional[str]
    max_cpu_memory: Optional[str]

    # quantization
    quantization: str
    int8_threshold: float
    int8_skip_modules: List[str]

    @classmethod
    def from_yaml(cls, config_path: str = "eval_config.yaml") -> "ExperimentConfig":
        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f) or {}

        dtype_name = str(cfg.get("dtype", "bfloat16")).lower()
        dtype = cls.parse_dtype(dtype_name)

        quantization = str(cfg.get("quantization", "none")).lower()
        cls.validate_quantization_config(quantization)

        int8_skip_modules = cfg.get("int8_skip_modules", ["lm_head"]) or ["lm_head"]

        return cls(
            root_dir=Path(cfg["root_dir"]),
            tasks=cfg.get("tasks", ["mmlu", "gsm8k"]),
            device=cfg.get("device", "cuda:0"),
            dtype_name=dtype_name,
            dtype=dtype,
            batch_size=cfg.get("batch_size", "auto"),
            max_batch_size=int(cfg.get("max_batch_size", 32)),
            output_json=cfg.get("output_json", "./eval_out/all_results.json"),
            output_csv=cfg.get("output_csv", "./eval_out/all_results.csv"),
            use_multi_gpu=bool(cfg.get("use_multi_gpu", False)),
            gpus=cfg.get("gpus", None),
            max_memory_per_gpu=cfg.get("max_memory_per_gpu", None),
            max_cpu_memory=cfg.get("max_cpu_memory", None),
            offload_folder=cfg.get("offload_folder", "./offload"),
            quantization=quantization,
            int8_threshold=float(cfg.get("int8_threshold", 6.0)),
            int8_skip_modules=list(int8_skip_modules),
        )

    @staticmethod
    def parse_dtype(dtype_value: str) -> torch.dtype:
        mapping = {
            "float32": torch.float32,
            "fp32": torch.float32,
            "float16": torch.float16,
            "fp16": torch.float16,
            "bfloat16": torch.bfloat16,
            "bf16": torch.bfloat16,
        }
        key = str(dtype_value).lower()
        if key not in mapping:
            raise ValueError(f"Unsupported dtype: {dtype_value}")
        return mapping[key]

    @staticmethod
    def validate_quantization_config(quantization: str) -> None:
        allowed = {"none", "8bit"}
        if quantization not in allowed:
            raise ValueError(
                f"Unsupported quantization={quantization}. "
                f"Use one of: {sorted(allowed)}"
            )

    @property
    def is_int8(self) -> bool:
        return self.quantization == "8bit"

    def effective_gpu_count(self) -> int:
        if not torch.cuda.is_available():
            return 0
        if self.gpus is None:
            return torch.cuda.device_count()
        return int(self.gpus)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "root_dir": str(self.root_dir),
            "tasks": self.tasks,
            "device": self.device,
            "dtype_name": self.dtype_name,
            "dtype": str(self.dtype),
            "batch_size": self.batch_size,
            "max_batch_size": self.max_batch_size,
            "output_json": self.output_json,
            "output_csv": self.output_csv,
            "use_multi_gpu": self.use_multi_gpu,
            "gpus": self.gpus,
            "max_memory_per_gpu": self.max_memory_per_gpu,
            "max_cpu_memory": self.max_cpu_memory,
            "offload_folder": self.offload_folder,
            "quantization": self.quantization,
            "int8_threshold": self.int8_threshold,
            "int8_skip_modules": self.int8_skip_modules,
        }