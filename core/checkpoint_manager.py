from __future__ import annotations

import json
from pathlib import Path
from typing import Optional


class CheckpointManager:
    def __init__(self) -> None:
        pass

    def is_pretrained_dir(self, model_dir: Path) -> bool:
        has_config = (model_dir / "config.json").exists()
        has_weights = (
            (model_dir / "model.safetensors").exists()
            or (model_dir / "model.safetensors.index.json").exists()
            or (model_dir / "pytorch_model.bin").exists()
            or (model_dir / "pytorch_model.bin.index.json").exists()
        )
        return has_config and has_weights

    def find_prune_config(self, model_dir: Path) -> Optional[Path]:
        candidates = [
            model_dir / "prune_config.json",
        ]
        for path in candidates:
            if path.exists():
                return path
        return None

    def find_custom_weight_file(self, model_dir: Path) -> Optional[Path]:
        candidates = [
            model_dir / "pytorch_model.bin",
            model_dir / "model.safetensors",
        ]
        for path in candidates:
            if path.exists():
                return path
        return None

    def is_custom_pruned_dir(self, model_dir: Path) -> bool:
        return (
            self.find_prune_config(model_dir) is not None
            and self.find_custom_weight_file(model_dir) is not None
        )

    def get_checkpoint_type(self, model_dir: Path) -> str:
        if self.is_custom_pruned_dir(model_dir):
            return "custom"
        if self.is_pretrained_dir(model_dir):
            return "pretrained"
        return "unknown"

    def read_json(self, path: Path) -> dict:
        with open(path, "r") as f:
            return json.load(f)

    def read_model_config(self, model_dir: Path) -> dict:
        config_path = model_dir / "config.json"
        if not config_path.exists():
            return {}
        return self.read_json(config_path)

    def read_prune_config(self, model_dir: Path) -> dict:
        prune_config_path = self.find_prune_config(model_dir)
        if prune_config_path is None:
            return {}
        return self.read_json(prune_config_path)

    def detect_architecture(self, model_dir: Path) -> str:
        """
        Return one of:
        - gemma3
        - gemma4
        - llama
        - mistral
        - unknown
        """
        prune_cfg = self.read_prune_config(model_dir)
        model_cfg = self.read_model_config(model_dir)

        candidates = [
            str(prune_cfg.get("base_model_name", "")).lower(),
            str(model_cfg.get("_name_or_path", "")).lower(),
            str(model_cfg.get("model_type", "")).lower(),
            str(model_dir.name).lower(),
        ]

        joined = " ".join(candidates)

        if "gemma-3" in joined or "gemma3" in joined:
            return "gemma3"
        if "gemma-4" in joined or "gemma4" in joined:
            return "gemma4"
        if "mistral" in joined:
            return "mistral"
        if "llama" in joined:
            return "llama"

        return "unknown"