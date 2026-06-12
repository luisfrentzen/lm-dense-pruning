from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn as nn
from safetensors.torch import load_file
from transformers import AutoTokenizer, Gemma3ForConditionalGeneration

from adapters.base_adapter import BaseModelAdapter


class Gemma3Adapter(BaseModelAdapter):
    name = "gemma3"

    def matches(self, model_dir: Path, checkpoint_manager) -> bool:
        return checkpoint_manager.detect_architecture(model_dir) == "gemma3"

    def load_pretrained_model(self, model_dir: Path, cfg, checkpoint_manager, load_kwargs=None):
        load_kwargs = load_kwargs or {}
        model_dir = model_dir.expanduser().resolve()
        return Gemma3ForConditionalGeneration.from_pretrained(
            str(model_dir),
            **load_kwargs,
        )

    def load_tokenizer(self, model_dir: Path):
        return AutoTokenizer.from_pretrained(str(model_dir), trust_remote_code=True)

    def get_text_layers(self, model):
        return model.model.language_model.layers

    def rebuild_custom_model(self, model_dir: Path, cfg, checkpoint_manager):
        prune_config_path = checkpoint_manager.find_prune_config(model_dir)
        if prune_config_path is None:
            raise FileNotFoundError(f"prune_config.json not found in {model_dir}")
    
        with open(prune_config_path, "r") as f:
            meta = json.load(f)
    
        # Load on CPU — surgery + state_dict load must happen before dispatch
        model = Gemma3ForConditionalGeneration.from_pretrained(
            meta["base_model_name"],
            torch_dtype=cfg.dtype,
            trust_remote_code=True,
            local_files_only=True,
            low_cpu_mem_usage=True,
        )
    
        self._resize_pruned_layers(model, meta)
    
        weight_path = checkpoint_manager.find_custom_weight_file(model_dir)
        if weight_path is None:
            raise FileNotFoundError(f"No weight file found in {model_dir}")
    
        if weight_path.name == "pytorch_model.bin":
            self._load_bin_checkpoint(model, weight_path)
        elif weight_path.name == "model.safetensors":
            self._load_safetensors_checkpoint(model, weight_path)
        else:
            raise ValueError(f"Unsupported weight file: {weight_path}")
    
        model.eval()
        model.tie_weights()  # safe no-op if already tied; needed before dispatch
    
        # Now dispatch across GPUs with proper hooks
        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            from accelerate import dispatch_model, infer_auto_device_map
            from accelerate.utils import get_balanced_memory
    
            no_split = getattr(model, "_no_split_modules", None) or []
            max_memory = get_balanced_memory(
                model, dtype=cfg.dtype, low_zero=False,
                no_split_module_classes=no_split,
            )
            device_map = infer_auto_device_map(
                model,
                max_memory=max_memory,
                dtype=cfg.dtype,
                no_split_module_classes=no_split,
            )
            model = dispatch_model(model, device_map=device_map)
    
        return model

    def _resize_pruned_layers(self, model, meta: dict):
        layers = self.get_text_layers(model)
        layer_shapes = meta["layer_shapes"]

        if len(layers) != len(layer_shapes):
            raise ValueError(
                f"Layer count mismatch: model has {len(layers)} layers, "
                f"metadata has {len(layer_shapes)}"
            )

        cpu = torch.device("cpu")

        for layer, shape in zip(layers, layer_shapes):
            attn = layer.self_attn
            mlp = layer.mlp
            old_dtype = attn.q_proj.weight.dtype

            if attn.q_proj.weight.shape != torch.Size([shape["q_out"], shape["q_in"]]):
                attn.q_proj = nn.Linear(
                    shape["q_in"], shape["q_out"], bias=False, device=cpu, dtype=old_dtype
                )

            if attn.k_proj.weight.shape != torch.Size([shape["k_out"], shape["k_in"]]):
                attn.k_proj = nn.Linear(
                    shape["k_in"], shape["k_out"], bias=False, device=cpu, dtype=old_dtype
                )

            if attn.v_proj.weight.shape != torch.Size([shape["v_out"], shape["v_in"]]):
                attn.v_proj = nn.Linear(
                    shape["v_in"], shape["v_out"], bias=False, device=cpu, dtype=old_dtype
                )

            if attn.o_proj.weight.shape != torch.Size([shape["o_out"], shape["o_in"]]):
                attn.o_proj = nn.Linear(
                    shape["o_in"], shape["o_out"], bias=False, device=cpu, dtype=old_dtype
                )

            if mlp.gate_proj.weight.shape != torch.Size([shape["gate_out"], shape["gate_in"]]):
                mlp.gate_proj = nn.Linear(
                    shape["gate_in"], shape["gate_out"], bias=False, device=cpu, dtype=old_dtype
                )

            if mlp.up_proj.weight.shape != torch.Size([shape["up_out"], shape["up_in"]]):
                mlp.up_proj = nn.Linear(
                    shape["up_in"], shape["up_out"], bias=False, device=cpu, dtype=old_dtype
                )

            if mlp.down_proj.weight.shape != torch.Size([shape["down_out"], shape["down_in"]]):
                mlp.down_proj = nn.Linear(
                    shape["down_in"], shape["down_out"], bias=False, device=cpu, dtype=old_dtype
                )

    def _load_bin_checkpoint(self, model, weight_path: Path):
        state_dict = torch.load(weight_path, map_location="cpu")
        missing, unexpected = model.load_state_dict(state_dict, strict=False)

        if missing or unexpected:
            print("Missing:", missing[:20])
            print("Unexpected:", unexpected[:20])
            raise RuntimeError("Checkpoint did not reload cleanly")

    def _load_safetensors_checkpoint(self, model, weight_path: Path):
        state_dict = load_file(weight_path, device="cpu")

        target_state = model.state_dict()
        target_keys = set(target_state.keys())

        remapped_state_dict = {}
        unmapped = []

        for key, value in state_dict.items():
            matched = False
            for candidate in self._candidate_keys(key):
                if candidate in target_keys and target_state[candidate].shape == value.shape:
                    remapped_state_dict[candidate] = value
                    matched = True
                    break
            if not matched:
                unmapped.append(key)

        missing, unexpected = model.load_state_dict(remapped_state_dict, strict=False)

        allowed_missing_prefixes = (
            "model.vision_tower.",
            "vision_tower.",
            "model.multi_modal_projector.",
            "multi_modal_projector.",
            "model.image_newline",
            "image_newline",
        )

        filtered_missing = [
            key for key in missing
            if not key.startswith(allowed_missing_prefixes)
        ]

        if filtered_missing == ["lm_head.weight"] and not unexpected:
            self._repair_lm_head(model)
            return

        if filtered_missing or unexpected:
            print("Unmapped checkpoint keys:", unmapped[:20])
            print("Missing:", filtered_missing[:20])
            print("Unexpected:", unexpected[:20])
            raise RuntimeError("Checkpoint did not reload cleanly")

    def _repair_lm_head(self, model):
        input_embed = model.get_input_embeddings().weight

        if hasattr(model, "lm_head") and model.lm_head.weight.shape == input_embed.shape:
            model.lm_head.weight.data.copy_(input_embed.data)
        else:
            model.tie_weights()

    def _candidate_keys(self, src_key: str):
        candidates = [src_key]

        if src_key.startswith("language_model.model."):
            suffix = src_key[len("language_model.model."):]
            candidates += [
                "model.language_model." + suffix,
                "model.language_model.model." + suffix,
            ]

        if src_key.startswith("language_model."):
            suffix = src_key[len("language_model."):]
            candidates += [
                "model.language_model." + suffix,
                "model.language_model.model." + suffix,
            ]

        if src_key.startswith("vision_tower."):
            suffix = src_key[len("vision_tower."):]
            candidates += ["model.vision_tower." + suffix]

        if src_key.startswith("multi_modal_projector."):
            suffix = src_key[len("multi_modal_projector."):]
            candidates += ["model.multi_modal_projector." + suffix]

        if src_key.startswith("language_model.lm_head."):
            suffix = src_key[len("language_model.lm_head."):]
            candidates += ["lm_head." + suffix]

        if src_key == "image_newline":
            candidates += ["model.image_newline"]

        seen = set()
        unique_candidates = []
        for key in candidates:
            if key not in seen:
                unique_candidates.append(key)
                seen.add(key)

        return unique_candidates
    
    def load_base_model_and_tokenizer(self, base_model_path: str, torch_dtype, device_map=None):
        print("DEBUG GEMMA")
        tokenizer = AutoTokenizer.from_pretrained(base_model_path, local_files_only=True)
        model = Gemma3ForConditionalGeneration.from_pretrained(
            base_model_path,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
            local_files_only=True,
            device_map=device_map,
        )
        return model, tokenizer

    def set_special_tokens(self, model, tokenizer):
        if tokenizer.pad_token_id is None:
            if tokenizer.eos_token is not None:
                tokenizer.pad_token = tokenizer.eos_token
            elif tokenizer.unk_token is not None:
                tokenizer.pad_token = tokenizer.unk_token

        if hasattr(model.config, "pad_token_id"):
            model.config.pad_token_id = tokenizer.pad_token_id
        if hasattr(model.config, "bos_token_id"):
            model.config.bos_token_id = tokenizer.bos_token_id
        if hasattr(model.config, "eos_token_id"):
            model.config.eos_token_id = tokenizer.eos_token_id

        if hasattr(model, "generation_config") and model.generation_config is not None:
            model.generation_config.pad_token_id = tokenizer.pad_token_id
            model.generation_config.bos_token_id = tokenizer.bos_token_id
            model.generation_config.eos_token_id = tokenizer.eos_token_id

    def get_text_model(self, model):
        return model.model.language_model

    def get_text_config(self, model):
        return model.config.text_config if hasattr(model.config, "text_config") else model.config