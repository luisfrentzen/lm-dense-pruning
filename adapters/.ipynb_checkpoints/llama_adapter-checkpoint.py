from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn as nn
from safetensors.torch import load_file
from transformers import AutoTokenizer, LlamaForCausalLM

from adapters.base_adapter import BaseModelAdapter


class LlamaAdapter(BaseModelAdapter):
    name = "llama"

    def matches(self, model_dir: Path, checkpoint_manager) -> bool:
        return checkpoint_manager.detect_architecture(model_dir) == "llama"

    def load_pretrained_model(self, model_dir: Path, cfg, checkpoint_manager, load_kwargs=None):
        load_kwargs = load_kwargs or {}
        model_dir = model_dir.expanduser().resolve()
        return LlamaForCausalLM.from_pretrained(
            str(model_dir),
            **load_kwargs,
        )

    def load_tokenizer(self, model_dir: Path):
        return AutoTokenizer.from_pretrained(str(model_dir), trust_remote_code=True)

    def get_text_layers(self, model):
        return model.model.layers

    def get_text_model(self, model):
        return model.model

    def get_text_config(self, model):
        return model.config

    def rebuild_custom_model(self, model_dir: Path, cfg, checkpoint_manager):
        prune_config_path = checkpoint_manager.find_prune_config(model_dir)
        if prune_config_path is None:
            raise FileNotFoundError(f"prune_config.json not found in {model_dir}")

        with open(prune_config_path, "r") as f:
            meta = json.load(f)

        model = LlamaForCausalLM.from_pretrained(
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
        model.tie_weights()

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
        missing, unexpected = model.load_state_dict(state_dict, strict=False)

        if missing == ["lm_head.weight"] and not unexpected:
            self._repair_lm_head(model)
            return

        if missing or unexpected:
            print("Missing:", missing[:20])
            print("Unexpected:", unexpected[:20])
            raise RuntimeError("Checkpoint did not reload cleanly")

    def _repair_lm_head(self, model):
        input_embed = model.get_input_embeddings().weight

        if hasattr(model, "lm_head") and model.lm_head.weight.shape == input_embed.shape:
            model.lm_head.weight.data.copy_(input_embed.data)
        else:
            model.tie_weights()

    def load_base_model_and_tokenizer(self, base_model_path: str, torch_dtype):
        print("DEBUG LLAMA")
        tokenizer = AutoTokenizer.from_pretrained(base_model_path, local_files_only=True)
        model = LlamaForCausalLM.from_pretrained(
            base_model_path,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
            local_files_only=True,
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
