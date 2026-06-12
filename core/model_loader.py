from __future__ import annotations

import torch
import torch.nn as nn
from transformers import BitsAndBytesConfig

from adapters.gemma3_adapter import Gemma3Adapter


class ModelLoader:
    def __init__(self, cfg, checkpoint_manager, adapters=None):
        self.cfg = cfg
        self.ckpt = checkpoint_manager
        self.adapters = adapters or [Gemma3Adapter()]

    def get_adapter(self, model_dir):
        for adapter in self.adapters:
            if adapter.matches(model_dir, self.ckpt):
                return adapter
        raise ValueError(f"No adapter found for: {model_dir}")

    def import_bitsandbytes(self):
        try:
            import bitsandbytes as bnb
            return bnb
        except Exception as e:
            raise ImportError(
                "8bit quantization requires bitsandbytes. "
                "Install with: pip install --upgrade transformers accelerate bitsandbytes"
            ) from e

    def should_skip_int8_module(self, module_name: str) -> bool:
        for name in self.cfg.int8_skip_modules:
            if module_name == name or module_name.endswith("." + name):
                return True
        return False

    def convert_linear_layers_to_int8(self, module, prefix=""):
        """
        Replace nn.Linear -> bitsandbytes.nn.Linear8bitLt recursively.
        Modules like lm_head can be skipped.
        """
        bnb = self.import_bitsandbytes()
        converted = []

        for child_name, child in list(module.named_children()):
            full_name = f"{prefix}.{child_name}" if prefix else child_name

            if isinstance(child, nn.Linear) and not self.should_skip_int8_module(full_name):
                int8_layer = bnb.nn.Linear8bitLt(
                    child.in_features,
                    child.out_features,
                    bias=(child.bias is not None),
                    has_fp16_weights=False,
                    threshold=self.cfg.int8_threshold,
                )
                int8_layer.load_state_dict(child.state_dict(), strict=False)
                int8_layer.requires_grad_(False)

                setattr(module, child_name, int8_layer)
                converted.append(full_name)
            else:
                converted.extend(self.convert_linear_layers_to_int8(child, full_name))

        return converted

    def get_pretrained_load_kwargs(self):
        kwargs = {
            "torch_dtype": self.cfg.dtype,
            "trust_remote_code": True,
        }

        if self.cfg.quantization == "8bit":
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_threshold=self.cfg.int8_threshold,
            )

            if self.cfg.use_multi_gpu and torch.cuda.is_available() and self.cfg.effective_gpu_count() > 1:
                kwargs["device_map"] = "auto"
                if self.cfg.max_memory_per_gpu is not None:
                    kwargs["max_memory"] = {
                        i: self.cfg.max_memory_per_gpu
                        for i in range(self.cfg.effective_gpu_count())
                    }
            else:
                kwargs["device_map"] = "auto"

        return kwargs

    def maybe_move_model_to_device(self, model):
        if self.cfg.quantization == "8bit":
            # pretrained 8bit usually uses device_map during loading,
            # custom 8bit quantizes when moved to CUDA
            if not torch.cuda.is_available():
                return model

        if self.cfg.device.startswith("cuda") and torch.cuda.is_available():
            return model.to(self.cfg.device)

        return model

    def maybe_convert_custom_model_to_int8(self, model, model_dir):
        if self.cfg.quantization != "8bit":
            return model

        if not torch.cuda.is_available():
            raise RuntimeError(
                "Custom 8bit path currently requires CUDA, because Linear8bitLt "
                "quantizes when the module is moved to CUDA."
            )

        converted_layers = self.convert_linear_layers_to_int8(model)
        print(
            f"[{model_dir.name}] converted {len(converted_layers)} linear layers to 8bit "
            f"(skipped={self.cfg.int8_skip_modules})"
        )
        return model

    def load_pretrained_model(self, model_dir):
        adapter = self.get_adapter(model_dir)

        load_kwargs = self.get_pretrained_load_kwargs()
        model = adapter.load_pretrained_model(
            model_dir=model_dir,
            cfg=self.cfg,
            checkpoint_manager=self.ckpt,
            load_kwargs=load_kwargs,
        )

        tokenizer = adapter.load_tokenizer(model_dir)

        if self.cfg.quantization != "8bit":
            model = self.maybe_move_model_to_device(model)

        return {
            "model": model,
            "tokenizer": tokenizer,
            "model_type": "pretrained",
            "adapter": adapter,
        }

    def load_custom_model(self, model_dir):
        adapter = self.get_adapter(model_dir)

        model = adapter.rebuild_custom_model(
            model_dir=model_dir,
            cfg=self.cfg,
            checkpoint_manager=self.ckpt,
        )

        tokenizer = adapter.load_tokenizer(model_dir)

        model = self.maybe_convert_custom_model_to_int8(model, model_dir)
        model = self.maybe_move_model_to_device(model)

        return {
            "model": model,
            "tokenizer": tokenizer,
            "model_type": "custom",
            "adapter": adapter,
        }

    def load_model(self, model_dir):
        checkpoint_type = self.ckpt.get_checkpoint_type(model_dir)

        if checkpoint_type == "custom":
            return self.load_custom_model(model_dir)

        if checkpoint_type == "pretrained":
            return self.load_pretrained_model(model_dir)

        raise ValueError(f"{model_dir} is neither pretrained nor custom checkpoint.")