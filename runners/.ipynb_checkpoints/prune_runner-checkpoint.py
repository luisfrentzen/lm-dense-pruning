import gc
import json
import random
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import numpy as np
import torch

import LLMPruner.torch_pruning as tp
from LLMPruner.datasets.example_samples import get_examples
from LLMPruner.pruner import hf_gemma3_12b_it_prunner as gemma_pruner
from LLMPruner.templates.prompts import prompts
from LLMPruner.utils.logger import LoggerWithDepth

from core.prune_config import PruneConfig

def _resolve_model_type(model_type: str):
    print(model_type)
    if model_type == "gemma3":
        from transformers.models.gemma3.modeling_gemma3 import Gemma3Attention, Gemma3RMSNorm
        from adapters.gemma3_adapter import Gemma3Adapter
        return Gemma3Adapter(), Gemma3Attention, Gemma3RMSNorm
    elif model_type == "llama":
        from transformers.models.llama.modeling_llama import LlamaAttention, LlamaRMSNorm
        from adapters.llama_adapter import LlamaAdapter
        return LlamaAdapter(), LlamaAttention, LlamaRMSNorm

    raise ValueError(f"Unsupported model_type: {model_type!r}. Use 'gemma3' or 'llama'.")

from core.utils import (
    cleanup_memory,
    count_trainable_parameters,
    get_torch_dtype,
    log_cuda_memory,
    move_batch_to_device,
    set_random_seed,
    show_step,
)

from tqdm import tqdm

class PruneRunner:
    def __init__(self, config_path="config/prune_config.yaml"):
        self.cfg = PruneConfig.from_yaml(config_path)

        self.run_name = self.build_run_name()
        self.logger = None
        self.tokenizer = None
        self.model = None
        self.text_model = None
        self.adapter = None
        self._attention_cls = None
        self._rmsnorm_cls = None

    def build_run_name(self):
        model_tag = f"{self.cfg.base_model.split('/')[-1]}"

        return f"{model_tag}_P{self.cfg.pruning_ratio}{self.cfg.dataset}"

    def get_input_device(self):
        return next(self.model.parameters()).device

    def setup(self):
        self.adapter, self._attention_cls, self._rmsnorm_cls = _resolve_model_type(self.cfg.model_type)
        set_random_seed(self.cfg.seed)

        self.logger = LoggerWithDepth(
            env_name=self.run_name,
            config=vars(self.cfg),
            root_dir="prune_log",
            setup_sublogger=True,
        )

        load_dtype = get_torch_dtype(self.cfg.device)
        print("DEBUG", self.cfg.base_model, self.cfg.model_type)
        self.model, self.tokenizer = self.adapter.load_base_model_and_tokenizer(
            self.cfg.base_model,
            torch_dtype=load_dtype,
        )
        self.model.to(self.get_input_device())
        
        self.adapter.set_special_tokens(self.model, self.tokenizer)
        self.text_model = self.adapter.get_text_model(self.model)
        self.model.config.use_cache = False
        self.text_model.config.use_cache = False

    def build_forward_prompts(self):
        seed_texts = [
            "Large language models can be pruned efficiently.",
            "This input is only used to build the dependency graph.",
        ]
        enc = self.tokenizer(
            seed_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=32,
        )
        return {
            "input_ids": enc["input_ids"].to(self.get_input_device()),
            "attention_mask": enc["attention_mask"].to(self.get_input_device()),
        }

    def clear_acc_grad(self, module):
        for p in module.parameters():
            if hasattr(p, "acc_grad"):
                delattr(p, "acc_grad")

    def clear_weight_grads(self, module):
        module.zero_grad()
        for name, param in module.named_parameters():
            if "weight" in name:
                param.grad = None

    def run_taylor_backward(self, iterative_step):
        self.clear_acc_grad(self.model)

        example_prompts = get_examples(
            "bookcorpus",
            self.tokenizer,
            self.cfg.num_examples,
            seq_len=64,
        ).to(self.get_input_device())

        self.logger.log(f"Start Backwarding in iterative steps = {iterative_step}...")

        if self.cfg.taylor in ["param_mix", "param_second"]:
            for j in range(self.cfg.num_examples):
                batch_input = example_prompts[j].unsqueeze(0)
                loss = self.model(input_ids=batch_input, labels=batch_input).loss
                self.logger.log(f"Loss = {loss}")
                loss.backward()

                for module_param in self.model.parameters():
                    if module_param.grad is None:
                        continue
                    sq_grad = module_param.grad * module_param.grad / self.cfg.num_examples
                    if hasattr(module_param, "acc_grad"):
                        module_param.acc_grad += sq_grad
                    else:
                        module_param.acc_grad = sq_grad.clone()

                self.model.zero_grad()

        loss = self.model(input_ids=example_prompts, labels=example_prompts).loss
        self.logger.log(f"Loss = {loss}")
        loss.backward()

    def collect_layer_shapes(self):
        layers = self.adapter.get_text_layers(self.model)
        info = []

        for i, layer in enumerate(layers):
            attn = layer.self_attn
            mlp = layer.mlp

            info.append({
                "layer": i,
                "q_out": int(attn.q_proj.weight.shape[0]),
                "q_in": int(attn.q_proj.weight.shape[1]),
                "k_out": int(attn.k_proj.weight.shape[0]),
                "k_in": int(attn.k_proj.weight.shape[1]),
                "v_out": int(attn.v_proj.weight.shape[0]),
                "v_in": int(attn.v_proj.weight.shape[1]),
                "o_out": int(attn.o_proj.weight.shape[0]),
                "o_in": int(attn.o_proj.weight.shape[1]),
                "gate_out": int(mlp.gate_proj.weight.shape[0]),
                "gate_in": int(mlp.gate_proj.weight.shape[1]),
                "up_out": int(mlp.up_proj.weight.shape[0]),
                "up_in": int(mlp.up_proj.weight.shape[1]),
                "down_out": int(mlp.down_proj.weight.shape[0]),
                "down_in": int(mlp.down_proj.weight.shape[1]),
            })

        return info

    def can_save_as_pretrained(self):
        layer_shapes = self.collect_layer_shapes()
        if not layer_shapes:
            return False, None, None, None, "No text layers found."

        patterns = [{k: v for k, v in s.items() if k != "layer"} for s in layer_shapes]
        if len({json.dumps(p, sort_keys=True) for p in patterns}) != 1:
            return False, None, None, None, "Per-layer shapes are not uniform."

        s = patterns[0]
        text_cfg = self.adapter.get_text_config(self.model)
        hidden_size = text_cfg.hidden_size

        if not (
            s["q_in"] == hidden_size and
            s["k_in"] == hidden_size and
            s["v_in"] == hidden_size and
            s["o_out"] == hidden_size and
            s["gate_in"] == hidden_size and
            s["up_in"] == hidden_size and
            s["down_out"] == hidden_size
        ):
            return False, None, None, None, "Hidden-size side does not match config."

        if not (s["gate_out"] == s["up_out"] == s["down_in"]):
            return False, None, None, None, "MLP shape is inconsistent."

        if not (s["q_out"] == s["o_in"] and s["k_out"] == s["v_out"]):
            return False, None, None, None, "Attention shape is inconsistent."

        head_dim = text_cfg.head_dim
        if s["q_out"] % head_dim != 0 or s["k_out"] % head_dim != 0:
            return False, None, None, None, "Attention dims are not divisible by head_dim."

        new_num_attention_heads = s["q_out"] // head_dim
        new_num_key_value_heads = s["k_out"] // head_dim
        new_intermediate_size = s["gate_out"]

        return True, new_intermediate_size, new_num_attention_heads, new_num_key_value_heads, "Uniform standard architecture."

    def test_generation_before_pruning(self):
        if not self.cfg.test_before_train:
            return

        self.logger.log("\n==================Generation Results before Pruning================\n")
        self.model.eval()

        with torch.no_grad():
            for prompt in prompts:
                enc = self.tokenizer(prompt, return_tensors="pt")
                enc = move_batch_to_device(enc, self.get_input_device())

                generation_output = self.model.generate(
                    **enc,
                    do_sample=True,
                    top_k=50,
                    max_new_tokens=self.cfg.max_seq_len,
                    top_p=self.cfg.top_p,
                    temperature=self.cfg.temperature,
                )

                result = self.tokenizer.decode(generation_output[0], skip_special_tokens=False)
                self.logger.log(result)

    def build_importance(self):
        pruner_type = self.cfg.pruner_type.lower()
        assert pruner_type in ["random", "l2", "l1", "taylor"]

        if pruner_type == "random":
            return tp.importance.RandomImportance()
        if pruner_type == "l1":
            return gemma_pruner.MagnitudeImportance(
                p=1,
                group_reduction=self.cfg.grouping_strategy,
            )
        if pruner_type == "l2":
            return gemma_pruner.MagnitudeImportance(
                p=2,
                group_reduction=self.cfg.grouping_strategy,
            )
        if pruner_type == "taylor":
            return gemma_pruner.TaylorImportance(
                group_reduction=self.cfg.grouping_strategy,
                taylor=self.cfg.taylor,
            )

        raise NotImplementedError

    def enable_grads(self):
        for param in self.model.parameters():
            param.requires_grad_(True)

    def run_block_wise_pruning(self, imp, before_pruning_parameters):
        num_layers = len(self.text_model.layers)
        attn_start = max(0, min(self.cfg.block_attention_layer_start, num_layers))
        attn_end = max(attn_start, min(self.cfg.block_attention_layer_end, num_layers))
        mlp_start = max(0, min(self.cfg.block_mlp_layer_start, num_layers))
        mlp_end = max(mlp_start, min(self.cfg.block_mlp_layer_end, num_layers))

        kwargs = {
            "importance": imp,
            "global_pruning": self.cfg.global_pruning,
            "iterative_steps": self.cfg.iterative_steps,
            "ch_sparsity": self.cfg.pruning_ratio,
            "ignored_layers": [],
            "channel_groups": {},
            "consecutive_groups": {
                layer.self_attn.q_proj: layer.self_attn.head_dim for layer in self.text_model.layers
            },
            "customized_pruners": {
                self._rmsnorm_cls: gemma_pruner.hf_rmsnorm_pruner,
                self._attention_cls: gemma_pruner.hf_attention_pruner,
            },
            "root_module_types": None,
            "root_instances":
                [self.text_model.layers[i].self_attn.q_proj for i in range(attn_start, attn_end)] +
                [self.text_model.layers[i].mlp.gate_proj for i in range(mlp_start, mlp_end)]
        }

        self.logger.log(f"Pruning Attention Layer = {list(range(attn_start, attn_end))}")
        self.logger.log(f"Pruning MLP Layer = {list(range(mlp_start, mlp_end))}")

        forward_prompts = self.build_forward_prompts()
        pruner = tp.pruner.MetaPruner(self.text_model, forward_prompts, **kwargs)
        self.model.zero_grad()

        after_pruning_parameters = before_pruning_parameters

        self.logger.log("Start Pruning")
        for i in range(self.cfg.iterative_steps):
            if self.cfg.pruner_type.lower() == "taylor":
                self.run_taylor_backward(i)

            pruner.step()

            after_pruning_parameters = count_trainable_parameters(self.text_model)
            self.logger.log(
                f"After Iter {i + 1}/{self.cfg.iterative_steps}, #parameters: {after_pruning_parameters}"
            )

        self.clear_weight_grads(self.model)
        self.clear_acc_grad(self.model)
        del pruner

        return after_pruning_parameters

    def run_channel_wise_pruning(self, imp, before_pruning_parameters):
        kwargs = {
            "importance": imp,
            "global_pruning": self.cfg.global_pruning,
            "iterative_steps": self.cfg.iterative_steps,
            "ch_sparsity": self.cfg.pruning_ratio,
            "ignored_layers": [],
            "channel_groups": {},
            "customized_pruners": {
                self._rmsnorm_cls: gemma_pruner.hf_rmsnorm_pruner,
                self._attention_cls: gemma_pruner.hf_attention_pruner,
            },
            "root_module_types": [self._rmsnorm_cls, self._attention_cls],
        }

        forward_prompts = self.build_forward_prompts()
        pruner = tp.pruner.MetaPruner(self.text_model, forward_prompts, **kwargs)
        self.model.zero_grad()

        after_pruning_parameters = before_pruning_parameters

        self.logger.log("Start Pruning")
        for i in range(self.cfg.iterative_steps):
            if self.cfg.pruner_type.lower() == "taylor":
                self.run_taylor_backward(i)

            pruner.step()

            after_pruning_parameters = count_trainable_parameters(self.text_model)
            self.logger.log(
                f"After Iter {i + 1}/{self.cfg.iterative_steps}, #parameters: {after_pruning_parameters}"
            )

        self.clear_weight_grads(self.model)
        self.clear_acc_grad(self.model)

        if hasattr(self.text_model, "embed_tokens"):
            new_hidden_size = self.text_model.embed_tokens.weight.shape[1]
            if hasattr(self.text_model.config, "hidden_size"):
                self.text_model.config.hidden_size = new_hidden_size
            if hasattr(self.model.config, "text_config") and hasattr(self.model.config.text_config, "hidden_size"):
                self.model.config.text_config.hidden_size = new_hidden_size

        self.model.zero_grad()
        del pruner

        return after_pruning_parameters

    def run_layer_wise_pruning(self):
        num_layers = len(self.text_model.layers)
        keep_layers = max(1, min(self.cfg.layer, num_layers))
        self.text_model.layers = self.text_model.layers[:keep_layers]

        if hasattr(self.text_model.config, "num_hidden_layers"):
            self.text_model.config.num_hidden_layers = keep_layers
        if hasattr(self.model.config, "text_config") and hasattr(self.model.config.text_config, "num_hidden_layers"):
            self.model.config.text_config.num_hidden_layers = keep_layers

        return count_trainable_parameters(self.text_model)

    def prune(self):
        self.enable_grads()

        before_pruning_parameters = count_trainable_parameters(self.text_model)
        imp = self.build_importance()

        self.logger.log(f"Use {self.cfg.pruner_type.lower()} pruner...")

        if self.cfg.block_wise:
            after_pruning_parameters = self.run_block_wise_pruning(imp, before_pruning_parameters)
        elif self.cfg.channel_wise:
            after_pruning_parameters = self.run_channel_wise_pruning(imp, before_pruning_parameters)
        elif self.cfg.layer_wise:
            after_pruning_parameters = self.run_layer_wise_pruning()
        else:
            raise NotImplementedError("Enable one of: block_wise, channel_wise, layer_wise")

        self.logger.log(
            "#Text Param before: {}, #Text Param after: {}, Ratio = {:.4f}%".format(
                before_pruning_parameters,
                after_pruning_parameters,
                100.0 * after_pruning_parameters / before_pruning_parameters
            )
        )

    def save_model(self):
        if not self.cfg.save_model:
            return

        save_dir = os.path.join(self.cfg.save_dir, self.run_name)
        os.makedirs(save_dir, exist_ok=True)

        self.tokenizer.save_pretrained(save_dir)

        ok, new_intermediate_size, new_num_attention_heads, new_num_key_value_heads, reason = self.can_save_as_pretrained()

        if ok:
            text_cfg = self.adapter.get_text_config(self.model)
            text_cfg.intermediate_size = new_intermediate_size
            text_cfg.num_attention_heads = new_num_attention_heads
            text_cfg.num_key_value_heads = new_num_key_value_heads
            text_cfg.num_hidden_layers = len(self.adapter.get_text_layers(self.model))

            self.model.save_pretrained(save_dir, safe_serialization=True)
            print("Saved as standard pretrained model:", save_dir)
        else:
            torch.save(self.model.state_dict(), os.path.join(save_dir, "pytorch_model.bin"))

            meta = {
                "base_model_name": str(self.cfg.base_model),
                "num_hidden_layers": len(self.adapter.get_text_layers(self.model)),
                "layer_shapes": self.collect_layer_shapes(),
                "model_class": self.model.__class__.__name__,
                "torch_dtype": str(next(self.model.parameters()).dtype),
                "reason": reason,
            }

            with open(os.path.join(save_dir, "prune_config.json"), "w") as f:
                json.dump(meta, f, indent=2)

            print("Saved as custom pruned checkpoint:", save_dir)
            print("Reason:", reason)

    def prepare_for_eval(self):
        self.model.to(self.cfg.eval_device)
        self.adapter.set_special_tokens(self.model, self.tokenizer)

        self.model.config.use_cache = True
        self.text_model.config.use_cache = True

    def test_generation_after_pruning(self):
        if not self.cfg.test_after_train:
            return

        self.logger.log("\n==================Generation Results After Pruning================\n")

        self.model.eval()
        with torch.no_grad():
            for prompt in prompts:
                enc = self.tokenizer(prompt, return_tensors="pt").to(self.cfg.eval_device)

                generation_output = self.model.generate(
                    **enc,
                    do_sample=True,
                    top_k=50,
                    max_new_tokens=self.cfg.max_seq_len,
                    top_p=self.cfg.top_p,
                    temperature=self.cfg.temperature,
                )

                result = self.tokenizer.decode(generation_output[0], skip_special_tokens=False)
                self.logger.log(result)

        self.logger.log("\n==================Finish================\n")

    def log_memory(self):
        msg = log_cuda_memory()
        if msg and self.logger is not None:
            self.logger.log(msg)

    def run(self):
        show_step("Step 1/8: setup")
        self.setup()

        show_step("Step 2/8: test_generation_before_pruning")
        self.test_generation_before_pruning()

        show_step("Step 3/8: prune")
        self.prune()

        show_step("Step 4/8: cleanup_memory")
        cleanup_memory()

        show_step("Step 5/8: save_model")
        self.save_model()

        show_step("Step 6/8: prepare_for_eval")
        self.prepare_for_eval()

        show_step("Step 7/8: test_generation_after_pruning")
        self.test_generation_after_pruning()

        show_step("Step 8/8: log_memory")
        self.log_memory()