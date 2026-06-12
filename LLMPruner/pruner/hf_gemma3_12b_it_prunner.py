import torch
import torch.nn as nn

import LLMPruner.torch_pruning as tp
from LLMPruner.torch_pruning import BasePruningFunc, ops

from typing import Sequence


##############################
# Helpers
##############################

def _safe_getattr(obj, names, default=None):
    for name in names:
        if hasattr(obj, name):
            return getattr(obj, name)
    return default


def _get_attention_hidden_size(layer: nn.Module) -> int:
    # Prefer live module shape after pruning
    if hasattr(layer, "q_proj") and hasattr(layer.q_proj, "in_features"):
        return layer.q_proj.in_features

    # LLaMA-style
    hidden_size = _safe_getattr(layer, ["hidden_size"], None)
    if hidden_size is not None:
        return hidden_size

    # Gemma3-style: stored on config
    config = getattr(layer, "config", None)
    if config is not None and hasattr(config, "hidden_size"):
        return config.hidden_size

    raise AttributeError(f"Cannot infer hidden_size for attention layer: {type(layer)}")


def _require_grad(weight: torch.Tensor, layer_name: str) -> torch.Tensor:
    if weight.grad is None:
        raise RuntimeError(
            f"Missing .grad for {layer_name}. "
            "Run backward() before Taylor importance."
        )
    return weight.grad


def _require_acc_grad(weight: torch.Tensor, layer_name: str) -> torch.Tensor:
    if not hasattr(weight, "acc_grad"):
        raise RuntimeError(
            f"Missing .acc_grad for {layer_name}. "
            "param_second/param_mix requires accumulated squared gradients."
        )
    return weight.acc_grad


##############################
# Pruners
##############################

class HFRMSNormPrunner(BasePruningFunc):

    def prune_out_channels(self, layer: nn.Module, idxs: Sequence[int]) -> nn.Module:
        keep_idxs = list(set(range(layer.weight.size(0))) - set(idxs))
        keep_idxs.sort()

        layer.weight = torch.nn.Parameter(layer.weight[keep_idxs])
        return layer

    prune_in_channels = prune_out_channels

    def get_out_channels(self, layer):
        return layer.weight.size(0)

    def get_in_channels(self, layer):
        return layer.weight.size(0)


class HFAttentionPrunner(BasePruningFunc):
    """
    Works for both:
      - HF LLaMA attention modules
      - HF Gemma3 attention modules

    This pruner removes hidden channels at the attention block boundary:
      - o_proj: prune output rows
      - q_proj/k_proj/v_proj: prune input columns

    That matches the hidden model dimension and does not require pruning
    q/k/v output heads directly.
    """

    def prune_out_channels(self, layer: nn.Module, idxs: Sequence[int]) -> nn.Module:
        # o_proj: prune output rows (hidden_size dimension)
        sub_layer = layer.o_proj
        keep_idxs = list(set(range(sub_layer.out_features)) - set(idxs))
        keep_idxs.sort()
        sub_layer.out_features = sub_layer.out_features - len(idxs)
        sub_layer.weight = torch.nn.Parameter(sub_layer.weight.data[keep_idxs])
        if sub_layer.bias is not None:
            sub_layer.bias = torch.nn.Parameter(sub_layer.bias.data[keep_idxs])

        # q_proj/k_proj/v_proj: prune input columns (hidden_size dimension)
        for sub_layer in [layer.q_proj, layer.k_proj, layer.v_proj]:
            keep_idxs = list(set(range(sub_layer.in_features)) - set(idxs))
            keep_idxs.sort()
            sub_layer.in_features = sub_layer.in_features - len(idxs)
            sub_layer.weight = torch.nn.Parameter(sub_layer.weight.data[:, keep_idxs])

        # Update local attention metadata if present.
        new_hidden_size = layer.q_proj.in_features
        if hasattr(layer, "hidden_size"):
            layer.hidden_size = new_hidden_size

        return layer

    prune_in_channels = prune_out_channels

    def get_out_channels(self, layer):
        return _get_attention_hidden_size(layer)

    def get_in_channels(self, layer):
        return _get_attention_hidden_size(layer)


class HFLinearPrunner(BasePruningFunc):
    TARGET_MODULES = ops.TORCH_LINEAR

    def __init__(self, mode: str = "delete"):
        assert mode in ["delete", "merge"], f"Unsupported mode: {mode}"
        self.mode = mode

    def prune_out_channels(self, layer: nn.Module, idxs: Sequence[int]) -> nn.Module:
        keep_idxs = sorted(set(range(layer.out_features)) - set(idxs))
        idxs = sorted(idxs)

        if len(idxs) == 0:
            return layer

        if self.mode == "delete":
            layer.out_features = len(keep_idxs)
            layer.weight = torch.nn.Parameter(layer.weight.data[keep_idxs].clone())
            if layer.bias is not None:
                layer.bias = torch.nn.Parameter(layer.bias.data[keep_idxs].clone())
            return layer

        # mode == "merge"
        layer.out_features = len(keep_idxs)

        keep_weight = layer.weight.data[keep_idxs].clone()
        remove_weight = layer.weight.data[idxs].clone()

        sim = torch.mm(remove_weight, keep_weight.t())
        max_indices = torch.argmax(sim, dim=-1)

        keep_weight.index_add_(0, max_indices, remove_weight)

        cnt = torch.ones(
            (keep_weight.size(0), 1),
            device=keep_weight.device,
            dtype=keep_weight.dtype,
        )
        cnt.index_add_(
            0,
            max_indices,
            torch.ones(
                (len(idxs), 1),
                device=keep_weight.device,
                dtype=keep_weight.dtype,
            ),
        )
        keep_weight = keep_weight / cnt
        layer.weight = torch.nn.Parameter(keep_weight)

        if layer.bias is not None:
            keep_bias = layer.bias.data[keep_idxs].clone()
            remove_bias = layer.bias.data[idxs].clone()

            keep_bias.index_add_(0, max_indices, remove_bias)

            bias_cnt = cnt.squeeze(-1).to(dtype=keep_bias.dtype, device=keep_bias.device)
            keep_bias = keep_bias / bias_cnt
            layer.bias = torch.nn.Parameter(keep_bias)

        return layer

    def prune_in_channels(self, layer: nn.Module, idxs: Sequence[int]) -> nn.Module:
        keep_idxs = sorted(set(range(layer.in_features)) - set(idxs))
        idxs = sorted(idxs)

        if len(idxs) == 0:
            return layer

        if self.mode == "delete":
            layer.in_features = len(keep_idxs)
            layer.weight = torch.nn.Parameter(layer.weight.data[:, keep_idxs].clone())
            return layer

        # mode == "merge"
        layer.in_features = len(keep_idxs)

        keep_weight = layer.weight.data[:, keep_idxs].clone()
        remove_weight = layer.weight.data[:, idxs].clone()

        sim = torch.mm(remove_weight.t(), keep_weight)
        max_indices = torch.argmax(sim, dim=-1)

        keep_weight.index_add_(1, max_indices, remove_weight)

        cnt = torch.ones(
            (1, keep_weight.size(1)),
            device=keep_weight.device,
            dtype=keep_weight.dtype,
        )
        cnt.index_add_(
            1,
            max_indices,
            torch.ones(
                (1, len(idxs)),
                device=keep_weight.device,
                dtype=keep_weight.dtype,
            ),
        )
        keep_weight = keep_weight / cnt
        layer.weight = torch.nn.Parameter(keep_weight)

        return layer

    def get_out_channels(self, layer):
        return layer.out_features

    def get_in_channels(self, layer):
        return layer.in_features


hf_attention_pruner = HFAttentionPrunner()
hf_rmsnorm_pruner = HFRMSNormPrunner()
# hf_linear_pruner = HFLinearPrunner()
# delete as default
hf_linear_pruner = HFLinearPrunner(mode="delete")
# hf_linear_pruner = HFLinearPrunner(mode="merge")


##############################
# Importance
##############################
class MagnitudeImportance(tp.importance.Importance):
    def __init__(self, p=2, group_reduction="mean", normalizer=None):
        self.p = p
        self.group_reduction = group_reduction
        self.normalizer = normalizer

    def _reduce(self, group_imp):
        if self.group_reduction == "sum":
            group_imp = group_imp.sum(dim=0)
        elif self.group_reduction == "mean":
            group_imp = group_imp.mean(dim=0)
        elif self.group_reduction == "max":
            group_imp = group_imp.max(dim=0)[0]
        elif self.group_reduction == "prod":
            group_imp = torch.prod(group_imp, dim=0)
        elif self.group_reduction == "first":
            group_imp = group_imp[0]
        elif self.group_reduction is None:
            group_imp = group_imp
        else:
            raise NotImplementedError
        return group_imp

    @torch.no_grad()
    def __call__(self, group, ch_groups=1, consecutive_groups=1):
        group_imp = []
        for dep, idxs in group:
            idxs.sort()
            layer = dep.target.module
            prune_fn = dep.handler

            # Linear out_channels
            if prune_fn in [tp.prune_linear_out_channels, hf_linear_pruner.prune_out_channels]:
                w = layer.weight.data[idxs].flatten(1)
                local_norm = w.abs().pow(self.p).sum(1)
                group_imp.append(local_norm)

            # Linear in_channels
            elif prune_fn in [tp.prune_linear_in_channels, hf_linear_pruner.prune_in_channels]:
                w = layer.weight
                local_norm = w.abs().pow(self.p).sum(0)

                max_idx = max(idxs) if len(idxs) > 0 else -1
                min_idx = min(idxs) if len(idxs) > 0 else -1
                if min_idx < 0 or max_idx >= local_norm.numel():
                    raise RuntimeError(
                        f"Invalid pruning idxs for linear in_channels: "
                        f"layer={layer.__class__.__name__}, "
                        f"weight_shape={tuple(layer.weight.shape)}, "
                        f"local_norm_numel={local_norm.numel()}, "
                        f"idx_range=[{min_idx}, {max_idx}], "
                        f"num_idxs={len(idxs)}"
                    )

                local_norm = local_norm[idxs]
                group_imp.append(local_norm)

            # RMSNorm (Gemma3 effective scale is 1 + weight)
            elif prune_fn == hf_rmsnorm_pruner.prune_out_channels:
                w = 1.0 + layer.weight.data[idxs]
                local_norm = w.abs().pow(self.p)
                group_imp.append(local_norm)

            # Embedding
            elif prune_fn == tp.prune_embedding_out_channels:
                w = layer.weight.data[:, idxs]
                local_norm = w.abs().pow(self.p).sum(0)
                group_imp.append(local_norm)

            # Attention
            elif prune_fn == hf_attention_pruner.prune_out_channels:
                local_norm = 0
                for sub_layer in [layer.o_proj]:
                    w_out = sub_layer.weight.data[idxs]
                    local_norm += w_out.abs().pow(self.p).sum(1)

                for sub_layer in [layer.q_proj, layer.k_proj, layer.v_proj]:
                    w_in = sub_layer.weight.data[:, idxs]
                    local_norm += w_in.abs().pow(self.p).sum(0)
                group_imp.append(local_norm)

        if len(group_imp) == 0:
            return None

        min_imp_size = min(len(imp) for imp in group_imp)
        aligned_group_imp = []
        for imp in group_imp:
            if len(imp) > min_imp_size and len(imp) % min_imp_size == 0:
                imp = imp.view(len(imp) // min_imp_size, min_imp_size).sum(0)
                aligned_group_imp.append(imp)
            elif len(imp) == min_imp_size:
                aligned_group_imp.append(imp)

        if len(aligned_group_imp) == 0:
            return None

        group_imp = torch.stack(aligned_group_imp, dim=0)
        group_imp = self._reduce(group_imp)
        if self.normalizer is not None:
            group_imp = self.normalizer(group, group_imp)
        return group_imp


class TaylorImportance(tp.importance.Importance):
    def __init__(self, group_reduction="sum", normalizer=None, taylor=None):
        self.group_reduction = group_reduction
        self.normalizer = normalizer
        self.taylor = taylor

    def _reduce(self, group_imp):
        if self.group_reduction == "sum":
            group_imp = group_imp.sum(dim=0)
        elif self.group_reduction == "mean":
            group_imp = group_imp.mean(dim=0)
        elif self.group_reduction == "max":
            group_imp = group_imp.max(dim=0)[0]
        elif self.group_reduction == "prod":
            group_imp = torch.prod(group_imp, dim=0)
        elif self.group_reduction == "first":
            group_imp = group_imp[0]
        elif self.group_reduction == "second":
            group_imp = group_imp[1]
        elif self.group_reduction is None:
            group_imp = group_imp
        else:
            raise NotImplementedError
        return group_imp

    @torch.no_grad()
    def __call__(self, group, ch_groups=1, consecutive_groups=1):
        group_imp = []
        for dep, idxs in group:
            idxs.sort()
            layer = dep.target.module
            prune_fn = dep.handler

            if prune_fn not in [
                tp.prune_linear_out_channels,
                tp.prune_linear_in_channels,
                hf_rmsnorm_pruner.prune_out_channels,
                tp.prune_embedding_out_channels,
                hf_attention_pruner.prune_out_channels,
                hf_linear_pruner.prune_out_channels,
                hf_linear_pruner.prune_in_channels,
            ]:
                continue

            if prune_fn == hf_attention_pruner.prune_out_channels:
                salience = {}
                for sub_layer in [layer.o_proj, layer.q_proj, layer.k_proj, layer.v_proj]:
                    grad = _require_grad(sub_layer.weight, f"{layer.__class__.__name__}.{sub_layer.__class__.__name__}")
                    salience[sub_layer] = sub_layer.weight * grad

                    if self.taylor in ["param_second"]:
                        acc_grad = _require_acc_grad(sub_layer.weight, f"{layer.__class__.__name__}.{sub_layer.__class__.__name__}")
                        salience[sub_layer] = sub_layer.weight * acc_grad * sub_layer.weight
                    elif self.taylor in ["param_mix"]:
                        acc_grad = _require_acc_grad(sub_layer.weight, f"{layer.__class__.__name__}.{sub_layer.__class__.__name__}")
                        salience[sub_layer] = (
                            salience[sub_layer]
                            - 0.5 * sub_layer.weight * acc_grad * sub_layer.weight
                        )
            else:
                grad = _require_grad(layer.weight, layer.__class__.__name__)
                salience = layer.weight * grad

                if self.taylor in ["param_second"]:
                    acc_grad = _require_acc_grad(layer.weight, layer.__class__.__name__)
                    salience = layer.weight * acc_grad * layer.weight
                elif self.taylor in ["param_mix"]:
                    acc_grad = _require_acc_grad(layer.weight, layer.__class__.__name__)
                    salience = salience - 0.5 * layer.weight * acc_grad * layer.weight

            # Linear out_channels
            if prune_fn in [tp.prune_linear_out_channels, hf_linear_pruner.prune_out_channels]:
                if self.taylor == "vectorize":
                    local_norm = salience.sum(1).abs()
                elif self.taylor is not None and "param" in self.taylor:
                    local_norm = salience.abs().sum(1)
                else:
                    raise NotImplementedError
                local_norm = local_norm[idxs]
                group_imp.append(local_norm)

            # Linear in_channels
            elif prune_fn in [tp.prune_linear_in_channels, hf_linear_pruner.prune_in_channels]:
                if self.taylor == "vectorize":
                    local_norm = salience.sum(0).abs()
                elif self.taylor is not None and "param" in self.taylor:
                    local_norm = salience.abs().sum(0)
                else:
                    raise NotImplementedError

                max_idx = max(idxs) if len(idxs) > 0 else -1
                min_idx = min(idxs) if len(idxs) > 0 else -1
                if min_idx < 0 or max_idx >= local_norm.numel():
                    raise RuntimeError(
                        f"Invalid pruning idxs for linear in_channels: "
                        f"layer={layer.__class__.__name__}, "
                        f"weight_shape={tuple(layer.weight.shape)}, "
                        f"local_norm_numel={local_norm.numel()}, "
                        f"idx_range=[{min_idx}, {max_idx}], "
                        f"num_idxs={len(idxs)}"
                    )

                local_norm = local_norm[idxs]
                group_imp.append(local_norm)

            # RMSNorm
            elif prune_fn == hf_rmsnorm_pruner.prune_out_channels:
                local_norm = salience.abs()[idxs]
                group_imp.append(local_norm)

            # Embedding
            elif prune_fn == tp.prune_embedding_out_channels:
                if self.taylor == "vectorize":
                    local_norm = salience[:, idxs].sum(0).abs()
                elif self.taylor is not None and "param" in self.taylor:
                    local_norm = salience[:, idxs].abs().sum(0)
                else:
                    raise NotImplementedError
                group_imp.append(local_norm)

            # Attention
            elif prune_fn == hf_attention_pruner.prune_out_channels:
                local_norm = 0
                for sub_layer in [layer.o_proj]:
                    if self.taylor == "vectorize":
                        local_norm += salience[sub_layer].sum(1).abs()
                    elif self.taylor is not None and "param" in self.taylor:
                        local_norm += salience[sub_layer].abs().sum(1)
                    else:
                        raise NotImplementedError

                for sub_layer in [layer.q_proj, layer.k_proj, layer.v_proj]:
                    if self.taylor == "vectorize":
                        local_norm += salience[sub_layer].sum(0).abs()
                    elif self.taylor is not None and "param" in self.taylor:
                        local_norm += salience[sub_layer].abs().sum(0)
                    else:
                        raise NotImplementedError

                local_norm = local_norm[idxs]
                group_imp.append(local_norm)

        if len(group_imp) == 0:
            return None

        min_imp_size = min(len(imp) for imp in group_imp)
        aligned_group_imp = []
        for imp in group_imp:
            if len(imp) > min_imp_size and len(imp) % min_imp_size == 0:
                imp = imp.view(len(imp) // min_imp_size, min_imp_size).sum(0)
                aligned_group_imp.append(imp)
            elif len(imp) == min_imp_size:
                aligned_group_imp.append(imp)

        if len(aligned_group_imp) == 0:
            return None

        group_imp = torch.stack(aligned_group_imp, dim=0)
        group_imp = self._reduce(group_imp)
        if self.normalizer is not None:
            group_imp = self.normalizer(group, group_imp)
        return group_imp