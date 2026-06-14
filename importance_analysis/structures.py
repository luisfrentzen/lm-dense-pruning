"""Locate and slice the prunable block-wise structures of a decoder layer.

The block-wise pruner in this repo prunes two unit types per text layer:
  - attention heads   : groups of `head_dim` consecutive q_proj output channels
                        (o_proj input channels move with them; in GQA each kv head
                        is shared by `Hq // Hkv` query heads)
  - MLP channels      : gate_proj/up_proj output rows (down_proj input columns)

This module exposes those units in a model-agnostic way (llama / gemma3) via the
existing adapters, plus per-element "contribution" reducers used by the metrics.
"""

from __future__ import annotations

from dataclasses import dataclass


def resolve_adapter(model_type: str):
    mt = model_type.lower()
    if mt == "gemma3":
        from adapters.gemma3_adapter import Gemma3Adapter
        return Gemma3Adapter()
    if mt == "llama":
        from adapters.llama_adapter import LlamaAdapter
        return LlamaAdapter()
    raise ValueError(f"Unsupported model_type: {model_type!r}. Use 'gemma3' or 'llama'.")


@dataclass(frozen=True)
class LayerDims:
    hidden: int
    head_dim: int
    num_heads: int        # query heads (the attention pruning unit)
    num_kv_heads: int
    intermediate: int     # MLP channels (the mlp pruning unit)

    @property
    def gqa_group(self) -> int:
        return self.num_heads // self.num_kv_heads


def layer_dims(layer, text_cfg, head_dim_override: int | None = None) -> LayerDims:
    attn = layer.self_attn
    mlp = layer.mlp

    hidden = int(attn.q_proj.weight.shape[1])
    q_out = int(attn.q_proj.weight.shape[0])
    kv_out = int(attn.k_proj.weight.shape[0])

    if head_dim_override is not None:
        head_dim = head_dim_override
    else:
        head_dim = getattr(text_cfg, "head_dim", None)
        if not head_dim:
            # llama-style configs may leave head_dim unset; derive it from heads
            head_dim = hidden // int(text_cfg.num_attention_heads)

    if q_out % head_dim != 0 or kv_out % head_dim != 0:
        raise ValueError(
            f"q_out={q_out} / kv_out={kv_out} not divisible by head_dim={head_dim}; "
            f"pass an explicit head_dim."
        )

    return LayerDims(
        hidden=hidden,
        head_dim=head_dim,
        num_heads=q_out // head_dim,
        num_kv_heads=kv_out // head_dim,
        intermediate=int(mlp.gate_proj.weight.shape[0]),
    )


# ---- per-element contribution reducers -------------------------------------
# Each reducer maps (weight, grad) -> a non-negative float tensor shaped like the
# weight. Structures aggregate these by summing over their slice of the weight.

def contrib_magnitude(weight, grad):
    return weight.detach().float().abs()


def contrib_taylor(weight, grad):
    # First-order Taylor importance |w * dL/dw| -- the pruner's param_first criterion.
    return (weight.detach().float() * grad.float()).abs()


def contrib_fisher(weight, grad):
    # Diagonal empirical Fisher: (dL/dw)^2 (accumulated/averaged over samples).
    return grad.float().pow(2)
