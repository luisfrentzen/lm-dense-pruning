"""Map a pruned model's surviving structures back to the unpruned model's indices.

Block-wise pruning removes whole heads / channels by index and leaves the kept
ones byte-identical (pre-recovery), so the mapping is exact. Two ways to obtain it:

  1. Record at prune time (robust, survives later recovery/fine-tuning):
       before pruning   -> fingerprint_structures(model, adapter)
       after pruning     -> resolve_kept_indices(model, adapter, fingerprints)
     Persist the result next to the checkpoint; positions never move during
     recovery, so the same map stays valid afterwards.

  2. Reconstruct post-hoc from a pre-recovery checkpoint pair:
       build_mapping_by_weights(unpruned_model, pruned_model, adapter)
     Exact for block-wise, pre-recovery models; used for legacy checkpoints that
     have no sidecar.

A StructureMapping stores, per layer, the original index of each surviving
structure: head[layer][p] = original head index of the p-th surviving head.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

import torch

from .structures import layer_dims


@dataclass
class StructureMapping:
    head: dict           # layer_idx -> list[int]  (original head index per surviving head)
    channel: dict        # layer_idx -> list[int]  (original channel index per surviving channel)
    meta: dict

    def kept(self, kind, layer):
        return (self.head if kind == "head" else self.channel)[layer]


# ---- byte fingerprints ------------------------------------------------------

def _hash(t) -> str:
    return hashlib.sha1(
        t.detach().to("cpu").contiguous().view(-1).numpy().tobytes()
    ).hexdigest()


def _head_block_hashes(q_weight, head_dim):
    H = q_weight.shape[0] // head_dim
    return [_hash(q_weight[h * head_dim : (h + 1) * head_dim, :]) for h in range(H)]


def _channel_row_hashes(gate_weight):
    return [_hash(gate_weight[j, :]) for j in range(gate_weight.shape[0])]


def fingerprint_structures(model, adapter, head_dim=None) -> dict:
    """Snapshot per-structure fingerprints of the *unpruned* model (call before pruning)."""
    layers = adapter.get_text_layers(model)
    cfg = adapter.get_text_config(model)
    fp = {"head_dim": None, "layers": {}}
    for i, layer in enumerate(layers):
        dims = layer_dims(layer, cfg, head_dim)
        fp["head_dim"] = dims.head_dim
        fp["layers"][i] = {
            "orig_heads": dims.num_heads,
            "orig_channels": dims.intermediate,
            "head_hashes": _head_block_hashes(layer.self_attn.q_proj.weight, dims.head_dim),
            "channel_hashes": _channel_row_hashes(layer.mlp.gate_proj.weight),
        }
    return fp


def resolve_kept_indices(pruned_model, adapter, fingerprints: dict) -> StructureMapping:
    """Match a pruned model's surviving structures against pre-prune fingerprints."""
    layers = adapter.get_text_layers(pruned_model)
    cfg = adapter.get_text_config(pruned_model)
    head_dim = fingerprints["head_dim"]

    head_map, chan_map = {}, {}
    for i, layer in enumerate(layers):
        ref = fingerprints["layers"][i] if i in fingerprints["layers"] else fingerprints["layers"][str(i)]
        head_index = {h: k for k, h in enumerate(ref["head_hashes"])}
        chan_index = {h: k for k, h in enumerate(ref["channel_hashes"])}

        dims = layer_dims(layer, cfg, head_dim)
        kept_heads = _lookup(_head_block_hashes(layer.self_attn.q_proj.weight, head_dim), head_index, f"layer {i} head")
        kept_chans = _lookup(_channel_row_hashes(layer.mlp.gate_proj.weight), chan_index, f"layer {i} channel")
        head_map[i] = kept_heads
        chan_map[i] = kept_chans

    meta = {
        "source": "fingerprint",
        "head_dim": head_dim,
        "num_layers": len(layers),
        "orig_heads": {i: fingerprints["layers"][i]["orig_heads"] for i in range(len(layers))} if 0 in fingerprints["layers"] else None,
    }
    return StructureMapping(head_map, chan_map, meta)


def _lookup(hashes, index, what):
    kept = []
    for h in hashes:
        if h not in index:
            raise ValueError(
                f"{what}: a surviving structure has no exact match in the unpruned "
                f"model. The pruned checkpoint is not a byte-identical block-wise "
                f"subset (was it recovered/fine-tuned, or pruned in another dimension?). "
                f"Record the mapping at prune time instead."
            )
        kept.append(index[h])
    return kept


# ---- weight-match reconstruction (legacy checkpoints) -----------------------

def build_mapping_by_weights(unpruned_model, pruned_model, adapter, head_dim=None) -> StructureMapping:
    """Reconstruct the mapping from a pre-recovery (unpruned, pruned) checkpoint pair."""
    fp = fingerprint_structures(unpruned_model, adapter, head_dim)
    mapping = resolve_kept_indices(pruned_model, adapter, fp)
    mapping.meta["source"] = "weight_match"

    # sanity: block-wise keeps hidden size and layer count fixed
    lu = adapter.get_text_layers(unpruned_model)
    lp = adapter.get_text_layers(pruned_model)
    if len(lu) != len(lp):
        raise ValueError(
            f"Layer count differs ({len(lu)} vs {len(lp)}); this tool targets block-wise "
            f"pruning (heads + MLP channels) which preserves the number of layers."
        )
    hu = lu[0].self_attn.q_proj.weight.shape[1]
    hp = lp[0].self_attn.q_proj.weight.shape[1]
    if hu != hp:
        raise ValueError(
            f"Hidden size differs ({hu} vs {hp}); block-wise pruning leaves the residual "
            f"stream intact. This checkpoint was also pruned in the hidden dimension."
        )
    return mapping


# ---- IO ---------------------------------------------------------------------

def save_mapping(mapping: StructureMapping, path):
    payload = {
        "head": {str(k): v for k, v in mapping.head.items()},
        "channel": {str(k): v for k, v in mapping.channel.items()},
        "meta": mapping.meta,
    }
    with open(path, "w") as f:
        json.dump(payload, f)


def load_mapping(path) -> StructureMapping:
    with open(path) as f:
        d = json.load(f)
    head = {int(k): v for k, v in d["head"].items()}
    channel = {int(k): v for k, v in d["channel"].items()}
    return StructureMapping(head, channel, d.get("meta", {}))
