"""Per-structure importance scores for a model (attention heads + MLP channels).

A "metric" decides how a single weight element contributes to its structure's
score; aggregation over the structure's slice is shared. Three are built in:

    magnitude : sum |w|                       (no backward; baseline)
    taylor    : sum |w * dL/dw|               (pruner's param_first criterion)
    fisher    : mean_samples sum (dL/dw)^2    (diagonal empirical Fisher)

Add a metric by registering a (needs_backward, per_sample, contrib_fn) entry in
`_METRICS`; `contrib_fn(weight, grad) -> float tensor` shaped like the weight.
The loss feeding the backward is pluggable (`loss_fn(model, input_ids)->scalar`,
default causal-LM) so importance can reflect any objective.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from .structures import (
    LayerDims,
    contrib_fisher,
    contrib_magnitude,
    contrib_taylor,
    layer_dims,
)

# name -> (needs_backward, per_sample, contrib_fn)
_METRICS = {
    "magnitude": (False, False, contrib_magnitude),
    "taylor": (True, False, contrib_taylor),
    "fisher": (True, True, contrib_fisher),
}


def list_metrics():
    return sorted(_METRICS)


def register_metric(name, contrib_fn, needs_backward=True, per_sample=False):
    """Add a custom metric. See module docstring for the contract."""
    _METRICS[name] = (needs_backward, per_sample, contrib_fn)


def clm_loss(model, input_ids):
    return model(input_ids=input_ids, labels=input_ids).loss


@dataclass
class ImportanceResult:
    metric: str
    head_scores: dict           # layer_idx -> 1D float tensor [num_heads]
    channel_scores: dict        # layer_idx -> 1D float tensor [intermediate]
    meta: dict = field(default_factory=dict)

    def kinds(self):
        return ("head", "channel")

    def scores(self, kind):
        return self.head_scores if kind == "head" else self.channel_scores

    def save(self, path):
        torch.save(
            {
                "metric": self.metric,
                "head_scores": {i: v.cpu() for i, v in self.head_scores.items()},
                "channel_scores": {i: v.cpu() for i, v in self.channel_scores.items()},
                "meta": self.meta,
            },
            path,
        )

    @classmethod
    def load(cls, path):
        d = torch.load(path, map_location="cpu", weights_only=False)
        return cls(d["metric"], d["head_scores"], d["channel_scores"], d.get("meta", {}))


# ---- per-layer aggregation --------------------------------------------------

def _head_scores(layer, dims: LayerDims, contrib_fn):
    a = layer.self_attn
    d, Hq, Hkv = dims.head_dim, dims.num_heads, dims.num_kv_heads

    # q rows and o columns are organised head-by-head -> reduce to [Hq]
    s = contrib_fn(a.q_proj.weight, a.q_proj.weight.grad).sum(1).view(Hq, d).sum(1)
    s = s + contrib_fn(a.o_proj.weight, a.o_proj.weight.grad).sum(0).view(Hq, d).sum(1)

    # k/v live at kv-head granularity; attribute each kv head to its query group
    kc = contrib_fn(a.k_proj.weight, a.k_proj.weight.grad).sum(1).view(Hkv, d).sum(1)
    vc = contrib_fn(a.v_proj.weight, a.v_proj.weight.grad).sum(1).view(Hkv, d).sum(1)
    s = s + (kc + vc).repeat_interleave(dims.gqa_group)
    return s


def _channel_scores(layer, contrib_fn):
    m = layer.mlp
    s = contrib_fn(m.gate_proj.weight, m.gate_proj.weight.grad).sum(1)
    s = s + contrib_fn(m.up_proj.weight, m.up_proj.weight.grad).sum(1)
    s = s + contrib_fn(m.down_proj.weight, m.down_proj.weight.grad).sum(0)
    return s


def _enable_grads(layers):
    for layer in layers:
        for proj in (
            layer.self_attn.q_proj, layer.self_attn.k_proj,
            layer.self_attn.v_proj, layer.self_attn.o_proj,
            layer.mlp.gate_proj, layer.mlp.up_proj, layer.mlp.down_proj,
        ):
            proj.weight.requires_grad_(True)


@torch.enable_grad()
def compute_importance(
    model,
    adapter,
    input_ids,
    metric="taylor",
    loss_fn=clm_loss,
    head_dim=None,
    logger=None,
):
    """Compute per-structure importance for `model` over the calibration `input_ids`.

    input_ids: LongTensor [num_samples, seq_len] (already tokenised).
    Returns an ImportanceResult with head/channel scores per layer.
    """
    if metric not in _METRICS:
        raise ValueError(f"Unknown metric {metric!r}; available: {list_metrics()}")
    needs_backward, per_sample, contrib_fn = _METRICS[metric]

    layers = adapter.get_text_layers(model)
    cfg = adapter.get_text_config(model)
    device = next(model.parameters()).device
    input_ids = input_ids.to(device)

    def _log(msg):
        if logger is not None:
            logger.log(msg)

    model.eval()
    if needs_backward:
        _enable_grads(layers)

    dims = [layer_dims(layer, cfg, head_dim) for layer in layers]

    if not per_sample:
        if needs_backward:
            model.zero_grad(set_to_none=True)
            loss = loss_fn(model, input_ids)
            _log(f"[importance/{metric}] loss = {float(loss):.4f}")
            loss.backward()
        with torch.no_grad():
            head = {i: _head_scores(layers[i], dims[i], contrib_fn).cpu() for i in range(len(layers))}
            chan = {i: _channel_scores(layers[i], contrib_fn).cpu() for i in range(len(layers))}
    else:
        # per-sample accumulation (Fisher): reduce on the fly to avoid storing grads
        head = {i: torch.zeros(dims[i].num_heads) for i in range(len(layers))}
        chan = {i: torch.zeros(dims[i].intermediate) for i in range(len(layers))}
        n = input_ids.shape[0]
        for s in range(n):
            model.zero_grad(set_to_none=True)
            loss = loss_fn(model, input_ids[s : s + 1])
            loss.backward()
            with torch.no_grad():
                for i in range(len(layers)):
                    head[i] += _head_scores(layers[i], dims[i], contrib_fn).cpu()
                    chan[i] += _channel_scores(layers[i], contrib_fn).cpu()
            if logger is not None and (s + 1) % max(1, n // 4) == 0:
                _log(f"[importance/{metric}] sample {s + 1}/{n}")
        head = {i: v / n for i, v in head.items()}
        chan = {i: v / n for i, v in chan.items()}

    model.zero_grad(set_to_none=True)
    meta = {
        "num_layers": len(layers),
        "head_dim": dims[0].head_dim if dims else None,
        "num_heads": [d.num_heads for d in dims],
        "intermediate": [d.intermediate for d in dims],
        "model_class": model.__class__.__name__,
        "num_samples": int(input_ids.shape[0]),
    }
    return ImportanceResult(metric, head, chan, meta)
