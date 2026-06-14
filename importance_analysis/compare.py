"""Compare importance of an unpruned model against its pruned child.

Given importance for the unpruned model, importance for the pruned model, and the
structure mapping (surviving pruned position -> original index), this answers two
distinct questions per structure kind (head / channel), per layer and overall:

  1. "Did pruning keep the important structures?"
     keep_auc = P(original score of a survivor > original score of a pruned-away one).
     0.5 = pruning ignored this importance signal; 1.0 = it removed exactly the
     lowest-scoring structures. (Mann-Whitney U statistic.)
     topk_survival = fraction of the original top-k that survived.

  2. "Among survivors, did important stay important after pruning?"
     spearman / pearson between each survivor's ORIGINAL score and its score
     recomputed on the PRUNED model.

All scores are compared within a layer (importance scales vary by depth); global
figures are sample-weighted means across layers.
"""

from __future__ import annotations

import torch


def _ranks(x):
    # average ranks (ties shared), 1..n
    order = torch.argsort(x)
    ranks = torch.empty_like(x)
    ranks[order] = torch.arange(1, len(x) + 1, dtype=x.dtype)
    return ranks


def _pearson(x, y):
    if len(x) < 2:
        return float("nan")
    x = x - x.mean()
    y = y - y.mean()
    denom = x.norm() * y.norm()
    if denom == 0:
        return float("nan")
    return float((x @ y) / denom)


def _spearman(x, y):
    if len(x) < 2:
        return float("nan")
    return _pearson(_ranks(x), _ranks(y))


def _keep_auc(orig_scores, kept_idx):
    # P(survivor score > pruned-away score); equivalent to ROC-AUC of "survived"
    n = len(orig_scores)
    kept = torch.zeros(n, dtype=torch.bool)
    kept[torch.tensor(kept_idx, dtype=torch.long)] = True
    pruned = ~kept
    n1, n0 = int(kept.sum()), int(pruned.sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = _ranks(orig_scores.float())
    auc = (r[kept].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)
    return float(auc)


def _topk_survival(orig_scores, kept_idx, k):
    if k <= 0:
        return float("nan")
    top = set(torch.argsort(orig_scores, descending=True)[:k].tolist())
    return len(top & set(kept_idx)) / float(k)


def compare_importance(full, pruned, mapping, topk_frac=None):
    """Return per-(kind,layer) rows plus aggregate metrics.

    full / pruned : ImportanceResult for the unpruned and pruned models.
    mapping       : StructureMapping (pruned position -> original index).
    topk_frac     : if set, top-k uses k = round(topk_frac * num_survivors).
    """
    rows = []
    per_kind = {"head": [], "channel": []}

    for kind in ("head", "channel"):
        full_s = full.scores(kind)
        pruned_s = pruned.scores(kind)
        for layer in sorted(full_s):
            orig = full_s[layer].float()
            surv_pruned = pruned_s[layer].float()
            kept_idx = mapping.kept(kind, layer)

            if len(kept_idx) != len(surv_pruned):
                raise ValueError(
                    f"{kind} layer {layer}: mapping has {len(kept_idx)} survivors but "
                    f"pruned importance has {len(surv_pruned)}. Mapping/model mismatch."
                )

            orig_of_survivors = orig[torch.tensor(kept_idx, dtype=torch.long)]
            k = round((topk_frac or len(kept_idx) / max(1, len(orig))) * len(kept_idx))

            row = {
                "kind": kind,
                "layer": layer,
                "n_total": len(orig),
                "n_kept": len(kept_idx),
                "keep_ratio": len(kept_idx) / float(len(orig)),
                "keep_auc": _keep_auc(orig, kept_idx),
                "spearman_retained": _spearman(orig_of_survivors, surv_pruned),
                "pearson_retained": _pearson(orig_of_survivors, surv_pruned),
                "topk_survival": _topk_survival(orig, kept_idx, k),
            }
            rows.append(row)
            per_kind[kind].append(row)

    def _agg(rows_subset, key):
        vals, wts = [], []
        for r in rows_subset:
            v = r[key]
            if v == v:  # not nan
                vals.append(v)
                wts.append(r["n_kept"])
        if not vals:
            return float("nan")
        vals = torch.tensor(vals)
        wts = torch.tensor(wts, dtype=torch.float)
        return float((vals * wts).sum() / wts.sum())

    summary = {}
    for kind in ("head", "channel"):
        summary[kind] = {
            key: _agg(per_kind[kind], key)
            for key in ("keep_auc", "spearman_retained", "pearson_retained", "topk_survival")
        }
    summary["overall"] = {
        key: _agg(rows, key)
        for key in ("keep_auc", "spearman_retained", "pearson_retained", "topk_survival")
    }
    return {"rows": rows, "summary": summary}
