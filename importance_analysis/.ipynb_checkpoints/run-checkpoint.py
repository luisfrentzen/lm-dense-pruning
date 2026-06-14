"""Compute + compare structure importance for an unpruned/pruned model pair.

Example:
    python -m importance_analysis.run \
        --unpruned /mnt/Model-Weights/meta-llama/Llama-3.1-8B-Instruct \
        --pruned   out/llama_P0.1bookcorpus \
        --model_type llama \
        --metric taylor \
        --dataset bookcorpus --num_examples 32 --seq_len 64 \
        --out analysis_out/llama_P0.1_taylor

If the prune run wrote a structure-map sidecar, pass --mapping <map.json> to use
the exact recorded mapping instead of reconstructing it from the weights.
"""

from __future__ import annotations

import argparse
import csv
import json
import os

import torch

from LLMPruner.datasets.example_samples import get_examples

from .structures import resolve_adapter
from .importance import compute_importance, list_metrics
from .mapping import (
    build_mapping_by_weights,
    fingerprint_structures,
    load_mapping,
    resolve_kept_indices,
    save_mapping,
)
from .compare import compare_importance


def _load_model(path, adapter, dtype, device):
    model, tokenizer = adapter.load_base_model_and_tokenizer(path, torch_dtype=dtype)
    model.to(device)
    model.config.use_cache = False
    adapter.set_special_tokens(model, tokenizer)
    return model, tokenizer


def main():
    ap = argparse.ArgumentParser(description="Structure-level importance comparison.")
    ap.add_argument("--unpruned", required=True)
    ap.add_argument("--pruned", required=True)
    ap.add_argument("--model_type", required=True, choices=["llama", "gemma3"])
    ap.add_argument("--metric", default="taylor", choices=list_metrics())
    ap.add_argument("--dataset", default="bookcorpus")
    ap.add_argument("--num_examples", type=int, default=32)
    ap.add_argument("--seq_len", type=int, default=64)
    ap.add_argument("--head_dim", type=int, default=None, help="override if config lacks head_dim")
    ap.add_argument("--mapping", default=None, help="recorded structure-map sidecar (json)")
    ap.add_argument("--mobility", action="store_true", help="rank-mobility analysis + plots")
    ap.add_argument("--out", required=True, help="output directory")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    adapter = resolve_adapter(args.model_type)

    # ---- unpruned: importance (+ fingerprints for mapping) ----
    print(f"[1/3] unpruned importance ({args.metric}) from {args.unpruned}")
    full_model, tokenizer = _load_model(args.unpruned, adapter, dtype, device)
    calib = get_examples(args.dataset, tokenizer, args.num_examples, seq_len=args.seq_len)
    imp_full = compute_importance(full_model, adapter, calib, metric=args.metric, head_dim=args.head_dim)
    fingerprints = None if args.mapping else fingerprint_structures(full_model, adapter, args.head_dim)
    imp_full.save(os.path.join(args.out, "importance_unpruned.pt"))
    del full_model
    if device == "cuda":
        torch.cuda.empty_cache()

    # ---- pruned: importance ----
    print(f"[2/3] pruned importance ({args.metric}) from {args.pruned}")
    pruned_model, _ = _load_model(args.pruned, adapter, dtype, device)
    imp_pruned = compute_importance(pruned_model, adapter, calib, metric=args.metric, head_dim=args.head_dim)
    imp_pruned.save(os.path.join(args.out, "importance_pruned.pt"))

    # ---- mapping ----
    if args.mapping:
        print(f"[3/3] using recorded mapping {args.mapping}")
        mapping = load_mapping(args.mapping)
    else:
        print("[3/3] reconstructing mapping by exact weight match (pre-recovery)")
        mapping = resolve_kept_indices(pruned_model, adapter, fingerprints)
        mapping.meta["source"] = "weight_match"
    save_mapping(mapping, os.path.join(args.out, "structure_map.json"))
    del pruned_model
    if device == "cuda":
        torch.cuda.empty_cache()

    # ---- compare + write report ----
    result = compare_importance(imp_full, imp_pruned, mapping)
    with open(os.path.join(args.out, "report.json"), "w") as f:
        json.dump({"metric": args.metric, "summary": result["summary"], "rows": result["rows"]}, f, indent=2)
    _write_structures_csv(os.path.join(args.out, "structures.csv"), imp_full, imp_pruned, mapping)

    if args.mobility:
        from .mobility import compute_mobility
        mob = compute_mobility(imp_full, imp_pruned, mapping)
        with open(os.path.join(args.out, "mobility.json"), "w") as f:
            json.dump({k: {"table": v["table"], "transition_frac": v["transition_frac"]} for k, v in mob.items()}, f, indent=2)
        try:
            from .plot import plot_mobility
            for kind in ("head", "channel"):
                plot_mobility(mob[kind], args.out, kind)
        except Exception as e:
            print("plotting skipped:", e)

    print("\n=== summary (metric: {}) ===".format(args.metric))
    for kind in ("head", "channel", "overall"):
        s = result["summary"][kind]
        print(
            f"{kind:8s} keep_auc={s['keep_auc']:.3f}  "
            f"spearman_retained={s['spearman_retained']:.3f}  "
            f"topk_survival={s['topk_survival']:.3f}"
        )
    print(f"\nwrote: {args.out}/report.json, structures.csv, structure_map.json, importance_*.pt")


def _write_structures_csv(path, full, pruned, mapping):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["kind", "layer", "orig_index", "survived", "score_unpruned", "score_pruned"])
        for kind in ("head", "channel"):
            for layer in sorted(full.scores(kind)):
                orig = full.scores(kind)[layer]
                kept = mapping.kept(kind, layer)
                pruned_scores = pruned.scores(kind)[layer]
                pos_of_orig = {o: p for p, o in enumerate(kept)}
                for o in range(len(orig)):
                    survived = o in pos_of_orig
                    sp = float(pruned_scores[pos_of_orig[o]]) if survived else ""
                    w.writerow([kind, layer, o, int(survived), float(orig[o]), sp])


if __name__ == "__main__":
    main()
