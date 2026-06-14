# importance_analysis

Answer: **after block-wise pruning, do important structures stay important?**

It scores every prunable structure of a model (attention heads + MLP intermediate
channels — the same units the block-wise pruner removes), maps a pruned model's
surviving structures back to their location in the unpruned model, and compares
the two.

## What it computes

**Importance** (per attention head and per MLP channel, one score each):

| metric | formula | needs backward |
|---|---|---|
| `magnitude` | `Σ|w|` | no |
| `taylor` | `Σ|w · ∂L/∂w|` (the pruner's `param_first` criterion) | yes |
| `fisher` | `mean_samples Σ (∂L/∂w)²` (diagonal empirical Fisher) | yes (per-sample) |

The loss `L` is pluggable (`loss_fn(model, input_ids) -> scalar`, default causal-LM),
so importance can reflect any objective. Add a metric with
`importance.register_metric(name, contrib_fn, needs_backward, per_sample)`.

A structure's score aggregates its coupled weights: a head sums its `q_proj` rows,
`o_proj` columns, and (shared, GQA) `k/v` rows; an MLP channel sums its `gate_proj`
/`up_proj` rows and `down_proj` column.

**Mapping** (pruned position → original index): block-wise pruning removes whole
heads/channels by index and leaves the survivors byte-identical (pre-recovery), so
the mapping is exact. Two sources:

1. **Recorded at prune time** (robust, survives later recovery): set
   `record_structure_map: true` in `config/prune_config.yaml`. The prune run writes
   `structure_map.json` next to the checkpoint. Positions never move during recovery,
   so the same map stays valid after distillation.
2. **Reconstructed** from a pre-recovery `(unpruned, pruned)` pair by exact weight
   match — used automatically for existing checkpoints with no sidecar.

**Comparison** (per kind, per layer, and overall):

- `keep_auc` — P(original score of a survivor > original score of a pruned-away one).
  `0.5` = pruning ignored this importance signal; `1.0` = it removed exactly the
  lowest-scoring structures. *Did pruning keep the important ones?*
- `spearman_retained` / `pearson_retained` — correlation between each survivor's
  **original** score and its score recomputed on the **pruned** model.
  *Among survivors, did important stay important?*
- `topk_survival` — fraction of the original top-k that survived.

## Usage

One command does importance(unpruned) + importance(pruned) + mapping + compare:

```bash
python -m importance_analysis.run \
    --unpruned /mnt/Model-Weights/meta-llama/Llama-3.1-8B-Instruct \
    --pruned   out/llama_P0.1bookcorpus \
    --model_type llama \
    --metric taylor \
    --dataset bookcorpus --num_examples 32 --seq_len 64 \
    --out analysis_out/llama_P0.1_taylor
# add --mapping <structure_map.json> to use an exact recorded mapping
```

Outputs in `--out`:

- `report.json` — summary metrics + per-layer rows
- `structures.csv` — every structure: `kind, layer, orig_index, survived, score_unpruned, score_pruned`
- `structure_map.json` — the pruned→unpruned index mapping used
- `importance_unpruned.pt`, `importance_pruned.pt` — reusable `ImportanceResult`s

## Assumptions / scope

- Targets **block-wise** pruning: attention heads + MLP channels. The tool validates
  that the pruned model keeps the unpruned model's **hidden size** and **layer count**
  (it errors otherwise — that checkpoint was pruned in another dimension).
- Comparison is meant for **unpruned vs pre-recovery** (exact mapping). To analyze a
  **recovered** model, record the map at prune time (option 1) and pass `--mapping`.
- Models are loaded one at a time on GPU (bf16) to bound memory.
