"""Structure-level importance analysis for block-wise pruned models.

Answers: after block-wise pruning, do important structures (attention heads /
MLP channels) stay important? It computes a per-structure importance score for a
model, maps the surviving structures of a pruned model back to their location in
the unpruned model, and compares the two.

Public API:
    compute_importance(model, adapter, batch, metric=...)  -> ImportanceResult
    build_mapping_by_weights(unpruned, pruned, adapter)    -> StructureMapping
    fingerprint_structures / resolve_kept_indices          -> record at prune time
    compare_importance(full, pruned, mapping)              -> comparison metrics
"""

from .importance import ImportanceResult, compute_importance, list_metrics
from .mapping import (
    StructureMapping,
    build_mapping_by_weights,
    fingerprint_structures,
    resolve_kept_indices,
    save_mapping,
    load_mapping,
)
from .compare import compare_importance
from .mobility import compute_mobility

__all__ = [
    "ImportanceResult",
    "compute_importance",
    "list_metrics",
    "StructureMapping",
    "build_mapping_by_weights",
    "fingerprint_structures",
    "resolve_kept_indices",
    "save_mapping",
    "load_mapping",
    "compare_importance",
    "compute_mobility",
]
