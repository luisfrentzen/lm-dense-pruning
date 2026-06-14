import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

_LABELS = {5: ["0-20", "20-40", "40-60", "60-80", "80-100"]}


def _labels(n):
    return _LABELS.get(n, [str(i) for i in range(n)])


def plot_mobility(m, outdir, kind):
    _scatter(m, os.path.join(outdir, f"mobility_scatter_{kind}.png"), kind)
    _transition(m, os.path.join(outdir, f"mobility_transition_{kind}.png"), kind)
    _bands(m, os.path.join(outdir, f"mobility_bands_{kind}.png"), kind)


def _scatter(m, path, kind):
    recs = m["records"]
    if not recs:
        return
    bp = [r[2] for r in recs]
    ap = [r[3] for r in recs]
    layer = [r[0] for r in recs]
    fig, ax = plt.subplots(figsize=(5, 5))
    sc = ax.scatter(bp, ap, c=layer, cmap="viridis", s=6, alpha=0.5)
    ax.plot([0, 1], [0, 1], "r--", lw=1)
    ax.set_xlabel("before percentile")
    ax.set_ylabel("after percentile")
    ax.set_title(f"{kind} rank mobility")
    fig.colorbar(sc, label="layer")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def _transition(m, path, kind):
    n = m["nbands"]
    T = torch.tensor(m["transition_frac"])
    labels = _labels(n)
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(T, cmap="Blues", vmin=0, vmax=1, origin="lower")
    ax.set_xticks(range(n)); ax.set_xticklabels(labels)
    ax.set_yticks(range(n)); ax.set_yticklabels(labels)
    ax.set_xlabel("after band")
    ax.set_ylabel("before band")
    ax.set_title(f"{kind} band transitions")
    for i in range(n):
        for j in range(n):
            ax.text(j, i, f"{T[i, j]:.2f}", ha="center", va="center", fontsize=7)
    fig.colorbar(im)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def _bands(m, path, kind):
    tbl = m["table"]
    n = m["nbands"]
    labels = _labels(n)
    x = [t["band"] for t in tbl]
    absmove = [t["mean_abs_move"] for t in tbl]
    ret = [t["retention"] for t in tbl]
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar([i - 0.2 for i in x], absmove, width=0.4, label="mean |move|")
    ax.bar([i + 0.2 for i in x], ret, width=0.4, label="retention")
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_xlabel("before band")
    ax.set_title(f"{kind} per-band movement")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
