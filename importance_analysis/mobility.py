import torch

QUINTILES = 5


def _pct_and_band(scores, nbands):
    s = scores.float()
    n = len(s)
    if n == 1:
        pct = torch.zeros(1)
    else:
        ranks = torch.empty(n)
        ranks[torch.argsort(s)] = torch.arange(n, dtype=torch.float)
        pct = ranks / (n - 1)
    band = (pct * nbands).long().clamp_max(nbands - 1)
    return pct, band


def compute_mobility(full, pruned, mapping, nbands=QUINTILES):
    out = {}
    for kind in ("head", "channel"):
        recs = []
        for layer in sorted(full.scores(kind)):
            kept = mapping.kept(kind, layer)
            if len(kept) < 2:
                continue
            orig = full.scores(kind)[layer].float()[torch.tensor(kept, dtype=torch.long)]
            post = pruned.scores(kind)[layer].float()
            if len(post) != len(kept):
                raise ValueError(
                    f"{kind} layer {layer}: {len(kept)} survivors vs {len(post)} pruned scores"
                )
            bpct, bband = _pct_and_band(orig, nbands)
            apct, aband = _pct_and_band(post, nbands)
            for i in range(len(kept)):
                recs.append((layer, kept[i], float(bpct[i]), float(apct[i]), int(bband[i]), int(aband[i])))
        out[kind] = _summarize(recs, nbands)
    return out


def _summarize(recs, nbands):
    trans = torch.zeros(nbands, nbands)
    bands = {b: [] for b in range(nbands)}
    for (_, _, bp, ap, bb, ab) in recs:
        trans[bb, ab] += 1
        bands[bb].append((ap - bp, bb == ab))

    table = []
    for b in range(nbands):
        items = bands[b]
        n = len(items)
        if n == 0:
            table.append({"band": b, "n": 0, "mean_move": float("nan"),
                          "mean_abs_move": float("nan"), "std_move": float("nan"), "retention": float("nan")})
            continue
        moves = torch.tensor([m for m, _ in items])
        table.append({
            "band": b, "n": n,
            "mean_move": float(moves.mean()),
            "mean_abs_move": float(moves.abs().mean()),
            "std_move": float(moves.std(unbiased=False)),
            "retention": sum(1 for _, r in items if r) / n,
        })

    trans_frac = trans / trans.sum(1, keepdim=True).clamp_min(1)
    return {"records": recs, "table": table,
            "transition": trans.tolist(), "transition_frac": trans_frac.tolist(), "nbands": nbands}
