# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compute a static expert placement (VLLM_HPU_MOE_EXPERT_PERM_FILE) from per-expert MoE stats.

The stats file holds JSON lines. The first line is a header:

    {"ep_size": E, "local_num_experts": L, "layers": ["model.layers.1.mlp.experts", ...]}

(other keys are ignored), with the MoE layer names in the order of the "layer" field below. Every further line is
one record per MoE layer and token-count bucket:

    {"layer": <position in header "layers">, "bucket": "<label, e.g. 17-64>", "forwards": <MoE forwards>,
     "rows": [...], "active": [...], "chunks": [...]}

where each list has E * L values indexed by global expert id, summed over the forwards of the bucket: routed token
rows, forwards with at least one row and expert chunks of the fused MoE op (only the --feature list is read).
Buckets without forwards may be omitted. Stats of a run that already used a permutation are indexed by physical slot
(pass --applied-perm).

The experts of every layer are assigned to the EP ranks with LPT (heaviest expert first, onto the least loaded rank
that still has a free slot) and the permutation file is written. Rank load is the sum of the chosen feature over the
fit buckets.

Reports, per evaluation set, the rank imbalance before (current placement) and after: mean over layers of
(max - mean) / mean and the sum over layers of the max rank load relative to the sum of the mean loads, next to
the floor set by the heaviest expert of each layer.

    python tools/moe_expert_perm.py --stats moe_expert_stats.log --fit-buckets 1-16,17-64 --out perm.json \
        --eval 1-16 --eval 17-64 --eval other.log:17-64
"""

import argparse
import json
import sys


def read_stats(path):
    """(header, {layer index in file: {bucket: record}})."""
    header, records = None, {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if header is None:
                header = rec
                continue
            records.setdefault(rec["layer"], {})[rec["bucket"]] = rec
    if header is None:
        raise ValueError(f"{path}: empty stats file")
    return header, records


def decoder_layer_index(name):
    parts = name.split(".")
    for a, b in zip(parts, parts[1:]):
        if a == "layers" and b.isdigit():
            return int(b)
    raise ValueError(f"no decoder layer index in {name!r}")


def load_weights(path, buckets, feature, applied=None):
    """{decoder layer index: per logical expert load summed over buckets}, header."""
    header, records = read_stats(path)
    num_experts = header["ep_size"] * header["local_num_experts"]
    out = {}
    for li, name in enumerate(header["layers"]):
        index = decoder_layer_index(name)
        w = [0.0] * num_experts
        for b in buckets:
            rec = records.get(li, {}).get(b)
            if rec is None:
                continue
            for e, v in enumerate(rec[feature]):
                w[e] += v
        if applied is not None:
            # stats of a run with a permutation are indexed by physical slot
            perm = applied[index]
            w = [w[perm[e]] for e in range(num_experts)]
        out[index] = w
    return out, header


def lpt(weights, ep_size, local):
    """perm[e] = physical slot of expert e."""
    loads, members = [0.0] * ep_size, [[] for _ in range(ep_size)]
    for e in sorted(range(len(weights)), key=lambda e: (-weights[e], e)):
        r = min((r for r in range(ep_size) if len(members[r]) < local), key=lambda r: (loads[r], r))
        members[r].append(e)
        loads[r] += weights[e]
    perm = [0] * len(weights)
    for r, experts in enumerate(members):
        for slot, e in enumerate(sorted(experts)):
            perm[e] = r * local + slot
    return perm


def rank_loads(weights, perm, ep_size, local):
    loads = [0.0] * ep_size
    for e, w in enumerate(weights):
        loads[perm[e] // local] += w
    return loads


def imbalance(weights_by_layer, perms, ep_size, local):
    """(mean over layers of (max - mean) / mean, sum of max / sum of mean, per-layer (max - mean) / mean)."""
    per_layer, total_max, total_mean = {}, 0.0, 0.0
    for index, w in weights_by_layer.items():
        loads = rank_loads(w, perms[index], ep_size, local)
        mean = sum(loads) / ep_size
        if mean <= 0:
            continue
        per_layer[index] = (max(loads) - mean) / mean
        total_max += max(loads)
        total_mean += mean
    avg = sum(per_layer.values()) / max(1, len(per_layer))
    return avg, (total_max / total_mean if total_mean else 0.0), per_layer


def placement_floor(weights_by_layer, ep_size):
    """Sum of max / sum of mean that no placement beats: a rank holds at least the heaviest expert."""
    total_floor, total_mean = 0.0, 0.0
    for w in weights_by_layer.values():
        mean = sum(w) / ep_size
        total_floor += max(max(w), mean)
        total_mean += mean
    return total_floor / total_mean if total_mean else 0.0


def parse_eval(spec, default_stats):
    path, _, buckets = spec.rpartition(":")
    return (path or default_stats), [b for b in buckets.split(",") if b]


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--stats", required=True, help="moe_expert_stats.log to fit on")
    p.add_argument("--fit-buckets", required=True, help="comma-separated bucket labels, e.g. 1-16,17-64")
    p.add_argument("--feature", default="chunks", choices=("chunks", "rows", "active"))
    p.add_argument("--applied-perm", help="permutation file active while the stats were collected")
    p.add_argument("--eval",
                   action="append",
                   default=[],
                   help="[stats file:]buckets to evaluate on (repeatable; default: the fit set)")
    p.add_argument("--out", help="permutation file to write")
    p.add_argument("--per-layer", action="store_true", help="print the per-layer imbalance")
    args = p.parse_args(argv)

    applied = None
    if args.applied_perm:
        with open(args.applied_perm) as f:
            applied = {int(k): v for k, v in json.load(f)["layers"].items()}
    fit_buckets = [b for b in args.fit_buckets.split(",") if b]
    fit, header = load_weights(args.stats, fit_buckets, args.feature, applied)
    ep_size, local = header["ep_size"], header["local_num_experts"]
    num_experts = ep_size * local
    if not any(any(w) for w in fit.values()):
        sys.exit(f"no {args.feature} in buckets {fit_buckets} of {args.stats} (buckets: {header.get('buckets')})")

    perms = {index: lpt(w, ep_size, local) for index, w in fit.items()}
    current = applied if applied is not None else {index: list(range(num_experts)) for index in fit}

    evals = [parse_eval(s, args.stats) for s in args.eval] or [(args.stats, fit_buckets)]
    print(f"{len(perms)} layers, {ep_size} ranks x {local} experts, feature {args.feature}, "
          f"fit {args.stats}:{','.join(fit_buckets)}")
    print(f"{'eval set':48} {'(max-mean)/mean before':>23} {'after':>7} {'sum max/sum mean before':>24} {'after':>7} "
          f"{'floor':>7}")
    for path, buckets in evals:
        w, _ = load_weights(path, buckets, args.feature, applied if path == args.stats else None)
        b_avg, b_tot, b_layer = imbalance(w, current, ep_size, local)
        a_avg, a_tot, a_layer = imbalance(w, perms, ep_size, local)
        label = f"{path}:{','.join(buckets)}"
        print(f"{label[-48:]:48} {b_avg:23.3f} {a_avg:7.3f} {b_tot:24.3f} {a_tot:7.3f} "
              f"{placement_floor(w, ep_size):7.3f}")
        if args.per_layer:
            for index in sorted(b_layer):
                print(f"  layer {index:3d} {b_layer[index]:7.3f} -> {a_layer.get(index, 0.0):7.3f}")

    if args.out:
        data = {
            "num_experts": num_experts,
            "ep_size": ep_size,
            "fit": {
                "buckets": fit_buckets,
                "feature": args.feature
            },
            "layers": {
                str(index): perms[index]
                for index in sorted(perms)
            },
        }
        with open(args.out, "w") as f:
            json.dump(data, f, separators=(",", ":"))
            f.write("\n")
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
