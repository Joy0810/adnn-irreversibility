"""Train / validation / test splits by scenario, never by row.

Each environment's scenarios are shuffled and split separately (20% test, then
20% of the rest for validation), so both environments appear in every part and
every minimal pair stays on one side.

Usage:
  python -m probes.splits --rows data/sandbox_pairs.jsonl data/sciworld_pairs.jsonl
writes data/splits/seed0.json, seed1.json, seed2.json ({"train": [ids], "val": [...], "test": [...]}).
"""
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path


def make_split(rows: list[dict], seed: int, test_frac: float = 0.2, val_frac: float = 0.2) -> dict:
    scenarios = defaultdict(set)
    for row in rows:
        scenarios[row["environment"]].add(row["scenario_id"])
    rng = random.Random(seed)
    part_of = {}
    for env in sorted(scenarios):
        ids = sorted(scenarios[env])
        rng.shuffle(ids)
        n_test = max(1, round(len(ids) * test_frac))
        n_val = max(1, round((len(ids) - n_test) * val_frac))
        for i, scenario in enumerate(ids):
            part_of[scenario] = "test" if i < n_test else "val" if i < n_test + n_val else "train"
    split = {"train": [], "val": [], "test": []}
    for row in rows:
        split[part_of[row["scenario_id"]]].append(row["id"])
    return split


def main():
    parser = argparse.ArgumentParser(description="Write scenario-level splits.")
    parser.add_argument("--rows", nargs="+", required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--out", type=Path, default=Path("data/splits"))
    args = parser.parse_args()

    from models.extract import load_rows
    rows = load_rows(args.rows)
    args.out.mkdir(parents=True, exist_ok=True)
    for seed in args.seeds:
        split = make_split(rows, seed)
        (args.out / f"seed{seed}.json").write_text(json.dumps(split))
        print(f"seed {seed}: " + ", ".join(f"{k} {len(v)}" for k, v in split.items()))


if __name__ == "__main__":
    main()
