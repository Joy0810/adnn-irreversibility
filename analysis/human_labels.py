"""Hand-label quality check (brief 4.2): two people label the same 100 random pairs blind.

  python -m analysis.human_labels sample --rows data/sandbox_pairs.jsonl data/sciworld_pairs.jsonl
      writes data/human_labels/sheet_A.csv and sheet_B.csv (same rows, no automatic label).
      Each labeller fills the `label` column with reversible / partial / irreversible.
  python -m analysis.human_labels score --rows data/sandbox_pairs.jsonl data/sciworld_pairs.jsonl
      prints Cohen's kappa A vs B, A vs automatic, B vs automatic; writes results/tables/human_agreement.json
"""
from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

from analysis.stats import kappa
from models.extract import load_rows

DIR = Path("data/human_labels")
LABELS = ("reversible", "partial", "irreversible")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("cmd", choices=["sample", "score"])
    parser.add_argument("--rows", nargs="+", required=True)
    parser.add_argument("--n", type=int, default=100)
    args = parser.parse_args()
    rows = {r["id"]: r for r in load_rows(args.rows)}

    if args.cmd == "sample":
        picked = random.Random(0).sample(sorted(rows), args.n)
        DIR.mkdir(parents=True, exist_ok=True)
        for who in "AB":
            with open(DIR / f"sheet_{who}.csv", "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["id", "task_text", "state_text", "action_text", "label"])
                for i in picked:
                    r = rows[i]
                    w.writerow([i, r["task_text"], r["state_text"], r["action_text"], ""])
        print(f"wrote {args.n} rows to {DIR}/sheet_A.csv and sheet_B.csv")
        return

    sheets = {}
    for who in "AB":
        with open(DIR / f"sheet_{who}.csv", encoding="utf-8") as f:
            sheets[who] = {r["id"]: r["label"].strip().lower() for r in csv.DictReader(f)}
    ids = [i for i in sheets["A"] if sheets["A"][i] in LABELS and sheets["B"].get(i) in LABELS]
    if len(ids) < len(sheets["A"]):
        print(f"warning: {len(sheets['A']) - len(ids)} rows missing or with an unknown label are left out")
    a, b = [sheets["A"][i] for i in ids], [sheets["B"][i] for i in ids]
    auto = [rows[i]["label"] for i in ids]
    out = {"n": len(ids), "kappa_A_B": kappa(a, b), "kappa_A_auto": kappa(a, auto), "kappa_B_auto": kappa(b, auto),
           "agree_A_B": sum(x == y for x, y in zip(a, b)) / len(ids)}
    Path("results/tables").mkdir(parents=True, exist_ok=True)
    Path("results/tables/human_agreement.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
