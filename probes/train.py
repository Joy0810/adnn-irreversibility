"""E2: per-layer linear probes with every compulsory control (RQ1).

For each split seed, token position (last, mean) and label set (binary, three-class),
and for every layer including layer 0 (the embeddings):

- probe: StandardScaler + L2 logistic regression (balanced class weights). C is
  chosen on the validation split, then the probe is refit on train + validation
  and scored on the test split;
- shuffled: the same probe trained on permuted labels (should sit at chance);
- control: the Hewitt-Liang control task, a random fixed label per action verb
  (`mechanism`); selectivity = probe minus control balanced accuracy;
- transfer: trained on one environment only, tested on the other's test rows.

Once per seed and label set: majority class, bag-of-words on the action text, and
bag-of-words on state plus action text.

Every test score is reported on the whole test split and on subsets (minimal
pairs, each environment). Output in results/probes/<model>/:
  results.csv                one row per (seed, labels, position, method, layer, train_env, subset)
  best_layer.json            pre-registered rule: seed 0, last token, binary, best validation score
  predictions_seed<k>.jsonl  test predictions at the best layer (for RQ2)
  direction.npy, direction.json   mean(irreversible) - mean(reversible) at the best layer (for RQ4)

Usage:
  python -m probes.train --model qwen7b --rows data/sandbox_pairs.jsonl data/sciworld_pairs.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

LABEL_SETS = {"binary": ["reversible", "irreversible"], "three": ["reversible", "partial", "irreversible"]}
CS = (0.001, 0.01, 0.1, 1.0)
POSITIONS = ("last", "mean")


def probe(C: float):
    return make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=3000, class_weight="balanced"))


def bow(C: float):
    return make_pipeline(CountVectorizer(token_pattern=r"[A-Za-z_]+|\d+", binary=True),
                         LogisticRegression(C=C, max_iter=3000, class_weight="balanced"))


def scores(y, pred, proba, k: int) -> dict:
    out = {"n": len(y), "balanced_acc": balanced_accuracy_score(y, pred),
           "macro_f1": f1_score(y, pred, average="macro", labels=list(range(k)), zero_division=0)}
    try:
        if k == 2:
            out["auroc"] = roc_auc_score(y, proba[:, 1])
        else:
            out["auroc"] = roc_auc_score(y, proba, multi_class="ovr", average="macro", labels=list(range(k)))
    except ValueError:  # a class is missing from this subset
        out["auroc"] = float("nan")
    return out


def subsets(rows: list[dict], idx: np.ndarray) -> dict:
    """Named subsets of the test indices: all, minimal pairs, and each environment."""
    def where(test):
        return idx[np.array([test(rows[i]) for i in idx], dtype=bool)] if len(idx) else idx
    named = {"all": idx, "minimal_pair": where(lambda r: r["source"] == "minimal_pair")}
    for env in sorted({rows[i]["environment"] for i in idx}):
        named[env] = where(lambda r, env=env: r["environment"] == env)
    return {name: sub for name, sub in named.items() if len(sub)}


def choose_C(make, X_train, y_train, X_val, y_val) -> tuple[float, float]:
    """The C with the best validation balanced accuracy, and that accuracy."""
    best = None
    for C in CS:
        score = balanced_accuracy_score(y_val, make(C).fit(X_train, y_train).predict(X_val))
        if best is None or score > best[1]:
            best = (C, score)
    return best


def control_labels(rows, y, train_idx, k, seed) -> np.ndarray:
    """Hewitt-Liang control task: each verb gets a fixed random label drawn from the training label distribution."""
    rng = np.random.default_rng(seed)
    dist = np.bincount(y[train_idx], minlength=k) / len(train_idx)
    verbs = sorted({r["mechanism"] for r in rows})
    train_verbs = {rows[i]["mechanism"] for i in train_idx}
    for _ in range(100):  # with few verbs, all could draw one label; a probe needs two
        mapping = {verb: int(rng.choice(k, p=dist)) for verb in verbs}
        if len({mapping[v] for v in train_verbs}) > 1:
            break
    return np.array([mapping[r["mechanism"]] for r in rows])


def layer_job(hidden_path, keep, layer, rows, y, y_control, split, k, seed, transfer_envs) -> list[dict]:
    """Every result for one layer of one (seed, labels, position)."""
    X = np.asarray(np.load(hidden_path, mmap_mode="r")[:, layer, :], dtype=np.float32)[keep]
    train, val, test = split["train"], split["val"], split["test"]
    trval = np.concatenate([train, val])
    C, val_score = choose_C(probe, X[train], y[train], X[val], y[val])
    results = []

    def record(method, model, labels, test_idx=test, train_env="all"):
        for name, sub in subsets(rows, test_idx).items():
            results.append({"method": method, "layer": layer, "train_env": train_env, "subset": name,
                            "C": C, "val_balanced_acc": val_score,
                            **scores(labels[sub], model.predict(X[sub]), model.predict_proba(X[sub]), k)})

    record("probe", probe(C).fit(X[trval], y[trval]), y)
    shuffled = np.random.default_rng(seed * 1000 + layer).permutation(y[trval])
    record("shuffled", probe(C).fit(X[trval], shuffled), y)
    record("control", probe(C).fit(X[trval], y_control[trval]), y_control)
    for env in transfer_envs:
        env_trval = trval[np.array([rows[i]["environment"] == env for i in trval], dtype=bool)]
        other_test = test[np.array([rows[i]["environment"] != env for i in test], dtype=bool)]
        if len(other_test) and len(np.unique(y[env_trval])) == k:
            record("probe", probe(C).fit(X[env_trval], y[env_trval]), y, other_test, train_env=env)
    return results


def text_baselines(rows, y, split, k) -> list[dict]:
    train, val, test = split["train"], split["val"], split["test"]
    trval = np.concatenate([train, val])
    texts = {"bow_action": [r["action_text"] for r in rows],
             "bow_state_action": [r["state_text"] + "\n" + r["action_text"] for r in rows]}
    results = []
    for method, text in texts.items():
        text = np.array(text, dtype=object)
        C, val_score = choose_C(bow, text[train], y[train], text[val], y[val])
        model = bow(C).fit(text[trval], y[trval])
        for name, sub in subsets(rows, test).items():
            results.append({"method": method, "layer": -1, "train_env": "all", "subset": name, "C": C,
                            "val_balanced_acc": val_score,
                            **scores(y[sub], model.predict(text[sub]), model.predict_proba(text[sub]), k)})
    dummy = DummyClassifier(strategy="most_frequent").fit(np.zeros((len(trval), 1)), y[trval])
    for name, sub in subsets(rows, test).items():
        blank = np.zeros((len(sub), 1))
        results.append({"method": "majority", "layer": -1, "train_env": "all", "subset": name,
                        **scores(y[sub], dummy.predict(blank), dummy.predict_proba(blank), k)})
    return results


def task_data(all_rows, part, classes):
    """Rows with a label in `classes`, their array indices, labels and split indices."""
    keep = np.array([i for i, r in enumerate(all_rows) if r["label"] in classes])
    rows = [all_rows[i] for i in keep]
    y = np.array([classes.index(r["label"]) for r in rows])
    split = {name: np.array([j for j, r in enumerate(rows) if part.get(r["id"]) == name], dtype=int)
             for name in ("train", "val", "test")}
    return keep, rows, y, split


def read_split(splits_dir: Path, seed: int) -> dict:
    split_ids = json.loads((splits_dir / f"seed{seed}.json").read_text())
    return {row_id: name for name, members in split_ids.items() for row_id in members}


def choose_best_layer(table: pd.DataFrame) -> dict:
    sel = table[(table.seed == table.seed.min()) & (table.position == "last") & (table.labels == "binary")
                & (table.method == "probe") & (table.train_env == "all") & (table.subset == "all")]
    top = sel.sort_values(["val_balanced_acc", "layer"], ascending=[False, True]).iloc[0]
    return {"layer": int(top.layer), "val_balanced_acc": float(top.val_balanced_acc), "C": float(top.C),
            "seed": int(table.seed.min()),
            "rule": "first seed, last token, binary labels, highest validation balanced accuracy, ties to the earlier layer"}


def save_predictions_and_direction(all_rows, hidden, splits_dir, seeds, layer, out):
    X = np.asarray(np.load(hidden / "last.npy", mmap_mode="r")[:, layer, :], dtype=np.float32)
    for seed in seeds:
        part = read_split(splits_dir, seed)
        records = {}
        for labels_name, classes in LABEL_SETS.items():
            keep, rows, y, split = task_data(all_rows, part, classes)
            Xk = X[keep]
            C, _ = choose_C(probe, Xk[split["train"]], y[split["train"]], Xk[split["val"]], y[split["val"]])
            trval = np.concatenate([split["train"], split["val"]])
            model = probe(C).fit(Xk[trval], y[trval])
            test = split["test"]
            for j, pred, proba in zip(test, model.predict(Xk[test]), model.predict_proba(Xk[test])):
                record = records.setdefault(rows[j]["id"], {"id": rows[j]["id"], "label": rows[j]["label"]})
                record[f"pred_{labels_name}"] = classes[pred]
                record[f"proba_{labels_name}"] = [round(float(p), 4) for p in proba]
        with open(out / f"predictions_seed{seed}.jsonl", "w") as f:
            for record in records.values():
                f.write(json.dumps(record) + "\n")

    # The steering direction uses the training split of the first seed only.
    part = read_split(splits_dir, seeds[0])
    rev = [i for i, r in enumerate(all_rows) if part.get(r["id"]) == "train" and r["label"] == "reversible"]
    irr = [i for i, r in enumerate(all_rows) if part.get(r["id"]) == "train" and r["label"] == "irreversible"]
    direction = X[irr].mean(axis=0) - X[rev].mean(axis=0)
    np.save(out / "direction.npy", direction)
    (out / "direction.json").write_text(json.dumps({
        "layer": layer, "position": "last", "norm": float(np.linalg.norm(direction)),
        "mean_hidden_norm": float(np.linalg.norm(X[rev + irr], axis=1).mean()),
        "n_reversible": len(rev), "n_irreversible": len(irr), "split_seed": seeds[0]}, indent=2))


def main():
    parser = argparse.ArgumentParser(description="Per-layer probes and controls (E2).")
    parser.add_argument("--model", required=True)
    parser.add_argument("--rows", nargs="+", required=True)
    parser.add_argument("--hidden", type=Path, help="default results/hidden/<model>")
    parser.add_argument("--splits", type=Path, default=Path("data/splits"))
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--positions", nargs="+", default=list(POSITIONS))
    parser.add_argument("--label-field", default="label",
                        help="label_h1 or label_h2 for the horizon ablation (default: label, i.e. horizon 3)")
    parser.add_argument("--out", type=Path, help="default results/probes/<model>[_<label field>]")
    parser.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    args = parser.parse_args()

    from models.extract import load_rows
    hidden = args.hidden or Path("results/hidden") / args.model
    suffix = "" if args.label_field == "label" else f"_{args.label_field}"
    out = args.out or Path("results/probes") / f"{args.model}{suffix}"
    out.mkdir(parents=True, exist_ok=True)

    by_id = {r["id"]: {**r, "label": r[args.label_field]} for r in load_rows(args.rows)}
    all_rows = [by_id[i] for i in json.loads((hidden / "ids.json").read_text())]  # array order
    n_layers = json.loads((hidden / "meta.json").read_text())["n_layers"]
    envs = sorted({r["environment"] for r in all_rows})
    transfer_envs = envs if len(envs) > 1 else []

    results = []
    for seed in args.seeds:
        part = read_split(args.splits, seed)
        for labels_name, classes in LABEL_SETS.items():
            keep, rows, y, split = task_data(all_rows, part, classes)
            y_control = control_labels(rows, y, split["train"], len(classes), seed)
            tag = {"seed": seed, "labels": labels_name}
            results += [{**tag, "position": "text", **r} for r in text_baselines(rows, y, split, len(classes))]
            slim = [{k: r[k] for k in ("source", "environment", "mechanism")} for r in rows]  # cheap to send to workers
            for position in args.positions:
                per_layer = Parallel(n_jobs=args.jobs)(
                    delayed(layer_job)(hidden / f"{position}.npy", keep, layer, slim, y, y_control, split,
                                       len(classes), seed, transfer_envs)
                    for layer in range(n_layers + 1))
                results += [{**tag, "position": position, **r} for layer_results in per_layer for r in layer_results]
                print(f"seed {seed}, {labels_name}, {position}: {n_layers + 1} layers done", flush=True)
            pd.DataFrame(results).to_csv(out / "results.csv", index=False)  # checkpoint

    table = pd.DataFrame(results)
    table.to_csv(out / "results.csv", index=False)
    best = choose_best_layer(table)
    (out / "best_layer.json").write_text(json.dumps(best, indent=2))
    print("best layer:", best)
    save_predictions_and_direction(all_rows, hidden, args.splits, args.seeds, best["layer"], out)


if __name__ == "__main__":
    main()
