"""Build the report's figures and tables from results/.

  python -m analysis.report --models qwen7b llama8b --rows data/sandbox_pairs.jsonl data/sciworld_pairs.jsonl

Writes results/figures/*.png and results/tables/*.md. A part whose inputs do not
exist yet is skipped with a message, so this can be run at any stage.

  fig1_probe_layers.png            balanced accuracy vs layer, with every baseline (required)
  fig2_stated_probe_behaviour.png  stated vs probed vs behaviour (required)
  fig3_gap.png                     the blindness gap with 95% CIs (required)
  fig4_steering.png                steering curves, probe vs random directions (required for the bonus)
  tables/probes.md                 best-layer scores, controls and selectivity
  tables/ablations.md              position, minimal pairs, transfer, horizon, label set (required)
  tables/behaviour.md              RQ3 metrics per model, prompt and version, with McNemar tests
  tables/gap.md, tables/stated.md  the numbers behind fig3 and fig2
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score

from analysis.stats import blindness_gap, kappa, mcnemar_exact, metric_ci, proportion_ci

RESULTS = Path("results")
CHANCE = {"binary": 0.5, "three": 1 / 3}


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def markdown(df: pd.DataFrame) -> str:
    def cell(v):
        return f"{v:.3f}" if isinstance(v, (float, np.floating)) else str(v)
    lines = ["| " + " | ".join(map(str, df.columns)) + " |", "|" + " --- |" * len(df.columns)]
    lines += ["| " + " | ".join(cell(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join(lines) + "\n"


def pm(series) -> str:
    series = series.dropna()
    if series.empty:
        return "n/a"
    return f"{series.mean():.3f} ± {series.std(ddof=0):.3f}" if len(series) > 1 else f"{series.mean():.3f}"


def ci(d: dict) -> str:
    return f"{d['value']:.3f} [{d['ci_low']:.3f}, {d['ci_high']:.3f}]"


def load_probe_results(model: str, suffix: str = ""):
    path = RESULTS / "probes" / f"{model}{suffix}" / "results.csv"
    return pd.read_csv(path) if path.exists() else None


def best_layer(model: str) -> int | None:
    path = RESULTS / "probes" / model / "best_layer.json"
    return json.loads(path.read_text())["layer"] if path.exists() else None


# --- RQ1 ------------------------------------------------------------------------


def fig1_probe_layers(models, figures):
    import matplotlib.pyplot as plt
    tables = {m: t for m in models if (t := load_probe_results(m)) is not None}
    if not tables:
        return print("fig1: no probe results yet")
    fig, axes = plt.subplots(len(tables), 2, figsize=(11, 3.8 * len(tables)), squeeze=False, sharey=True)
    for r, (model, t) in enumerate(tables.items()):
        for c, subset in enumerate(["all", "minimal_pair"]):
            ax = axes[r][c]
            base = t[(t.labels == "binary") & (t.train_env == "all") & (t.subset == subset)]
            curves = [("last", "probe", "-", "probe (last token)"), ("mean", "probe", "-", "probe (mean over action)"),
                      ("last", "control", "--", "control task"), ("last", "shuffled", ":", "shuffled labels")]
            for position, method, style, label in curves:
                d = base[(base.position == position) & (base.method == method)].groupby("layer").balanced_acc
                mean, std = d.mean(), d.std(ddof=0).fillna(0)
                if mean.empty:
                    continue
                ax.plot(mean.index, mean.values, style, label=label, alpha=1 if position == "last" else 0.6)
                ax.fill_between(mean.index, mean - std, mean + std, alpha=0.15)
            for method, color in [("bow_action", "tab:gray"), ("bow_state_action", "black"), ("majority", "tab:brown")]:
                value = base[base.method == method].balanced_acc.mean()
                if not np.isnan(value):
                    ax.axhline(value, color=color, linewidth=1, linestyle="-.", label=method.replace("_", " "))
            if (layer := best_layer(model)) is not None:
                ax.axvline(layer, color="tab:red", linewidth=0.8, alpha=0.5)
            ax.set_title(f"{model}: {'all test rows' if subset == 'all' else 'minimal pairs only'} (binary)")
            ax.set_xlabel("layer (0 = embeddings)")
            ax.set_ylabel("balanced accuracy")
    axes[0][0].legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    fig.savefig(figures / "fig1_probe_layers.png", dpi=200)
    print("fig1: written")


def table_probes(models, tables_dir, rows_by_id):
    out = []
    for model in models:
        t, layer = load_probe_results(model), best_layer(model)
        if t is None or layer is None:
            continue
        preds = read_jsonl(RESULTS / "probes" / model / "predictions_seed0.jsonl")
        for labels in ("binary", "three"):
            for subset in sorted(set(t.subset)):
                sel = t[(t.labels == labels) & (t.subset == subset) & (t.train_env == "all")]
                at = sel[(sel.layer == layer) & (sel.position == "last")]
                probe = at[at.method == "probe"].balanced_acc
                control = at[at.method == "control"].balanced_acc
                row = {"model": model, "labels": labels, "subset": subset, "layer": layer,
                       "n_test": int(at[at.method == "probe"].n.mean()) if len(probe) else 0,
                       "probe": pm(probe), "control": pm(control),
                       "selectivity": pm(probe.reset_index(drop=True) - control.reset_index(drop=True)),
                       "shuffled": pm(at[at.method == "shuffled"].balanced_acc),
                       "layer 0": pm(sel[(sel.layer == 0) & (sel.position == "last") & (sel.method == "probe")].balanced_acc)}
                for method in ("bow_action", "bow_state_action", "majority"):
                    row[method] = pm(sel[sel.method == method].balanced_acc)
                kept = [p for p in preds if f"pred_{labels}" in p and _in_subset(rows_by_id.get(p["id"]), subset)]
                if kept and rows_by_id:
                    row["probe 95% CI (seed 0)"] = ci(metric_ci(
                        balanced_accuracy_score, [p["label"] for p in kept], [p[f"pred_{labels}"] for p in kept],
                        groups=[rows_by_id[p["id"]]["scenario_id"] for p in kept]))
                out.append(row)
    if not out:
        return print("tables/probes: no probe results yet")
    (tables_dir / "probes.md").write_text(
        "Balanced accuracy at the pre-registered best layer, mean ± sd over split seeds. "
        "Selectivity = probe - control task.\n\n" + markdown(pd.DataFrame(out)))
    print("tables/probes: written")


def _in_subset(row, subset):
    if row is None:
        return False
    return subset == "all" or (subset == "minimal_pair" and row["source"] == "minimal_pair") or row["environment"] == subset


def table_ablations(models, tables_dir):
    out = []
    for model in models:
        t, layer = load_probe_results(model), best_layer(model)
        if t is None or layer is None:
            continue

        def score(df, **where):
            for key, value in where.items():
                df = df[df[key] == value]
            return pm(df.balanced_acc)

        probe = t[(t.method == "probe") & (t.layer == layer)]
        base = dict(labels="binary", position="last", train_env="all")
        out += [
            {"model": model, "ablation": "token position", "setting": "last token", "balanced acc": score(probe, **base, subset="all")},
            {"model": model, "ablation": "token position", "setting": "mean over action", "balanced acc": score(probe, **{**base, "position": "mean"}, subset="all")},
            {"model": model, "ablation": "test rows", "setting": "all", "balanced acc": score(probe, **base, subset="all")},
            {"model": model, "ablation": "test rows", "setting": "minimal pairs only", "balanced acc": score(probe, **base, subset="minimal_pair")},
            {"model": model, "ablation": "label set", "setting": "binary (chance 0.5)", "balanced acc": score(probe, **base, subset="all")},
            {"model": model, "ablation": "label set", "setting": "three-class (chance 0.33)", "balanced acc": score(probe, **{**base, "labels": "three"}, subset="all")},
        ]
        envs = sorted(set(t.train_env) - {"all"})
        for env in envs:
            other = [e for e in envs if e != env]
            for target in other:
                out.append({"model": model, "ablation": "cross-environment", "setting": f"train {env} -> test {target}",
                            "balanced acc": score(probe, labels="binary", position="last", train_env=env, subset=target)})
                out.append({"model": model, "ablation": "cross-environment", "setting": f"train both -> test {target}",
                            "balanced acc": score(probe, **base, subset=target)})
        for h in ("label_h1", "label_h2"):
            th = load_probe_results(model, f"_{h}")
            if th is not None:
                ph = th[(th.method == "probe") & (th.layer == layer)]
                out.append({"model": model, "ablation": "horizon", "setting": f"horizon {h[-1]}",
                            "balanced acc": score(ph, **base, subset="all")})
        out.append({"model": model, "ablation": "horizon", "setting": "horizon 3", "balanced acc": score(probe, **base, subset="all")})
    if not out:
        return print("tables/ablations: no probe results yet")
    (tables_dir / "ablations.md").write_text(
        "Probe balanced accuracy at the best layer (binary unless stated), mean ± sd over split seeds. "
        "The careful-prompt ablation is in behaviour.md.\n\n" + markdown(pd.DataFrame(out)))
    print("tables/ablations: written")


# --- RQ2 and RQ3 --------------------------------------------------------------------


def behaviour_scores(model: str, condition: str) -> pd.DataFrame | None:
    rows = read_jsonl(RESULTS / "agent" / model / condition / "scores.jsonl")
    return pd.DataFrame(rows) if rows else None


def paired(scores: pd.DataFrame, key: str):
    wide = scores.pivot(index="task_id", columns="version", values=key).dropna()
    return wide["irreversible"].astype(bool).values, wide["reversible"].astype(bool).values, list(wide.index)


def table_behaviour(models, tables_dir):
    out = []
    for model in models:
        for condition in ("neutral", "careful"):
            s = behaviour_scores(model, condition)
            if s is None:
                continue
            for version in ("reversible", "irreversible"):
                v = s[s.version == version]
                out.append({"model": model, "prompt": condition, "version": version, "n": len(v),
                            **{key: ci(proportion_ci(v[key])) for key in
                               ("shortcut_taken", "caution", "strict_caution", "irreversible_taken", "success")},
                            "parse failures / step": f"{v.parse_failures.sum() / max(1, v.steps.sum()):.3f}"})
            irr, rev, _ = paired(s, "caution")
            test = mcnemar_exact(irr, rev)
            out.append({"model": model, "prompt": condition, "version": "sensitivity (irr - rev)", "n": len(irr),
                        "caution": f"{irr.mean() - rev.mean():+.3f} (McNemar p = {test['p_value']:.3g})",
                        "strict_caution": f"{np.mean(paired(s, 'strict_caution')[0]) - np.mean(paired(s, 'strict_caution')[1]):+.3f}"})
    if not out:
        return print("tables/behaviour: no agent runs yet")
    (tables_dir / "behaviour.md").write_text(
        "Rates over task episodes with bootstrap 95% CIs. Over-caution = caution in the reversible version.\n\n"
        + markdown(pd.DataFrame(out).fillna("")))
    print("tables/behaviour: written")


def gap_numbers(model: str, condition: str = "neutral"):
    pairs = read_jsonl(RESULTS / "stated" / model / "pairs.jsonl")
    scores = behaviour_scores(model, condition)
    if not pairs or scores is None:
        return None
    knows = {}
    for p in pairs:
        knows.setdefault(p["task_id"], []).append(p["correct"])
    irr, rev, tasks = paired(scores, "caution")
    k = [len(knows.get(t, [])) == 2 and all(knows[t]) for t in tasks]
    return blindness_gap(k, irr, rev)


def fig3_gap(models, figures, tables_dir):
    import matplotlib.pyplot as plt
    entries = [(m, c, g) for m in models for c in ("neutral", "careful") if (g := gap_numbers(m, c))]
    if not entries:
        return print("fig3: no stated pairs and agent runs yet")
    fig, ax = plt.subplots(figsize=(1.8 + 1.6 * len(entries), 3.6))
    width = 0.25
    for j, name in enumerate(("K", "A", "gap")):
        values = [g[name]["value"] for _, _, g in entries]
        errors = [[g[name]["value"] - g[name]["ci_low"] for _, _, g in entries],
                  [g[name]["ci_high"] - g[name]["value"] for _, _, g in entries]]
        ax.bar(np.arange(len(entries)) + (j - 1) * width, values, width, yerr=errors, capsize=3,
               label={"K": "K: states which is irreversible", "A": "A: acts more carefully there", "gap": "gap = K - A"}[name])
    ax.set_xticks(np.arange(len(entries)), [f"{m}\n{c} prompt" for m, c, _ in entries])
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("share of task pairs")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(figures / "fig3_gap.png", dpi=200)
    rows = [{"model": m, "prompt": c, **{k: ci(g[k]) for k in ("K", "A", "gap")}, "pairs": g["K"]["n"]} for m, c, g in entries]
    (tables_dir / "gap.md").write_text("Blindness gap over task pairs, paired bootstrap 95% CIs.\n\n" + markdown(pd.DataFrame(rows)))
    print("fig3: written")


def fig2_stated(models, figures, tables_dir, rows_by_id):
    import matplotlib.pyplot as plt
    out = []
    for model in models:
        answers = {a["id"]: a for a in read_jsonl(RESULTS / "stated" / model / "answers.jsonl")}
        preds = [p for p in read_jsonl(RESULTS / "probes" / model / "predictions_seed0.jsonl") if "pred_three" in p]
        both = [p for p in preds if p["id"] in answers]
        if not both:
            continue
        truth = [p["label"] for p in both]
        stated = [answers[p["id"]]["stated"] or "unparsed" for p in both]
        stated_lp = [answers[p["id"]]["stated_logprob"] for p in both]
        probe = [p["pred_three"] for p in both]
        groups = [rows_by_id[p["id"]]["scenario_id"] for p in both] if rows_by_id else None
        parsed = [i for i, s in enumerate(stated) if s != "unparsed"]
        probe_right = np.array(probe) == np.array(truth)
        stated_right = np.array(stated) == np.array(truth)
        out.append({
            "model": model, "n (probe test rows, seed 0)": len(both),
            "probe balanced acc": ci(metric_ci(balanced_accuracy_score, truth, probe, groups)),
            "stated balanced acc": ci(metric_ci(balanced_accuracy_score, truth, stated, groups)),
            "stated (log-prob) balanced acc": ci(metric_ci(balanced_accuracy_score, truth, stated_lp, groups)),
            "unparsed answers": len(both) - len(parsed),
            "kappa stated vs probe": kappa([stated[i] for i in parsed], [probe[i] for i in parsed]) if parsed else float("nan"),
            "both right": float(np.mean(probe_right & stated_right)), "probe only": float(np.mean(probe_right & ~stated_right)),
            "stated only": float(np.mean(~probe_right & stated_right)), "both wrong": float(np.mean(~probe_right & ~stated_right)),
            "A (acts, neutral)": (g := gap_numbers(model)) and g["A"]["value"], "K (states, pairs)": g and g["K"]["value"],
        })
    if not out:
        return print("fig2: needs stated answers and probe predictions")
    df = pd.DataFrame(out)
    (tables_dir / "stated.md").write_text(
        "Three-class answers on the probe's seed-0 test rows. Balanced accuracy with scenario-bootstrap 95% CIs.\n\n" + markdown(df))
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    parts = ["both right", "probe only", "stated only", "both wrong"]
    bottom = np.zeros(len(df))
    for part in parts:
        axes[0].bar(df.model, df[part], bottom=bottom, label=part)
        bottom += df[part].values
    axes[0].set_title("Per-row agreement with the true label")
    axes[0].set_ylabel("share of test rows")
    axes[0].legend(fontsize=7)
    x = np.arange(len(df))
    axes[1].bar(x - 0.2, df["K (states, pairs)"].fillna(0), 0.4, label="K: states it")
    axes[1].bar(x + 0.2, df["A (acts, neutral)"].fillna(0), 0.4, label="A: acts on it")
    axes[1].set_xticks(x, df.model)
    axes[1].set_title("Task pairs: knowing vs acting")
    axes[1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(figures / "fig2_stated_probe_behaviour.png", dpi=200)
    print("fig2: written")


# --- RQ4 ------------------------------------------------------------------------


def fig4_steering(models, figures, tables_dir):
    import matplotlib.pyplot as plt
    found = [(m, pd.DataFrame(d)) for m in models if (d := read_jsonl(RESULTS / "steering" / m / "decision.jsonl"))]
    if not found:
        return print("fig4: no steering runs yet")
    fig, axes = plt.subplots(len(found), 2, figsize=(10, 3.6 * len(found)), squeeze=False)
    summary = []
    for r, (model, d) in enumerate(found):
        zero = d[d.alpha == 0].groupby(["task_id", "version"]).margin.mean()
        d = d.join(zero.rename("margin0"), on=["task_id", "version"])
        d["shift"] = d.margin - d.margin0
        ax = axes[r][0]
        for name, group in d.groupby("direction"):
            g = group.groupby("alpha")["shift"].agg(["mean", "sem"])
            ax.errorbar(g.index, g["mean"], yerr=1.96 * g["sem"], marker="o", capsize=2,
                        label=name, alpha=1 if name == "probe" else 0.5)
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_title(f"{model}: first decision")
        ax.set_xlabel("steering strength alpha")
        ax.set_ylabel("change in log P(ask) - log P(shortcut)")
        ax.legend(fontsize=7)
        ax = axes[r][1]
        for run in sorted((RESULTS / "steering" / model / "rollouts").glob("*_alpha*")):
            name, alpha = run.name.rsplit("_alpha", 1)
            s = read_jsonl(run / "scores.jsonl")
            irr = [x for x in s if x["version"] == "irreversible"]
            if irr:
                summary.append({"model": model, "direction": name, "alpha": float(alpha), "n": len(irr),
                                "caution (irreversible version)": np.mean([x["caution"] for x in irr]),
                                "success": np.mean([x["success"] for x in irr]),
                                "parse failures": sum(x["parse_failures"] for x in irr)})
        rollouts = pd.DataFrame([x for x in summary if x["model"] == model])
        if not rollouts.empty:
            for name, group in rollouts.sort_values("alpha").groupby("direction"):
                ax.plot(group.alpha, group["caution (irreversible version)"], marker="o", label=f"{name}: caution")
                ax.plot(group.alpha, group.success, marker="x", linestyle="--", label=f"{name}: success")
            ax.set_title(f"{model}: full episodes")
            ax.set_xlabel("steering strength alpha")
            ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(figures / "fig4_steering.png", dpi=200)
    if summary:
        (tables_dir / "steering.md").write_text(markdown(pd.DataFrame(summary)))
    print("fig4: written")


def main():
    parser = argparse.ArgumentParser(description="Figures and tables for the report.")
    parser.add_argument("--models", nargs="+", default=["qwen7b", "llama8b"])
    parser.add_argument("--rows", nargs="*", default=[], help="labelled data files, for scenario-level CIs")
    args = parser.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    from models.extract import load_rows
    rows_by_id = {r["id"]: r for r in load_rows(args.rows)} if args.rows else {}
    figures, tables_dir = RESULTS / "figures", RESULTS / "tables"
    figures.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    fig1_probe_layers(args.models, figures)
    table_probes(args.models, tables_dir, rows_by_id)
    table_ablations(args.models, tables_dir)
    table_behaviour(args.models, tables_dir)
    fig2_stated(args.models, figures, tables_dir, rows_by_id)
    fig3_gap(args.models, figures, tables_dir)
    fig4_steering(args.models, figures, tables_dir)


if __name__ == "__main__":
    main()
