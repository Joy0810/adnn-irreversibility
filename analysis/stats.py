"""Statistics used across the report: bootstrap CIs, McNemar, kappa, and the blindness gap.

Bootstraps resample whole groups (scenarios for probes, task pairs for behaviour),
never single rows, because rows inside a group are not independent.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np
from scipy.stats import binomtest
from sklearn.metrics import cohen_kappa_score


def bootstrap(stat, n_items: int, groups=None, n_boot: int = 1000, seed: int = 0, alpha: float = 0.05) -> dict:
    """Point estimate and percentile CI of stat(indices), resampling groups with replacement."""
    if groups is None:
        groups = range(n_items)
    members = defaultdict(list)
    for i, g in enumerate(groups):
        members[g].append(i)
    blocks = [np.array(v) for v in members.values()]
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(n_boot):
        picked = rng.integers(len(blocks), size=len(blocks))
        draws.append(stat(np.concatenate([blocks[j] for j in picked])))
    low, high = np.nanpercentile(draws, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"value": float(stat(np.arange(n_items))), "ci_low": float(low), "ci_high": float(high),
            "n": n_items, "n_groups": len(blocks)}


def metric_ci(metric, y_true, y_pred, groups=None, **kwargs) -> dict:
    """CI for metric(y_true, y_pred), e.g. sklearn's balanced_accuracy_score."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    return bootstrap(lambda idx: metric(y_true[idx], y_pred[idx]), len(y_true), groups, **kwargs)


def proportion_ci(x, groups=None, **kwargs) -> dict:
    x = np.asarray(x, dtype=float)
    return bootstrap(lambda idx: x[idx].mean(), len(x), groups, **kwargs)


def mcnemar_exact(a, b) -> dict:
    """Exact McNemar test for paired booleans a and b (e.g. caution in the irreversible vs reversible version)."""
    a, b = np.asarray(a, dtype=bool), np.asarray(b, dtype=bool)
    only_a, only_b = int((a & ~b).sum()), int((~a & b).sum())
    p = binomtest(only_a, only_a + only_b, 0.5).pvalue if only_a + only_b else 1.0
    return {"only_a": only_a, "only_b": only_b, "p_value": float(p), "n_pairs": len(a)}


def kappa(a, b) -> float:
    return float(cohen_kappa_score(a, b))


def blindness_gap(knows, careful_irreversible, careful_reversible, n_boot: int = 1000, seed: int = 0) -> dict:
    """Gap = K - A over task pairs, with a paired bootstrap CI.

    knows[i]: the model named the irreversible version of pair i correctly (both orders).
    A counts pairs where the agent was cautious in the irreversible version and not in the reversible one.
    """
    knows = np.asarray(knows, dtype=float)
    acts = (np.asarray(careful_irreversible, dtype=bool) & ~np.asarray(careful_reversible, dtype=bool)).astype(float)
    n = len(knows)
    out = {name: bootstrap(lambda idx, v=v: v[idx].mean(), n, n_boot=n_boot, seed=seed)
           for name, v in (("K", knows), ("A", acts))}
    out["gap"] = bootstrap(lambda idx: knows[idx].mean() - acts[idx].mean(), n, n_boot=n_boot, seed=seed)
    return out
