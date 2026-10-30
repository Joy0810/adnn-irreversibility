"""Probes and statistics on small synthetic data (seconds on a CPU)."""
import numpy as np
import pytest

from analysis.stats import blindness_gap, bootstrap, mcnemar_exact, proportion_ci
from probes.splits import make_split
from probes.train import control_labels, layer_job, task_data, text_baselines


def synthetic(n_scenarios=60, per=6, layers=3, width=16, seed=0):
    """Rows plus hidden states where only layer 2 carries the label."""
    rng = np.random.default_rng(seed)
    rows = []
    labels = ["reversible", "partial", "irreversible"]
    for s in range(n_scenarios):
        env = "sandbox" if s % 2 == 0 else "scienceworld"
        for j in range(per):
            rows.append({"id": f"{env}-{s}-{j}", "scenario_id": f"{env}-{s}", "environment": env,
                         "source": "minimal_pair" if env == "sandbox" and j < 3 else "random",
                         "mechanism": ["delete_file", "send_email", "move_file"][j % 3],
                         "label": labels[(s + j) % 3], "action_text": "same action", "state_text": f"state {s} {j}"})
    hidden = rng.standard_normal((len(rows), layers, width)).astype(np.float32)
    for i, r in enumerate(rows):
        hidden[i, 2, labels.index(r["label"])] += 4.0
    return rows, hidden


def test_split_keeps_scenarios_whole():
    rows, _ = synthetic()
    split = make_split(rows, seed=0)
    part = {i: name for name, ids in split.items() for i in ids}
    by_scenario = {}
    for r in rows:
        by_scenario.setdefault(r["scenario_id"], set()).add(part[r["id"]])
    assert all(len(parts) == 1 for parts in by_scenario.values())
    for name in ("train", "val", "test"):
        assert {r["environment"] for r in rows if part[r["id"]] == name} == {"sandbox", "scienceworld"}


@pytest.fixture(scope="module")
def setup(tmp_path_factory):
    rows, hidden = synthetic()
    path = tmp_path_factory.mktemp("hidden") / "last.npy"
    np.save(path, hidden)
    split_ids = make_split(rows, seed=0)
    part = {i: name for name, ids in split_ids.items() for i in ids}
    return rows, path, part


def run_layer(setup, layer, classes):
    rows, path, part = setup
    keep, kept_rows, y, split = task_data(rows, part, classes)
    y_control = control_labels(kept_rows, y, split["train"], len(classes), seed=0)
    return layer_job(path, keep, layer, kept_rows, y, y_control, split, len(classes), 0, ["sandbox", "scienceworld"])


def test_probe_finds_the_planted_layer(setup):
    for classes in (["reversible", "irreversible"], ["reversible", "partial", "irreversible"]):
        def acc(layer, method="probe"):
            return next(r["balanced_acc"] for r in run_layer(setup, layer, classes)
                        if r["method"] == method and r["subset"] == "all" and r["train_env"] == "all")
        assert acc(2) > 0.9
        assert acc(0) < 0.75
        assert acc(2, "shuffled") < 0.75


def test_transfer_and_subsets_are_reported(setup):
    results = run_layer(setup, 2, ["reversible", "irreversible"])
    assert {r["train_env"] for r in results} == {"all", "sandbox", "scienceworld"}
    assert {"all", "minimal_pair", "sandbox", "scienceworld"} <= {r["subset"] for r in results}


def test_text_baselines_cannot_read_identical_actions(setup):
    rows, _, part = setup
    keep, kept_rows, y, split = task_data(rows, part, ["reversible", "irreversible"])
    results = text_baselines(kept_rows, y, split, 2)
    bow = next(r for r in results if r["method"] == "bow_action" and r["subset"] == "all")
    assert bow["balanced_acc"] < 0.7  # every action text is identical


def test_mcnemar_and_gap():
    test = mcnemar_exact([1, 1, 1, 0], [0, 0, 0, 0])
    assert test["only_a"] == 3 and test["only_b"] == 0 and abs(test["p_value"] - 0.25) < 1e-9
    gap = blindness_gap([1, 1, 1, 1], [1, 0, 0, 0], [0, 0, 0, 0], n_boot=200)
    assert gap["K"]["value"] == 1 and gap["A"]["value"] == 0.25 and gap["gap"]["value"] == 0.75


def test_bootstrap_ci_brackets_the_estimate():
    x = np.r_[np.ones(30), np.zeros(70)]
    result = proportion_ci(x, n_boot=500)
    assert result["ci_low"] <= 0.3 <= result["ci_high"]
    grouped = bootstrap(lambda idx: x[idx].mean(), len(x), groups=np.arange(100) // 10, n_boot=200)
    assert grouped["n_groups"] == 10
