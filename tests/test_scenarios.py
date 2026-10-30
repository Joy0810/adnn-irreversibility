import json
import random
from collections import defaultdict

import pytest

from sandbox import TOOLS, render_state
from sandbox.scenarios import generate

REQUIRED = ("id", "environment", "state_text", "action_text", "label", "horizon", "source")


@pytest.fixture(scope="module")
def rows():
    # generate() raises if any minimal-pair variant gets a label other than its designed one.
    return generate(n_worlds=8, seed=123, random_per_world=6)


def test_required_fields_and_unique_ids(rows):
    for row in rows:
        for key in REQUIRED:
            assert row[key] not in (None, ""), key
        assert row["label"] in ("reversible", "partial", "irreversible")
        assert json.loads(row["action_text"]) == row["action"]
    assert len({r["id"] for r in rows}) == len(rows)


def test_no_read_only_actions(rows):
    assert not any(TOOLS[r["action"]["tool"]].read_only for r in rows)


def test_minimal_pairs_share_action_and_task_but_not_state(rows):
    groups = defaultdict(list)
    for row in rows:
        if row["pair_id"]:
            groups[row["pair_id"]].append(row)
    assert groups
    for pair_id, group in groups.items():
        assert len(group) == 3, pair_id
        assert len({r["action_text"] for r in group}) == 1
        assert len({r["task_text"] for r in group}) == 1
        assert len({r["cue_style"] for r in group}) == 1
        assert len({r["state_text"] for r in group}) == 3, pair_id
        assert sorted(r["label"] for r in group) == ["irreversible", "partial", "reversible"]
        assert len({r["scenario_id"] for r in group}) == 1


def test_minimal_pair_texts_differ_in_few_lines(rows):
    """Variants should differ only where the flipped feature is described."""
    groups = defaultdict(list)
    for row in rows:
        if row["pair_id"]:
            groups[row["pair_id"]].append(row["state_text"].split("\n"))
    for pair_id, texts in groups.items():
        for other in texts[1:]:
            changed = len(set(texts[0]) ^ set(other))
            assert changed <= 8, (pair_id, changed)


def test_generation_is_deterministic(rows):
    again = generate(n_worlds=8, seed=123, random_per_world=6)
    assert [r["state_text"] for r in again] == [r["state_text"] for r in rows]
    assert [r["label"] for r in again] == [r["label"] for r in rows]


def test_every_cue_style_renders(rows):
    state = rows[0]["state"]
    texts = {style: render_state(state, style, random.Random(0)) for style in ("config", "structural", "narrative")}
    assert len(set(texts.values())) == 3
    assert "[settings]" in texts["config"]
    assert texts["narrative"].startswith("Setup notes:")
