import copy
import json
import random
from collections import Counter

import pytest

from sciworld.env import _method, full_signature, tree_paths
from sciworld.generate import balance, label_state, mark_pairs
from sciworld.labeller import (DESTRUCTIVE, LABELS, META, Replayer, candidates, inverse_commands, label_action,
                               labellable, objects_of, placements, verb_of)
from sciworld.sample_states import choose_actions, choose_prefixes, plan_scenario

R, P, I = "reversible", "partial", "irreversible"


class FakeEnv:
    """A toy kitchen behind the same interface as sciworld.env.SciWorld, without Java.

    drawer: open and close it (reversible). cup of water: pouring it into the sink
    drains the water (irreversible). stove: while on, it heats itself and whatever sits
    on it, and nothing cools down (partial). coffee: cools one degree every tick, so
    the saved state is never reached again; only the idle counterfactual is.
    """
    MOVABLE = {"apple", "cup", "bowl", "pot", "sugar", "key"}
    CONTAINERS = {"table", "drawer", "stove", "sink", "cup", "bowl", "pot"}
    POUR_FROM = {"cup", "bowl", "pot"}
    POUR_INTO = {"cup", "bowl", "pot", "sink"}
    LIQUIDS = {"water"}
    ROOMS = ("hallway", "kitchen")
    FREE = ("look around", "inventory", "task")
    GOLD = ["open drawer", "pick up key", "close drawer", "activate stove", "move pot to stove"]

    def __init__(self, drift=True):
        self.drift = drift
        self.resets = 0
        self.load("toy-task", 0)

    def load(self, task, variation, simplification="easy"):
        self.task, self.variation, self.simplification = task, variation, simplification
        self.reset()

    def reset(self):
        self.resets += 1
        self.ticks = 0
        self.where = {"agent": "kitchen", "inventory": "agent", "table": "kitchen", "drawer": "kitchen",
                      "stove": "kitchen", "sink": "kitchen", "apple": "table", "cup": "table", "water": "cup",
                      "bowl": "table", "sugar": "bowl", "pot": "table", "coffee": "table", "key": "drawer"}
        self.is_open = {"drawer": False}
        self.on = {"stove": False}
        self.temp = {"stove": 20, "pot": 20, "coffee": 80}

    def _inside(self, obj, container):
        while obj in self.where:
            obj = self.where[obj]
            if obj == container:
                return True
        return False

    def _visible(self, obj):
        parent = self.where[obj]
        while parent not in self.ROOMS:
            if not self.is_open.get(parent, True):
                return False
            parent = self.where[parent]
        return parent == self.where["agent"]

    def _things(self):
        return sorted(o for o in self.where if o not in ("agent", "inventory") and self._visible(o))

    def valid_actions(self):
        things = self._things()
        actions = list(self.FREE) + ["wait", "wait1"]
        actions += [f"go to {room}" for room in self.ROOMS if room != self.where["agent"]]
        for obj in things:
            actions += [f"look at {obj}", f"focus on {obj}"]
            if obj in self.is_open:
                actions.append(("close " if self.is_open[obj] else "open ") + obj)
            if obj in self.on:
                actions.append(("deactivate " if self.on[obj] else "activate ") + obj)
            if obj in self.MOVABLE:
                actions.append(f"put down {obj}" if self.where[obj] == "inventory" else f"pick up {obj}")
                actions += [f"move {obj} to {c}" for c in things
                            if c in self.CONTAINERS and c not in (obj, self.where[obj])
                            and self.is_open.get(c, True) and not self._inside(c, obj)]
            if obj in self.POUR_FROM:
                actions += [f"pour {obj} into {c}" for c in things
                            if c in self.POUR_INTO and c != obj and not self._inside(c, obj)]
            if obj == "apple":
                actions.append("eat apple")
        return sorted(actions)

    def step(self, command):
        if command in self.FREE:
            return {"look around": self.look, "inventory": self.inventory, "task": self.task_description}[command]()
        if command not in self.valid_actions():
            return "No known action matches that input."
        verb = verb_of(command)
        if verb in ("open", "close"):
            self.is_open[command.split(" ", 1)[1]] = verb == "open"
        elif verb in ("activate", "deactivate"):
            self.on[command.split(" ", 1)[1]] = verb == "activate"
        elif verb == "pick up":
            self.where[command[len("pick up "):]] = "inventory"
        elif verb == "put down":
            self.where[command[len("put down "):]] = self.where["agent"]
        elif verb == "move":
            obj, dest = command[len("move "):].split(" to ")
            self.where[obj] = dest
        elif verb == "pour":
            src, dest = command[len("pour "):].split(" into ")
            for obj in [o for o, c in self.where.items() if c == src]:
                if dest == "sink" and obj in self.LIQUIDS:
                    del self.where[obj]  # down the drain
                else:
                    self.where[obj] = dest
        elif verb == "go to":
            self.where["agent"] = command[len("go to "):]
        elif verb == "eat":
            del self.where[command[len("eat "):]]
        for _ in range(10 if command == "wait" else 1):
            self.idle()
        return f"You {command}."

    def idle(self):
        self.ticks += 1
        if self.drift:
            self.temp["coffee"] = max(20, self.temp["coffee"] - 1)
        if self.on["stove"]:
            self.temp["stove"] += 10
            if self.where.get("pot") == "stove":
                self.temp["pot"] += 10

    def moves(self):
        return self.ticks

    def _path(self, obj):
        parts = [obj]
        while obj in self.where:
            obj = self.where[obj]
            parts.append(obj)
        return "/".join(reversed(parts))

    def signatures(self):
        full = json.dumps({"where": self.where, "open": self.is_open, "on": self.on, "temp": self.temp},
                          sort_keys=True)
        return full, tuple(sorted([self._path(o) for o in self.where] + list(self.ROOMS)))

    def look(self):
        here = self.where["agent"]
        things = [o for o in self._things() if self.where[o] == here]
        return f"This room is called the {here}. In it, you see:\n" + "\n".join(f"\t{o}" for o in things)

    def inventory(self):
        items = sorted(o for o, c in self.where.items() if c == "inventory")
        return "In your inventory, you see:\n" + ("\n".join(f"\t{o}" for o in items) or "\tnothing")

    def task_description(self):
        return "Your task is to heat the pot on the stove."

    def gold_actions(self):
        return list(self.GOLD)

    def train_variations(self):
        return [0, 1, 2]


class ShiftingFakeEnv(FakeEnv):
    """Each reset starts the coffee at a different temperature: replay is not deterministic."""

    def reset(self):
        super().reset()
        self.temp["coffee"] = 80 + self.resets


def labels_of(result):
    return [result.label(h) for h in (1, 2, 3)]


# --- commands -------------------------------------------------------------------


@pytest.mark.parametrize("command, verb, objects", [
    ("pick up apple", "pick up", ["apple"]),
    ("move apple to table", "move", ["apple", "table"]),
    ("pour cup containing water into sink", "pour", ["cup containing water", "sink"]),
    ("connect battery anode to red wire terminal 1", "connect", ["battery anode", "red wire terminal 1"]),
    ("deactivate stove", "deactivate", ["stove"]),
    ("disconnect red wire", "disconnect", ["red wire"]),
    ("open door to kitchen", "open", ["door to kitchen"]),
    ("teleport to kitchen", "teleport to", ["kitchen"]),
    ("wait1", "wait1", []),
    ("look at cup", "look at", ["cup"]),
])
def test_parse_commands(command, verb, objects):
    assert verb_of(command) == verb
    assert objects_of(command) == objects


def test_inverse_commands():
    assert inverse_commands("open drawer") == ["close drawer"]
    assert inverse_commands("deactivate stove") == ["activate stove"]
    assert inverse_commands("connect anode to wire") == ["disconnect anode", "disconnect wire"]
    assert inverse_commands("pour cup into bowl") == ["pour bowl into cup"]
    assert inverse_commands("eat apple") == []


def test_placements_put_displaced_objects_back():
    target = ("hallway", "kitchen", "kitchen/agent", "kitchen/agent/inventory", "kitchen/table", "kitchen/table/apple")
    core = ("hallway", "hallway/agent", "hallway/agent/inventory", "hallway/agent/inventory/apple", "kitchen",
            "kitchen/table")
    commands, displaced = placements(core, target)
    assert commands == ["go to kitchen", "teleport to kitchen", "move apple to table"]
    assert displaced == {"apple"}
    commands, _ = placements(("kitchen", "kitchen/agent", "kitchen/agent/inventory", "kitchen/agent/inventory/key"),
                             ("kitchen", "kitchen/agent", "kitchen/agent/inventory", "kitchen/key"))
    assert commands == ["move key to kitchen", "put down key"]


# --- labels -----------------------------------------------------------------------


@pytest.mark.parametrize("prefix, action, expected, restore", [
    ([], "open drawer", [R, R, R], ["close drawer"]),
    ([], "pick up apple", [R, R, R], ["move apple to table"]),  # put down would leave it on the floor
    ([], "go to hallway", [R, R, R], ["go to kitchen"]),
    (["open drawer"], "pick up key", [R, R, R], ["move key to drawer"]),
    ([], "pour cup into bowl", [I, R, R], ["pour bowl into cup", "move sugar to bowl"]),  # the sugar comes along
    ([], "activate stove", [P, P, P], []),  # things stay put at once, but the stove stays warm
    (["activate stove"], "move pot to stove", [P, P, P], ["move pot to table"]),  # back in place, still hot
    ([], "pour cup into sink", [I, I, I], []),
    ([], "eat apple", [I, I, I], []),
])
def test_mechanism_labels(prefix, action, expected, restore):
    result = label_action(Replayer(FakeEnv(), prefix), action)
    assert labels_of(result) == expected
    assert result.restore_sequence == restore
    assert not result.truncated


def test_restore_sequence_reaches_the_counterfactual():
    env = FakeEnv()
    rp = Replayer(env, [])
    result = label_action(rp, "pour cup into bowl")
    ticks, full, _ = rp.evaluate(["pour cup into bowl"] + result.restore_sequence)
    assert ticks == 3 and full == rp.baseline(3)[0]
    assert env.where["water"] == "cup" and env.where["sugar"] == "bowl"


def test_horizon_is_a_parameter():
    result = label_action(Replayer(FakeEnv(), []), "pour cup into bowl", max_h=1)
    assert result.labels == {1: I} and result.exact_depth is None
    with pytest.raises(ValueError):
        result.label(2)


def test_counterfactual_absorbs_drift():
    rp = Replayer(FakeEnv(drift=True), [])
    assert label_action(rp, "open drawer").exact_depth == 1
    # Against the saved state the same undo looks incomplete, because the coffee kept cooling.
    _, full, core = rp.evaluate(["open drawer", "close drawer"])
    assert full != rp.baseline(0)[0] and core == rp.baseline(0)[1]
    assert full == rp.baseline(2)[0]


def test_action_that_changes_nothing_is_reversible_at_depth_zero():
    for drift in (True, False):
        result = label_action(Replayer(FakeEnv(drift=drift), []), "pour pot into cup")  # the pot is empty
        assert result.exact_depth == 0 and result.labels == {1: R, 2: R, 3: R}


def test_labels_never_get_worse_with_horizon():
    order = {I: 0, P: 1, R: 2}
    env = FakeEnv()
    rp = Replayer(env, ["open drawer"])
    for action in choose_actions(env.valid_actions(), 6, random.Random(1)):
        found = labels_of(label_action(rp, action))
        assert [order[x] for x in found] == sorted(order[x] for x in found), (action, found)


def test_search_stops_at_the_node_budget():
    result = label_action(Replayer(FakeEnv(), []), "pour cup into sink", max_nodes=3)
    assert result.truncated and result.nodes == 3 and result.exact_depth is None


# --- replay -----------------------------------------------------------------------


def test_replayer_steps_forward_instead_of_resetting():
    env = FakeEnv()
    rp = Replayer(env, ["open drawer"])
    resets = env.resets
    rp.goto(["pick up key"])
    rp.goto(["pick up key", "close drawer"])
    assert env.resets == resets
    rp.goto(["pick up apple"])
    assert env.resets == resets + 1
    assert env.where["key"] == "drawer" and env.where["apple"] == "inventory"


def test_nondeterministic_replay_is_detected_and_skipped():
    assert Replayer(FakeEnv(), ["open drawer"]).check_determinism()
    env = ShiftingFakeEnv()
    assert not Replayer(env, ["open drawer"]).check_determinism()
    scenario = plan_scenario(env, "toy-task", 0)
    outcome = label_state(env, scenario, prefix_len=1)
    assert outcome["status"] == "nondeterministic" and outcome["rows"] == []


# --- sampling and meta actions ----------------------------------------------------


def test_meta_actions_are_never_labelled_or_tried():
    for command in ("look around", "look at cup", "look in drawer", "examine cup", "inventory", "task",
                    "read notebook", "focus on apple", "wait", "wait1", "reset task"):
        assert not labellable(command), command
    env = FakeEnv()
    valid = env.valid_actions()
    assert {"look around", "wait1", "focus on apple"} <= set(valid)  # the toy offers them
    chosen = choose_actions(valid, 100, random.Random(0))
    assert chosen and not any(verb_of(c) in META for c in chosen)

    rp = Replayer(env, [])
    target = rp.baseline(1)[1]
    _, _, core = rp.evaluate(["pick up apple"])
    tried = candidates(env.valid_actions(), ["pick up apple"], core, target, max_branch=100)
    assert tried[0] == "move apple to table"
    assert not any(verb_of(c) in META | DESTRUCTIVE for c in tried)  # "eat apple" is valid but never tried


def test_actions_are_stratified_by_verb():
    valid = FakeEnv().valid_actions()
    chosen = choose_actions(valid, 5, random.Random(0), prefer="open drawer")
    assert chosen[0] == "open drawer" and len(chosen) == 5
    assert len({verb_of(c) for c in chosen}) == 5
    assert choose_actions(valid, 3, random.Random(0), prefer="look at cup")[0] != "look at cup"


def test_prefixes_spread_over_the_gold_path():
    picked = choose_prefixes(20, 4, random.Random(0))
    assert len(picked) == 4 and picked == sorted(picked) and all(0 <= p < 20 for p in picked)
    assert [p // 5 for p in picked] == [0, 1, 2, 3]
    assert choose_prefixes(3, 5, random.Random(0)) == [0, 1, 2]


# --- rows -------------------------------------------------------------------------

REQUIRED = ("id", "environment", "state_text", "action_text", "label", "horizon", "source")


def test_rows_match_the_dataset_schema():
    env = FakeEnv()
    scenario = plan_scenario(env, "toy-task", 0)
    assert scenario["gold"] == FakeEnv.GOLD
    outcome = label_state(env, scenario, prefix_len=1, per_state=4, seed=0)
    rows = outcome["rows"]
    assert outcome["status"] == "ok" and rows
    for row in rows:
        for key in REQUIRED:
            assert row[key] not in (None, ""), key
        assert row["environment"] == "scienceworld" and row["source"] == "scienceworld"
        assert row["horizon"] == 3 and row["label"] == row["label_h3"]
        assert {row[f"label_h{h}"] for h in (1, 2, 3)} <= set(LABELS)
        assert row["scenario_id"] == "sw-toy-task-v0"
        assert row["action_text"] == row["action"] and row["mechanism"] == verb_of(row["action"])
        assert labellable(row["action"])
        assert row["pair_id"] is None and row["variant"] is None and row["cue_style"] is None
        assert row["task_text"] == env.task_description() and row["task_text"] in row["state_text"]
        assert "This room is called the kitchen" in row["state_text"]
        assert row["replay"] == {"task": "toy-task", "variation": 0, "simplification": "easy",
                                 "prefix": ["open drawer"]}
        assert all(isinstance(c, str) for c in row["restore_sequence"])
        assert row["search"]["exact_depth"] != 0  # no-ops are dropped
        json.dumps(row)
    assert len({r["id"] for r in rows}) == len(rows)
    first = rows[0]  # the next gold command goes first
    assert first["action"] == "pick up key" and first["label"] == R and first["restore_sequence"] == ["move key to drawer"]


def test_balance_caps_each_class():
    rows = [{"id": f"r{i}", "label": label} for i, label in enumerate([R] * 5 + [P] + [I] * 4)]
    out = balance(rows, target=6, seed=0)
    assert Counter(r["label"] for r in out) == {R: 2, P: 1, I: 2}
    assert [r["id"] for r in out] == sorted(r["id"] for r in out)


def test_pairs_are_same_action_different_label_within_a_scenario():
    def row(i, label, scenario, action):
        return {"id": i, "label": label, "scenario_id": scenario, "action_text": action, "pair_id": None}
    a, b = row("a", R, "s1", "open box"), row("b", I, "s1", "open box")
    c, d = row("c", I, "s2", "open box"), row("d", R, "s1", "close box")
    mark_pairs([a, b, c, d])
    assert a["pair_id"] and a["pair_id"] == b["pair_id"]
    assert c["pair_id"] is None and d["pair_id"] is None


# --- adapter helpers (no Java) ------------------------------------------------------

TREE = {"uuid": "0", "name": "universe", "type": "universe", "isDeleted": False, "propMaterial": None, "contents": {
    "kitchen-1": {"uuid": "1", "name": "kitchen", "type": "room", "isDeleted": False, "propMaterial": None, "contents": {
        "agent-2": {"uuid": "2", "name": "agent", "type": "agent", "isDeleted": False, "propMaterial": None,
                    "contents": {"inventory-3": {"uuid": "3", "name": "inventory", "type": "container",
                                                 "isDeleted": False, "propMaterial": None, "contents": {}}}},
        "cup-4": {"uuid": "4", "name": "cup", "type": "cup", "isDeleted": False,
                  "propMaterial": {"temperatureC": 24.98, "stateOfMatter": "solid"},
                  "propContainer": {"isContainer": "true", "isOpen": True, "isClosable": False}, "contents": {
                      "water-5": {"uuid": "5", "name": "water", "type": "substance", "isDeleted": False,
                                  "propMaterial": {"temperatureC": 25.0, "stateOfMatter": "liquid"}, "contents": {}}}},
        "apple-6": {"uuid": "6", "name": "apple", "type": "food", "isDeleted": True, "propMaterial": None,
                    "contents": {}}}}}}
LOOK = "This room is called the kitchen. In it, you see:\n\tthe agent\n\ta cup (containing water)"
INVENTORY = "In your inventory, you see:\n\tnothing"


def test_tree_paths_skip_deleted_objects():
    assert tree_paths(TREE) == ["kitchen", "kitchen/agent", "kitchen/agent/inventory", "kitchen/cup",
                                "kitchen/cup/water"]


def test_full_signature_ignores_uuids_order_and_float_noise_but_not_state():
    base = full_signature(TREE, LOOK, INVENTORY)
    noisy = copy.deepcopy(TREE)
    kitchen = noisy["contents"]["kitchen-1"]
    kitchen["contents"]["cup-4"]["propMaterial"]["temperatureC"] = 24.9800001
    kitchen["contents"]["cup-4"]["uuid"] = "99"
    kitchen["contents"] = dict(reversed(list(kitchen["contents"].items())))
    assert full_signature(noisy, LOOK, INVENTORY) == base

    boiled = copy.deepcopy(TREE)
    boiled["contents"]["kitchen-1"]["contents"]["cup-4"]["contents"]["water-5"]["propMaterial"]["stateOfMatter"] = "gas"
    assert full_signature(boiled, LOOK, INVENTORY) != base
    assert tree_paths(boiled) == tree_paths(TREE)  # the core view does not see it
    assert full_signature(TREE, LOOK + "\nYou also see:\n\tA door to the hallway (that is closed)", INVENTORY) != base


def test_method_lookup_falls_back_and_fails_loudly():
    class Old:
        def getNumMoves(self):
            return 7

    assert _method(Old(), "get_num_moves", "getNumMoves")() == 7
    with pytest.raises(AttributeError, match="get_num_moves"):
        _method(Old(), "get_num_moves", "num_moves")
