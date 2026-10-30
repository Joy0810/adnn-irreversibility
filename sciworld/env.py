"""Adapter around ScienceWorldEnv. The only module that imports scienceworld, lazily.

What the API offers (scienceworld 1.2.3 on PyPI and 1.3.0 on GitHub main; see
docs/sciworld_spec.md for sources): ScienceWorldEnv(taskName, serverPath, envStepLimit)
starts a JVM through py4j. load(task, variation, simplificationStr, generateGoldPath)
builds a variation, reset() rebuilds it with the same arguments, and step(command)
returns (observation, reward, isCompleted, info). There is no save, restore or clone,
so labeller.Replayer rebuilds states by reset and replay. getObjectTree() dumps the
whole universe as JSON, except doors and plant life stages.

Choices made here:
- Fast stepping calls the Java server's step() directly. ScienceWorldEnv.step also
  builds an info dict (look, inventory and every valid action) that replay never reads.
- An ambiguous command is resolved by answering 0, the first option.
- "look around", "inventory" and "task" take no time. Every other parsed command
  takes one tick, except that wait1 seems to take two (the agent's wait counter
  runs one extra pass). load() measures wait1, and idle() advances exactly one tick
  with wait1 if that is what it takes, else with a "look at" command, which
  changes nothing.
- The full signature hashes the object tree without uuids (children sorted,
  temperatures rounded to TEMP_DECIMALS) plus the sorted lines of the look and
  inventory texts. These add door states and life stages for the current room.
- The core signature is the containment path of every object, e.g. "kitchen/table/cup".
- Method names differ between versions (snake_case since 1.2, camelCase before), so
  each call goes through _method(), which fails loudly if neither name exists.

Check an installation first with: python -m sciworld.env --task boil
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time

AMBIGUOUS = "Ambiguous request"
UNKNOWN = "No known action matches that input."
TEMP_DECIMALS = 1


def _method(obj, *names):
    for name in names:
        method = getattr(obj, name, None)
        if callable(method):
            return method
    raise AttributeError(f"{type(obj).__name__} has none of {', '.join(names)}: check the installed "
                         "scienceworld version against docs/sciworld_spec.md")


def tree_paths(tree: dict) -> list[str]:
    """Containment path of every live object below the root, by name: "kitchen/table/cup"."""
    paths = []

    def walk(node, parent):
        for child in (node.get("contents") or {}).values():
            if child.get("isDeleted") in (True, "true"):
                continue
            path = f"{parent}/{child['name']}" if parent else child["name"]
            paths.append(path)
            walk(child, path)

    walk(tree, "")
    return paths


def canonical(node: dict) -> dict:
    """The object tree without uuids, children in a fixed order, temperatures rounded."""
    out = {}
    for key, value in node.items():
        if key == "uuid":
            continue
        if key == "contents":
            children = [canonical(child) for child in (value or {}).values()]
            out[key] = sorted(children, key=lambda c: json.dumps(c, sort_keys=True))
        elif isinstance(value, dict):
            out[key] = canonical(value)
        elif key == "temperatureC" and isinstance(value, (int, float)):
            out[key] = round(value, TEMP_DECIMALS)
        else:
            out[key] = value
    return out


def full_signature(tree: dict, look: str, inventory: str) -> str:
    lines = sorted(line.strip() for line in look.splitlines()) + ["--"]
    lines += sorted(line.strip() for line in inventory.splitlines())
    text = json.dumps(canonical(tree), sort_keys=True) + "\n" + "\n".join(lines)
    return hashlib.sha1(text.encode()).hexdigest()


class SciWorld:
    """ScienceWorldEnv behind the small interface the labeller uses (tests/test_sciworld.py has a fake)."""

    def __init__(self, fast: bool = True, server_path: str | None = None):
        try:
            from scienceworld import ScienceWorldEnv
        except ImportError as error:
            raise ImportError("needs scienceworld and Java 1.8+; see docs/sciworld_spec.md") from error
        self._env = ScienceWorldEnv(serverPath=server_path, envStepLimit=10**9)
        self._server = getattr(self._env, "server", None) if fast else None
        self.task = self.variation = self.simplification = None
        self.wait1_ticks = None
        self._gold = None

    def load(self, task: str, variation: int, simplification: str = "easy") -> None:
        self._env.load(task, variation, simplification, False)
        self.task, self.variation, self.simplification = task, variation, simplification
        self._gold = None
        self.reset()
        before = self.moves()
        self._raw_step("wait1")
        self.wait1_ticks = self.moves() - before
        self.reset()

    def gold_actions(self) -> list[str]:
        """The gold path. Generating it plays the task inside load(), so only this call pays for it."""
        if self._gold is None:
            self._env.load(self.task, self.variation, self.simplification, True)
            gold = [str(c) for c in _method(self._env, "get_gold_action_sequence", "getGoldActionSequence")()]
            self._env.load(self.task, self.variation, self.simplification, False)  # cheap resets from here on
            if gold and gold[0].startswith("ERROR"):
                raise RuntimeError(gold[0])
            self._gold = gold
        return list(self._gold)

    def reset(self) -> None:
        if self._server is not None:
            self._server.reset()
        else:
            self._env.reset()

    def _raw_step(self, command: str) -> str:
        if self._server is not None:
            return str(self._server.step(command))
        return self._env.step(command)[0]

    def step(self, command: str) -> str:
        observation = self._raw_step(command)
        if observation.startswith(AMBIGUOUS):
            observation = self._raw_step("0")
        return observation

    def idle(self) -> None:
        """Advance the world exactly one tick without changing anything."""
        if self.wait1_ticks == 1:
            command = "wait1"
        else:
            command = next((c for c in self.valid_actions() if c.startswith("look at ")), None)
            if command is None:
                raise RuntimeError("no one-tick no-op here: nothing to look at")
        before = self.moves()
        self.step(command)
        if self.moves() != before + 1:
            raise RuntimeError(f"{command!r} took {self.moves() - before} ticks; idle() needs exactly one")

    def moves(self) -> int:
        return int(_method(self._env, "get_num_moves", "getNumMoves")())

    def valid_actions(self) -> list[str]:
        method = _method(self._env, "get_valid_action_object_combinations", "getValidActionObjectCombinations")
        return sorted({str(c) for c in method()})

    def task_description(self) -> str:
        return str(_method(self._env, "get_task_description", "getTaskDescription", "taskdescription")())

    def look(self) -> str:
        return str(self._env.look())

    def inventory(self) -> str:
        return str(self._env.inventory())

    def train_variations(self) -> list[int]:
        return [int(v) for v in _method(self._env, "get_variations_train", "getVariationsTrain")()]

    def task_names(self) -> list[str]:
        return [str(t) for t in _method(self._env, "get_task_names", "getTaskNames")()]

    def object_tree(self) -> dict:
        tree = _method(self._env, "getObjectTree", "get_object_tree")()
        return json.loads(tree) if isinstance(tree, str) else tree

    def signatures(self) -> tuple[str, tuple]:
        """(full, core): a hash of everything observable, and the sorted containment paths."""
        tree = self.object_tree()
        return full_signature(tree, self.look(), self.inventory()), tuple(sorted(tree_paths(tree)))

    def close(self) -> None:
        self._env.close()


def _timed(fn, repeats=1):
    start = time.time()
    for _ in range(repeats):
        value = fn()
    return value, (time.time() - start) / repeats


def main():
    parser = argparse.ArgumentParser(description="Check the installed ScienceWorld against what sciworld/ assumes.")
    parser.add_argument("--task", default="boil")
    parser.add_argument("--variation", type=int, default=0)
    parser.add_argument("--simplification", default="easy")
    parser.add_argument("--prefix", type=int, default=10, help="gold commands to replay")
    parser.add_argument("--slow-step", action="store_true")
    args = parser.parse_args()

    from importlib.metadata import version

    from .labeller import Replayer
    print("scienceworld", version("scienceworld"))
    env, seconds = _timed(lambda: SciWorld(fast=not args.slow_step))
    print(f"JVM start: {seconds:.1f} s")
    _, seconds = _timed(lambda: env.load(args.task, args.variation, args.simplification))
    print(f"load: {seconds:.2f} s   wait1 takes {env.wait1_ticks} tick(s) (idle() uses "
          f"{'wait1' if env.wait1_ticks == 1 else 'look at'})")
    gold, seconds = _timed(env.gold_actions)
    print(f"gold path: {len(gold)} commands, {seconds:.2f} s")
    prefix = gold[:args.prefix]
    _, reset_s = _timed(env.reset, 3)
    start = time.time()
    for command in prefix:
        observation = env.step(command)
        if observation.startswith(UNKNOWN):
            print(f"  gold command not understood: {command!r}")
    step_s = (time.time() - start) / max(1, len(prefix))
    _, signature_s = _timed(env.signatures, 3)
    valid, valid_s = _timed(env.valid_actions)
    print(f"reset {reset_s:.3f} s   step {step_s:.3f} s   signatures {signature_s:.3f} s   "
          f"valid actions {valid_s:.3f} s ({len(valid)} commands)")
    node_s = reset_s + (len(prefix) + 2) * step_s + signature_s
    print(f"estimated search node at prefix {len(prefix)}: {node_s:.2f} s "
          f"(about 10 nodes for an easy reversible pair, up to 137 with the default beam)")
    paths = tree_paths(env.object_tree())
    print("agent and inventory paths:", [p for p in paths if p.endswith("/agent") or "/inventory" in p][:6])
    print("deterministic replay:", Replayer(env, prefix).check_determinism())
    print("\n" + env.task_description() + "\n\n" + env.look() + "\n\n" + env.inventory())
    print("\nsample valid actions:", valid[:15])
    env.close()


if __name__ == "__main__":
    main()
