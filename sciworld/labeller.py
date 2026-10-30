"""Reversibility labels for ScienceWorld actions by replay search.

ScienceWorld cannot save or restore a state (docs/sciworld_spec.md), so a Replayer
rebuilds every search node by resetting the task variation and replaying commands.
When the environment already sits on a prefix of the target, it steps forward instead.

The world also changes by itself: temperatures move every tick and plants grow. So
the target is a do-nothing counterfactual, not the saved state. Action a at state s
is undone at depth k if the signature after a and k further commands equals the
signature after the same number of idle ticks from s. The match is on ticks, not
commands, because one command can take more than one tick.

Two signatures come from env.signatures():
- full: everything the environment exposes (object properties, the agent, the inventory);
- core: where every object is, as containment paths such as "kitchen/table/cup".

Labels at horizon h, with the same rule as sandbox/labeller.py:
- reversible: the full signature comes back within h further commands;
- partial: only the core signature comes back (everything is where it was, but a
  temperature, a state of matter or a device setting differs);
- irreversible: neither.
A command that changes nothing is reversible at depth 0.

The search is breadth first to depth max_h and stops at the first exact restore. At
each node it tries at most max_branch restoring candidates, in this order:
1. the inverse of a: open/close, activate/deactivate, connect/disconnect, pour back;
2. commands that put a displaced object back where the counterfactual has it
   (move X to Y, pick up X, put down X), or take the agent back (go to, teleport to);
3. other valid commands that mention an object of a or a displaced object.
Meta commands (look, inventory, read, focus, wait, reset task) are never labelled and
never tried. Eat, mix and flush are never tried. Each depth keeps the `beam` nodes
closest to the target in core terms. The search gives up after max_nodes nodes and
marks the result truncated. Like the sandbox search, this is a heuristic, not an
exhaustive search.
"""
from __future__ import annotations

import re
from collections import Counter, defaultdict
from dataclasses import dataclass

from sandbox.labeller import LABELS, LabelResult

AGENT, INVENTORY = "agent", "inventory"
IDLE = None  # an idle tick in Replayer.applied; equal to no real command

VERBS = ("look around", "look at", "look in", "pick up", "put down", "go to", "teleport to", "focus on",
         "reset task", "deactivate", "activate", "disconnect", "connect", "open", "close", "move", "pour",
         "dunk", "mix", "use", "eat", "flush", "read", "examine", "wait1", "wait", "inventory", "task")
META = frozenset({"look around", "look at", "look in", "examine", "inventory", "task", "read", "focus on",
                  "wait", "wait1", "reset task"})
DESTRUCTIVE = frozenset({"eat", "mix", "flush"})
INVERSE = {"open": "close", "close": "open", "activate": "deactivate", "deactivate": "activate"}
SEPARATORS = {"move": " to ", "connect": " to ", "pour": " into ", "dunk": " into ", "use": " on "}
RESTORING_VERBS = ("move", "pick up", "put down", "pour", "close", "open", "deactivate", "activate",
                   "disconnect", "connect", "go to", "teleport to", "dunk", "use")


@dataclass
class SearchResult(LabelResult):
    nodes: int = 0  # search nodes evaluated, the action itself included
    truncated: bool = False  # stopped at max_nodes before finishing max_h


# --- commands -----------------------------------------------------------------


def verb_of(command: str) -> str:
    command = command.strip()
    for verb in VERBS:
        if command == verb or command.startswith(verb + " "):
            return verb
    return command.split(" ", 1)[0]


def objects_of(command: str) -> list[str]:
    """Object referents in a command: "move cup containing water to table" -> [cup containing water, table]."""
    verb = verb_of(command)
    rest = command.strip()[len(verb):].strip()
    separator = SEPARATORS.get(verb)
    if separator and separator in rest:
        return [part.strip() for part in rest.split(separator, 1)]
    return [rest] if rest else []


def labellable(command: str) -> bool:
    return verb_of(command) not in META


def head(referent: str) -> str:
    return referent.split(" containing ", 1)[0].strip()


def inverse_commands(command: str) -> list[str]:
    verb, objects = verb_of(command), objects_of(command)
    if verb in INVERSE and objects:
        return [f"{INVERSE[verb]} {objects[0]}"]
    if verb == "connect":
        return [f"disconnect {obj}" for obj in objects]
    if verb == "pour" and len(objects) == 2:
        return [f"pour {objects[1]} into {objects[0]}"]
    return []


def _same_object(a, b):
    a, b = head(a), head(b)
    return a == b or a.endswith(" " + b) or b.endswith(" " + a)


def _same_objects(a, b):
    """Pairwise the same objects, allowing a longer referent such as "glass cup" for "cup"."""
    return len(a) == len(b) and all(_same_object(x, y) for x, y in zip(a, b))


# --- core signatures ------------------------------------------------------------


def _parent_and_name(path):
    parent, _, name = path.rpartition("/")
    return parent.rpartition("/")[2], name


def distance(core, target_core) -> int:
    have, want = Counter(core), Counter(target_core)
    return sum((have - want).values()) + sum((want - have).values())


def placements(core, target_core) -> tuple[list[str], set[str]]:
    """Commands that would put displaced objects back where `target_core` has them,
    and the names of the displaced objects (moved, gone or new)."""
    have, want = Counter(core), Counter(target_core)
    rooms = {path for path in want if "/" not in path}
    current, wanted = defaultdict(set), defaultdict(set)
    for path in have - want:
        parent, name = _parent_and_name(path)
        current[name].add(parent)
    for path in want - have:
        parent, name = _parent_and_name(path)
        wanted[name].add(parent)
    commands = []
    for name in sorted(current):
        for target in sorted(wanted.get(name, ())):
            if target in current[name]:
                continue
            if name == AGENT:
                commands += [f"go to {target}", f"teleport to {target}"]
            elif target == INVENTORY:
                commands.append(f"pick up {name}")
            else:
                commands.append(f"move {name} to {target}")
                if target in rooms:
                    commands.append(f"put down {name}")
    displaced = {name for name in current.keys() | wanted.keys()
                 if current.get(name) != wanted.get(name)} - {AGENT, INVENTORY}
    return commands, displaced


def candidates(valid, path, core, target_core, max_branch: int = 8) -> list[str]:
    """Restoring commands to try after `path` (the action, then the restore so far)."""
    action, last = path[0], path[-1]
    verbs = {c: verb_of(c) for c in valid}
    pool = [c for c in valid if verbs[c] not in META | DESTRUCTIVE and c != last
            and not (len(path) > 1 and last in inverse_commands(c))]  # never undo the last restore step
    hints, displaced = placements(core, target_core)
    ranked = []
    for hint in inverse_commands(action) + hints:
        verb, objects = verb_of(hint), objects_of(hint)
        ranked += [c for c in pool if c == hint or (verbs[c] == verb and _same_objects(objects, objects_of(c)))]
    keys = {head(obj) for obj in objects_of(action)} | displaced
    patterns = [re.compile(rf"(?<![\w-]){re.escape(key)}(?![\w-])") for key in sorted(keys) if key]
    rank = {verb: i for i, verb in enumerate(RESTORING_VERBS)}
    others = sorted((c for c in pool if any(p.search(c) for p in patterns)),
                    key=lambda c: (rank.get(verbs[c], len(rank)), c))
    return list(dict.fromkeys(ranked + others))[:max_branch]


# --- replay and search ----------------------------------------------------------


class Replayer:
    """Rebuilds the state after `prefix` plus a path, and caches the idle counterfactual.

    `applied` records the commands run since the last reset, so a target that extends
    them costs only the missing steps. Idle ticks are recorded as IDLE, so leaving an
    idle run always resets.
    """

    def __init__(self, env, prefix):
        self.env, self.prefix = env, list(prefix)
        self.applied = None
        self.resets = 0
        self.goto()
        self.start = env.moves()
        self._baseline = {0: env.signatures()}  # ticks after the prefix -> (full, core)

    def goto(self, path=()) -> None:
        target = self.prefix + list(path)
        if self.applied is None or self.applied != target[:len(self.applied)]:
            self.env.reset()
            self.resets += 1
            self.applied = []
        for command in target[len(self.applied):]:
            self.env.step(command)
            self.applied.append(command)

    def evaluate(self, path) -> tuple[int, str, tuple]:
        """(ticks since the prefix state, full signature, core signature) after `path`."""
        self.goto(path)
        full, core = self.env.signatures()
        return self.env.moves() - self.start, full, core

    def baseline(self, ticks: int) -> tuple[str, tuple]:
        """Signatures after `ticks` idle ticks from the prefix state."""
        while ticks not in self._baseline:
            idled = self.prefix + [IDLE] * max(self._baseline)
            if (self.applied is None or len(self.applied) < len(self.prefix)
                    or self.applied != idled[:len(self.applied)]):
                self.goto()
            while len(self.applied) <= len(idled):
                self.env.idle()
                self.applied.append(IDLE)
                done = len(self.applied) - len(self.prefix)
                if done not in self._baseline:
                    self._baseline[done] = self.env.signatures()
        return self._baseline[ticks]

    def check_determinism(self) -> bool:
        """Replay the prefix again from a reset; True if it lands on the same state."""
        self.applied = None
        self.goto()
        return self.env.moves() == self.start and self.env.signatures() == self._baseline[0]


def label_action(replayer: Replayer, action: str, max_h: int = 3, max_branch: int = 8, beam: int = 8,
                 max_nodes: int = 150) -> SearchResult:
    """Label `action` at the replayer's prefix state by search up to `max_h` restoring commands."""
    rp = replayer
    rp.baseline(max_h + 1)  # one idle run covers the usual one tick per command
    result = SearchResult(exact_depth=None, core_depth=None, max_h=max_h, nodes=1)
    ticks, full, core = rp.evaluate([action])
    target_full, target_core = rp.baseline(ticks)
    if full == target_full:
        result.exact_depth = result.core_depth = 0
        return result
    if core == target_core:
        result.core_depth = 0

    seen = {(ticks, full)}
    frontier = [([action], core, ticks)]
    for depth in range(1, max_h + 1):
        children = []
        for path, node_core, node_ticks in frontier:
            here_core = rp.baseline(node_ticks)[1]  # before goto: a new baseline moves the env
            rp.goto(path)
            for command in candidates(rp.env.valid_actions(), path, node_core, here_core, max_branch):
                if result.nodes >= max_nodes:
                    result.truncated = True
                    return result
                child = path + [command]
                ticks, full, core = rp.evaluate(child)
                result.nodes += 1
                if (ticks, full) in seen:
                    continue
                seen.add((ticks, full))
                target_full, target_core = rp.baseline(ticks)
                if full == target_full:
                    result.exact_depth = depth
                    if result.core_depth is None:
                        result.core_depth = depth
                    result.restore_sequence = child[1:]
                    return result
                if result.core_depth is None and core == target_core:
                    result.core_depth = depth
                    result.restore_sequence = child[1:]
                children.append((distance(core, target_core), len(children), child, core, ticks))
        frontier = [(child, core, ticks) for _, _, child, core, ticks in sorted(children)[:beam]]
    return result
