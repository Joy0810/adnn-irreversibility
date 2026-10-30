"""Choose the ScienceWorld states and actions to label.

States are prefixes of gold trajectories. The gold path is the cheap way to reach
states deep inside a task (a heated substance, a half-built circuit, a planted seed),
and replaying its first p commands rebuilds the state. For each task and train
variation we keep `states` prefixes, one from each equal slice of the path.

At each state the actions come from the valid-action list, stratified by verb: one
random command per verb in turn. Otherwise the "move X to Y" combinations, which make
up most of the list, would swamp it. The next gold command goes first when it can be
labelled, because it is what a competent agent would do next. Meta commands (look,
inventory, read, focus, wait, reset task) are never chosen; see labeller.META.

The default tasks between them involve heating, melting, measuring, devices,
electrical circuits, living things, planting, mixing and moving objects. The "easy"
simplifications (teleport, open doors, self-watering pots, and no electrical actions
outside the electrical tasks) cut the branching.
"""
from __future__ import annotations

import random

from .env import UNKNOWN
from .labeller import labellable, verb_of

DEFAULT_TASKS = ("boil", "melt", "use-thermometer", "power-component", "test-conductivity", "find-living-thing",
                 "grow-plant", "chemistry-mix", "chemistry-mix-paint-secondary-color",
                 "inclined-plane-determine-angle")


def plan_scenario(env, task: str, variation: int, simplification: str = "easy") -> dict:
    """Load one variation and keep the longest prefix of its gold path that replays cleanly."""
    env.load(task, variation, simplification)
    gold = env.gold_actions()
    env.reset()
    task_text = env.task_description()
    kept = []
    for command in gold:
        if command.strip().isdigit():  # an answer to an ambiguity prompt; env.step already answers 0
            continue
        if env.step(command).startswith(UNKNOWN):
            break
        kept.append(command)
    return {"task": task, "variation": variation, "simplification": simplification, "gold": kept,
            "gold_length": len(gold), "task_text": task_text}


def choose_prefixes(n_gold: int, k: int, rng: random.Random) -> list[int]:
    """k prefix lengths in [0, n_gold), one from each equal slice; all of them if there are fewer."""
    if n_gold <= k:
        return list(range(n_gold))
    edges = [i * n_gold // k for i in range(k + 1)]
    return [rng.randrange(lo, hi) for lo, hi in zip(edges, edges[1:])]


def choose_actions(valid, k: int, rng: random.Random, prefer: str | None = None) -> list[str]:
    """Up to k labellable commands, one verb at a time; `prefer` first if it is labellable."""
    by_verb = {}
    for command in sorted(set(valid)):
        if labellable(command):
            by_verb.setdefault(verb_of(command), []).append(command)
    for commands in by_verb.values():
        rng.shuffle(commands)
    verbs = sorted(by_verb)
    rng.shuffle(verbs)
    chosen = [prefer] if prefer and labellable(prefer) and k > 0 else []
    while len(chosen) < k and any(by_verb.values()):
        for verb in verbs:
            if by_verb[verb] and len(chosen) < k:
                command = by_verb[verb].pop()
                if command not in chosen:
                    chosen.append(command)
    return chosen


def state_text(env) -> str:
    """What a model sees: the task, the room and the inventory."""
    return "\n\n".join(part.strip() for part in (env.task_description(), env.look(), env.inventory()))
