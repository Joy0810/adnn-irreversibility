"""E6/E7: steering along the probe's irreversible direction (RQ4).

decision (E6): at each task's first turn, the margin
    log P('{"tool": "ask_user"') - log P('{"tool": "<shortcut tool>"')
for every alpha, with the probe direction and 3 random directions of the same norm.
The shared prefix '{"tool": "' cancels, so the margin compares only the tool choice.
One forward pass per candidate, so the whole sweep is cheap.

rollouts (E7): full episodes at a few alphas with the probe direction and one random
direction, scored exactly like E4. Report task success and parse failures next to
caution, so over-steering is visible.

Reads results/probes/<model>/direction.npy and direction.json (written by probes.train).

Usage:
  python -m experiments.steering --model qwen7b --mode decision
  python -m experiments.steering --model qwen7b --mode rollouts --alphas -2 0 2
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from experiments.agent_runs import run_condition
from models.prompts import agent_messages
from sandbox.tasks import VERSIONS, build_task_pairs

DEFAULT_ALPHAS = [-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0]


def main():
    parser = argparse.ArgumentParser(description="Activation steering (E6/E7).")
    parser.add_argument("--model", required=True)
    parser.add_argument("--mode", choices=["decision", "rollouts"], required=True)
    parser.add_argument("--alphas", type=float, nargs="+", default=DEFAULT_ALPHAS)
    parser.add_argument("--random-seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--limit", type=int, help="first N task pairs only")
    parser.add_argument("--task-seed", type=int, default=0)
    parser.add_argument("--probes", type=Path, help="default results/probes/<model>")
    parser.add_argument("--out", type=Path, help="default results/steering/<model>")
    args = parser.parse_args()

    from models.load import continuation_logprob, load, make_generate
    from models.steer import random_direction, steer

    probes = args.probes or Path("results/probes") / args.model
    out = args.out or Path("results/steering") / args.model
    out.mkdir(parents=True, exist_ok=True)
    direction = np.load(probes / "direction.npy")
    layer = json.loads((probes / "direction.json").read_text())["layer"]
    directions = {"probe": direction}
    directions.update({f"random{s}": random_direction(direction, s) for s in args.random_seeds})
    model, tokenizer = load(args.model)
    pairs = build_task_pairs(args.task_seed)[: args.limit]

    if args.mode == "decision":
        with open(out / "decision.jsonl", "w") as f:
            for name, vector in directions.items():
                for alpha in args.alphas:
                    with steer(model, layer, vector, alpha):
                        for pair in pairs:
                            for version in VERSIONS:
                                messages = agent_messages(pair.instruction, pair.state_text(version))
                                ask = continuation_logprob(model, tokenizer, messages, '{"tool": "ask_user"')
                                shortcut_tool = pair.shortcut_example["tool"]
                                short = continuation_logprob(model, tokenizer, messages, f'{{"tool": "{shortcut_tool}"')
                                f.write(json.dumps({"direction": name, "alpha": alpha, "layer": layer,
                                                    "task_id": pair.id, "template": pair.template,
                                                    "version": version, "logp_ask": ask, "logp_shortcut": short,
                                                    "margin": ask - short}) + "\n")
                    print(f"decision: {name} alpha {alpha} done", flush=True)
        return

    generate = make_generate(model, tokenizer)
    for name in ("probe", f"random{args.random_seeds[0]}"):
        for alpha in args.alphas:
            if name != "probe" and alpha == 0:
                continue  # identical to the unsteered probe run
            with steer(model, layer, directions[name], alpha):
                run_condition(pairs, lambda pair, version: generate, out / "rollouts" / f"{name}_alpha{alpha:g}")
            print(f"rollouts: {name} alpha {alpha} done", flush=True)


if __name__ == "__main__":
    main()
