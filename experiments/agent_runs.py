"""E4/E5: run every behaviour task pair in both versions and score caution (RQ3).

Each episode is saved as its own JSON file with every prompt and reply, so a
rerun skips finished episodes. scores.jsonl is rebuilt from all episode files at
the end.

Usage:
  python -m experiments.agent_runs --model qwen7b               # E4: neutral system prompt
  python -m experiments.agent_runs --model qwen7b --careful     # E5: careful-prompt ablation
  python -m experiments.agent_runs --model dry-run --limit 3    # no GPU: a scripted policy checks the plumbing
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from agent.caution import score_episode
from agent.loop import run_episode, scripted_policy
from sandbox.tasks import VERSIONS, build_task_pairs


def run_condition(pairs, generate_for, out_dir: Path, careful: bool = False, max_steps: int = 15) -> list[dict]:
    """Run (or reuse) every episode in out_dir and return the scores."""
    out_dir.mkdir(parents=True, exist_ok=True)
    scores = []
    for pair in pairs:
        for version in VERSIONS:
            path = out_dir / f"{pair.id}_{version}.json"
            if path.exists():
                episode = json.loads(path.read_text())
            else:
                episode = run_episode(pair, version, generate_for(pair, version), careful, max_steps)
                path.write_text(json.dumps(episode))
            scores.append(score_episode(pair, episode))
        print(f"{pair.id}: done", flush=True)
    with open(out_dir / "scores.jsonl", "w") as f:
        for score in scores:
            f.write(json.dumps(score) + "\n")
    return scores


def main():
    parser = argparse.ArgumentParser(description="Agent runs on the behaviour tasks (E4/E5).")
    parser.add_argument("--model", required=True, help="qwen7b, llama8b, qwen14b, or dry-run")
    parser.add_argument("--careful", action="store_true", help="add the careful instruction to the system prompt")
    parser.add_argument("--limit", type=int, help="first N task pairs only")
    parser.add_argument("--task-seed", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=15)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    pairs = build_task_pairs(args.task_seed)[: args.limit]
    if args.model == "dry-run":
        def generate_for(pair, version):
            return scripted_policy(pair)
    else:
        from models.load import load, make_generate
        generate = make_generate(*load(args.model))

        def generate_for(pair, version):
            return generate

    condition = "careful" if args.careful else "neutral"
    out = args.out or Path("results/agent") / args.model / condition
    scores = run_condition(pairs, generate_for, out, args.careful, args.max_steps)
    for version in VERSIONS:
        rows = [s for s in scores if s["version"] == version]
        rate = lambda key: sum(s[key] for s in rows) / len(rows)
        print(f"{version}: shortcut {rate('shortcut_taken'):.0%}, caution {rate('caution'):.0%}, "
              f"irreversible taken {rate('irreversible_taken'):.0%}, success {rate('success'):.0%}, "
              f"parse failures {sum(s['parse_failures'] for s in rows)}")


if __name__ == "__main__":
    main()
