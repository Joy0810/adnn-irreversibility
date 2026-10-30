"""Generate labelled ScienceWorld (state, action) pairs.

1. Plan. For each task, pick --variations train variations and fetch their gold paths
   (sample_states.plan_scenario).
2. Label. Each job is one state: the first p commands of a gold path, --states per
   variation. The worker rebuilds the state and replays it a second time to check that
   replay is deterministic (ScienceWorld 1.2.x is not; such states are skipped). It then
   picks up to --per-state actions, stratified by verb, and labels each one by replay
   search (labeller.label_action). Actions that change nothing are dropped unless
   --keep-noops is set.
3. Write. Every labelled row is appended to --raw as it arrives, so --resume can skip
   states that are already done. At the end the rows are subsampled to roughly balanced
   classes (at most --target/3 per class at horizon 3; --target 0 keeps every row). Rows
   in one scenario with the same action text but different labels get a shared pair_id.
   The result goes to --out.

Each worker process owns one ScienceWorld environment, so there is one JVM per worker.
All rows from one task variation share a scenario_id ("sw-boil-v12"). Split train and
test by that field.

Usage:
    python -m sciworld.generate --tasks boil,find-living-thing --variations 1 --states 3 \\
        --limit 20 --workers 1 --target 0 --out data/sciworld_smoke.jsonl
    python -m sciworld.generate --variations 5 --states 8 --per-state 4 --workers 16
"""
from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing
import random
import time
from collections import Counter, defaultdict
from pathlib import Path

from .labeller import LABELS, Replayer, label_action, verb_of
from .sample_states import DEFAULT_TASKS, choose_actions, choose_prefixes, plan_scenario, state_text


def scenario_id(scenario: dict) -> str:
    return f"sw-{scenario['task']}-v{scenario['variation']}"


def make_row(scenario, prefix_len, index, action, result, text, seconds) -> dict:
    sid = scenario_id(scenario)
    labels = {f"label_h{h}": result.label(h) if h <= result.max_h else None
              for h in range(1, max(3, result.max_h) + 1)}
    return {
        "id": f"{sid}-p{prefix_len:03d}-a{index}",
        "environment": "scienceworld",
        "state_text": text,
        "action_text": action,
        "label": result.label(result.max_h),
        "horizon": result.max_h,
        "source": "scienceworld",
        "task_text": scenario["task_text"],
        "scenario_id": sid,
        "pair_id": None,
        "mechanism": verb_of(action),
        "variant": None,
        "cue_style": None,
        **labels,
        "restore_sequence": result.restore_sequence,
        "action": action,
        "replay": {"task": scenario["task"], "variation": scenario["variation"],
                   "simplification": scenario["simplification"], "prefix": scenario["gold"][:prefix_len]},
        "search": {"exact_depth": result.exact_depth, "core_depth": result.core_depth, "nodes": result.nodes,
                   "truncated": result.truncated, "seconds": round(seconds, 3)},
    }


def label_state(env, scenario: dict, prefix_len: int, per_state: int = 4, max_h: int = 3, seed: int = 0,
                keep_noops: bool = False, **search) -> dict:
    """Label up to `per_state` actions at the state after the first `prefix_len` gold commands."""
    key = (scenario["task"], scenario["variation"], scenario["simplification"])
    if (env.task, env.variation, env.simplification) != key:
        env.load(*key)
    outcome = {"state": f"{scenario_id(scenario)}/p{prefix_len}", "rows": [], "noops": 0, "status": "ok"}
    gold = scenario["gold"]
    replayer = Replayer(env, gold[:prefix_len])
    if not replayer.check_determinism():
        outcome["status"] = "nondeterministic"
        return outcome
    text = state_text(env)
    rng = random.Random(f"{seed}/{scenario_id(scenario)}/{prefix_len}")
    upcoming = gold[prefix_len] if prefix_len < len(gold) else None
    for index, action in enumerate(choose_actions(env.valid_actions(), per_state, rng, prefer=upcoming)):
        start = time.time()
        result = label_action(replayer, action, max_h=max_h, **search)
        if result.exact_depth == 0 and not keep_noops:
            outcome["noops"] += 1
            continue
        outcome["rows"].append(make_row(scenario, prefix_len, index, action, result, text, time.time() - start))
    return outcome


def balance(rows: list[dict], target: int, seed: int = 0) -> list[dict]:
    """At most ceil(target / 3) rows per horizon-3 label; rarer classes keep every row."""
    rng = random.Random(seed)
    cap = -(-target // len(LABELS))
    out = []
    for label in LABELS:
        group = sorted((r for r in rows if r["label"] == label), key=lambda r: r["id"])
        out += rng.sample(group, min(cap, len(group)))
    return sorted(out, key=lambda r: r["id"])


def mark_pairs(rows: list[dict]) -> list[dict]:
    """Same scenario, same action text, different labels: give those rows one pair_id."""
    groups = defaultdict(list)
    for row in rows:
        groups[(row["scenario_id"], row["action_text"])].append(row)
    for (sid, action), group in groups.items():
        if len({r["label"] for r in group}) > 1:
            pair_id = f"{sid}-{hashlib.sha1(action.encode()).hexdigest()[:8]}"
            for row in group:
                row["pair_id"] = pair_id
    return rows


def summarize(rows: list[dict]) -> str:
    lines = [f"rows: {len(rows)}   scenarios: {len({r['scenario_id'] for r in rows})}"]
    for h in (1, 2, 3):
        counts = Counter(r.get(f"label_h{h}") for r in rows)
        lines.append(f"horizon {h}: " + ", ".join(f"{k} {counts[k]}" for k in LABELS))
    by_verb = defaultdict(Counter)
    for row in rows:
        by_verb[row["mechanism"]][row["label"]] += 1
    lines.append("by verb, h=3 (R/P/I): " + ", ".join(
        f"{verb} {c['reversible']}/{c['partial']}/{c['irreversible']}" for verb, c in sorted(by_verb.items())))
    if rows:
        search = [r["search"] for r in rows]
        lines.append(f"search: {sum(s['seconds'] for s in search) / len(rows):.1f} s and "
                     f"{sum(s['nodes'] for s in search) / len(rows):.0f} nodes per pair on average; "
                     f"{sum(s['truncated'] for s in search)} truncated")
    return "\n".join(lines)


# --- workers: one environment (one JVM) per process ---------------------------------

_ENV = None
_ENV_KWARGS: dict = {}


def _init_worker(env_kwargs):
    global _ENV_KWARGS
    _ENV_KWARGS = env_kwargs


def _env():
    global _ENV
    if _ENV is None:
        from .env import SciWorld
        _ENV = SciWorld(**_ENV_KWARGS)
    return _ENV


def _plan_job(job):
    task, n_variations, simplification, seed = job
    try:
        env = _env()
        env.load(task, 0, simplification)
        variations = sorted(env.train_variations())
        chosen = sorted(random.Random(f"{seed}/{task}").sample(variations, min(n_variations, len(variations))))
        return [plan_scenario(env, task, v, simplification) for v in chosen], None
    except Exception as error:  # one bad task should not stop the run
        return [], f"{task}: {type(error).__name__}: {error}"


def _state_job(job):
    scenario, prefix_len, options = job
    try:
        return label_state(_env(), scenario, prefix_len, **options)
    except Exception as error:
        return {"state": f"{scenario_id(scenario)}/p{prefix_len}", "rows": [], "noops": 0, "status": "error",
                "error": f"{type(error).__name__}: {error}"}


def _run(fn, jobs, workers, env_kwargs):
    """Yield fn(job) for every job: inline if workers <= 1, else in a pool of processes."""
    if workers <= 1:
        _init_worker(env_kwargs)
        for job in jobs:
            yield fn(job)
        return
    with multiprocessing.get_context("spawn").Pool(workers, _init_worker, (env_kwargs,)) as pool:
        yield from pool.imap_unordered(fn, jobs)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--tasks", default=",".join(DEFAULT_TASKS), help="comma-separated task names")
    parser.add_argument("--variations", type=int, default=5, help="train variations per task")
    parser.add_argument("--states", type=int, default=8, help="gold-path prefixes per variation")
    parser.add_argument("--per-state", type=int, default=4, help="actions labelled per state")
    parser.add_argument("--max-h", type=int, default=3)
    parser.add_argument("--max-branch", type=int, default=8, help="restoring candidates per search node")
    parser.add_argument("--beam", type=int, default=8, help="search nodes kept per depth")
    parser.add_argument("--max-nodes", type=int, default=150, help="search nodes per pair before giving up")
    parser.add_argument("--simplification", default="easy")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0, help="stop after this many new rows (0: no limit)")
    parser.add_argument("--target", type=int, default=800, help="size of the balanced output; 0 keeps every row")
    parser.add_argument("--keep-noops", action="store_true", help="keep actions that change nothing")
    parser.add_argument("--slow-step", action="store_true", help="step through ScienceWorldEnv.step")
    parser.add_argument("--resume", action="store_true", help="skip states already in --raw")
    parser.add_argument("--out", type=Path, default=Path("data/sciworld_pairs.jsonl"))
    parser.add_argument("--raw", type=Path, default=None, help="every labelled row (default: OUT_raw.jsonl)")
    args = parser.parse_args()

    raw_path = args.raw or args.out.with_name(args.out.stem + "_raw.jsonl")
    env_kwargs = {"fast": not args.slow_step}
    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    start = time.time()

    scenarios = []
    plan_jobs = [(task, args.variations, args.simplification, args.seed) for task in tasks]
    for found, problem in _run(_plan_job, plan_jobs, min(args.workers, len(tasks)), env_kwargs):
        scenarios += found
        if problem:
            print("planning failed:", problem)
    scenarios.sort(key=scenario_id)
    print(f"planned {len(scenarios)} scenarios in {time.time() - start:.0f} s; gold path lengths "
          + ", ".join(f"{scenario_id(s)} {len(s['gold'])}/{s['gold_length']}" for s in scenarios))

    rows, done = [], set()
    if args.resume and raw_path.exists():
        rows = [json.loads(line) for line in raw_path.open() if line.strip()]
        done = {(r["scenario_id"], len(r["replay"]["prefix"])) for r in rows}
    options = {"per_state": args.per_state, "max_h": args.max_h, "seed": args.seed, "keep_noops": args.keep_noops,
               "max_branch": args.max_branch, "beam": args.beam, "max_nodes": args.max_nodes}
    jobs = [(s, p, options) for s in scenarios
            for p in choose_prefixes(len(s["gold"]), args.states, random.Random(f"{args.seed}/{scenario_id(s)}"))
            if (scenario_id(s), p) not in done]
    random.Random(args.seed).shuffle(jobs)  # so that --limit samples across tasks
    print(f"labelling {len(jobs)} states with {args.workers} worker(s)"
          + (f"; {len(done)} states already in {raw_path}" if done else ""))

    statuses, errors, noops, new = Counter(), [], 0, 0
    label_start = time.time()
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    outcomes = _run(_state_job, jobs, args.workers, env_kwargs)
    try:
        with raw_path.open("a" if args.resume else "w") as raw:
            for outcome in outcomes:
                statuses[outcome["status"]] += 1
                noops += outcome["noops"]
                if "error" in outcome:
                    errors.append(f"{outcome['state']}: {outcome['error']}")
                for row in outcome["rows"][:args.limit - new if args.limit else None]:
                    raw.write(json.dumps(row) + "\n")
                    rows.append(row)
                    new += 1
                raw.flush()
                if args.limit and new >= args.limit:
                    break
    finally:
        outcomes.close()  # stops the pool, and with it the JVMs, when --limit cuts the run short
        if _ENV is not None:
            _ENV.close()
    elapsed = time.time() - label_start

    final = mark_pairs(balance(rows, args.target, args.seed) if args.target else sorted(rows, key=lambda r: r["id"]))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as f:
        for row in final:
            f.write(json.dumps(row) + "\n")

    print("states: " + ", ".join(f"{k} {v}" for k, v in sorted(statuses.items()))
          + f"; no-op actions dropped: {noops}")
    for error in errors[:10]:
        print("  error:", error)
    print("all labelled rows:\n" + summarize(rows))
    if new:
        print(f"wall time: {elapsed / new:.1f} s per new pair with {args.workers} worker(s) "
              f"({elapsed / 60:.1f} min for {new} pairs)")
    print(f"wrote {len(rows)} rows to {raw_path} and {len(final)} to {args.out}"
          + (f" (balanced, target {args.target})" if args.target else ""))
    if args.target:
        print("balanced output:\n" + summarize(final))


if __name__ == "__main__":
    main()
