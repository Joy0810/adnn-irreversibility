"""E3: the stated answer (RQ2) and the A/B pair question (the gap's K).

answers: for every labelled row, ask whether the action could be undone. We keep
the generated reply and its parsed label (primary; unparseable replies are kept
and counted), and the answer word with the highest log-probability (secondary,
never fails to parse).

pairs: for every behaviour task pair, show both versions as A and B, in both
orders, and ask in which version the shortcut would be harder to undo. A pair
counts towards K only if both orders are answered correctly.

Output in results/stated/<model>/: answers.jsonl and pairs.jsonl, both resumable.

Usage:
  python -m experiments.stated --model qwen7b --rows data/sandbox_pairs.jsonl data/sciworld_pairs.jsonl
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from models.prompts import (STATED_TO_LABEL, STATED_WORDS, pair_messages, parse_pair, parse_stated,
                            stated_messages)
from sandbox.tasks import build_task_pairs


def done_keys(path: Path, key: str) -> set:
    if not path.exists():
        return set()
    with open(path) as f:
        return {json.loads(line)[key] for line in f if line.strip()}


def main():
    parser = argparse.ArgumentParser(description="Stated answers and the A/B pair question (E3).")
    parser.add_argument("--model", required=True)
    parser.add_argument("--rows", nargs="+", required=True)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--limit", type=int, help="first N rows only (smoke test)")
    parser.add_argument("--task-seed", type=int, default=0)
    args = parser.parse_args()

    from models.extract import load_rows
    from models.load import continuation_logprob, generate, load
    out = args.out or Path("results/stated") / args.model
    out.mkdir(parents=True, exist_ok=True)
    model, tokenizer = load(args.model)

    answers = out / "answers.jsonl"
    seen = done_keys(answers, "id")
    with open(answers, "a") as f:
        for i, row in enumerate(load_rows(args.rows)[: args.limit]):
            if row["id"] in seen:
                continue
            messages = stated_messages(row)
            reply = generate(model, tokenizer, messages, max_new_tokens=64)
            logprobs = {word: continuation_logprob(model, tokenizer, messages, word) for word in STATED_WORDS}
            f.write(json.dumps({
                "id": row["id"], "environment": row["environment"], "label": row["label"],
                "reply": reply, "stated": parse_stated(reply),
                "stated_logprob": STATED_TO_LABEL[max(logprobs, key=logprobs.get)], "logprobs": logprobs,
                "messages": messages}) + "\n")
            f.flush()
            if i % 100 == 0:
                print(f"answers: {i} rows", flush=True)

    pairs_file = out / "pairs.jsonl"
    seen = done_keys(pairs_file, "key")
    with open(pairs_file, "a") as f:
        for pair in build_task_pairs(args.task_seed):
            for order in ("irreversible_first", "reversible_first"):
                key = f"{pair.id}:{order}"
                if key in seen:
                    continue
                a, b = ("irreversible", "reversible") if order == "irreversible_first" else ("reversible", "irreversible")
                messages = pair_messages(pair.instruction, pair.state_text(a), pair.state_text(b),
                                         json.dumps(pair.shortcut_example))
                reply = generate(model, tokenizer, messages, max_new_tokens=64)
                answer = parse_pair(reply)
                f.write(json.dumps({"key": key, "task_id": pair.id, "order": order, "reply": reply,
                                    "answer": answer, "correct": answer == ("A" if a == "irreversible" else "B"),
                                    "messages": messages}) + "\n")
                f.flush()
    print(f"wrote {answers} and {pairs_file}")


if __name__ == "__main__":
    main()
