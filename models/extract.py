"""E1: save the hidden states at the candidate action for every labelled row (RQ1).

Each row's prompt is the agent's first turn with the action written in as the
assistant reply (models.prompts.probe_messages). One forward pass gives every
layer; we keep two token positions:

- last: the last token of the action text;
- mean: the mean over the action's tokens.

Output in --out (default results/hidden/<model>):
  last.npy, mean.npy  float32 [n_rows, n_layers + 1, hidden_size]; index 0 is the embedding layer
  ids.json            row ids in array order
  meta.json           model, shapes, library versions, input files
  progress.json       rows finished; rerunning the same command resumes from here

Usage:
  python -m models.extract --model qwen7b --rows data/sandbox_pairs.jsonl data/sciworld_pairs.jsonl
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from models.prompts import probe_messages


def load_rows(paths) -> list[dict]:
    rows = []
    for path in paths:
        with open(path) as f:
            rows += [json.loads(line) for line in f if line.strip()]
    return rows


def action_token_span(offsets, char_start: int, char_end: int) -> tuple[int, int]:
    """First and last token index overlapping the character span [char_start, char_end)."""
    hits = [i for i, (a, b) in enumerate(offsets) if b > a and a < char_end and b > char_start]
    if not hits:
        raise ValueError("no token overlaps the action text")
    return hits[0], hits[-1]


def _squash(text: str) -> str:
    return "".join(text.split())


def _covers(tokenizer, ids, first, last, action_text) -> bool:
    return _squash(action_text) in _squash(tokenizer.decode(ids[first:last + 1]))


def prepare(tokenizer, row: dict) -> tuple[list, int, int]:
    from models.load import chat_text
    text = chat_text(tokenizer, probe_messages(row), add_generation_prompt=False)
    start = text.rfind(row["action_text"])
    if start < 0:
        raise ValueError(f"{row['id']}: the chat template changed the action text")
    end = start + len(row["action_text"])
    enc = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    ids = enc["input_ids"]
    first, last = action_token_span(enc["offset_mapping"], start, end)
    if not _covers(tokenizer, ids, first, last, row["action_text"]):
        # Some fast tokenizers report wrong offsets; count the prefix's tokens instead.
        first = len(tokenizer(text[:start], add_special_tokens=False)["input_ids"])
        last = len(tokenizer(text[:end], add_special_tokens=False)["input_ids"]) - 1
        if not _covers(tokenizer, ids, first, last, row["action_text"]):
            raise ValueError(f"{row['id']}: cannot locate the action's tokens")
    return ids, first, last


def main():
    parser = argparse.ArgumentParser(description="Extract hidden states at the action tokens (E1).")
    parser.add_argument("--model", required=True)
    parser.add_argument("--rows", nargs="+", required=True)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--limit", type=int, help="first N rows only (smoke test)")
    args = parser.parse_args()

    import numpy as np
    import torch
    from models.load import load, versions

    rows = load_rows(args.rows)[: args.limit]
    ids = [r["id"] for r in rows]
    out = args.out or Path("results/hidden") / args.model
    out.mkdir(parents=True, exist_ok=True)

    model, tokenizer = load(args.model)
    n_layers, width = model.config.num_hidden_layers, model.config.hidden_size
    shape = (len(rows), n_layers + 1, width)

    progress_file = out / "progress.json"
    resume = progress_file.exists() and json.loads((out / "ids.json").read_text()) == ids
    done = json.loads(progress_file.read_text())["done"] if resume else 0
    if not resume:
        progress_file.unlink(missing_ok=True)  # a stale count must never skip rows of a fresh array
    arrays = {pos: np.lib.format.open_memmap(out / f"{pos}.npy", mode="r+" if resume else "w+",
                                             dtype=np.float32, shape=shape)
              for pos in ("last", "mean")}
    (out / "ids.json").write_text(json.dumps(ids))
    (out / "meta.json").write_text(json.dumps({
        "model": args.model, "n_rows": len(rows), "n_layers": n_layers, "hidden_size": width,
        "positions": ["last", "mean"], "layer_0": "embeddings", "rows": args.rows, **versions()}, indent=2))

    start = time.time()
    for i in range(done, len(rows)):
        input_ids, first, last = prepare(tokenizer, rows[i])
        with torch.no_grad():
            hidden = model(torch.tensor([input_ids], device=model.device),
                           output_hidden_states=True, use_cache=False).hidden_states
        # Index each layer on its own device first: with several GPUs, layers live on different ones.
        arrays["last"][i] = torch.stack([h[0, last].float().cpu() for h in hidden]).numpy()
        arrays["mean"][i] = torch.stack([h[0, first:last + 1].float().mean(dim=0).cpu() for h in hidden]).numpy()
        if (i + 1) % 100 == 0 or i + 1 == len(rows):
            for array in arrays.values():
                array.flush()
            progress_file.write_text(json.dumps({"done": i + 1}))
            rate = (time.time() - start) / (i + 1 - done)
            print(f"{i + 1}/{len(rows)} rows, {rate:.2f} s/row", flush=True)
    print(f"saved {shape} per position to {out}")


if __name__ == "__main__":
    main()
