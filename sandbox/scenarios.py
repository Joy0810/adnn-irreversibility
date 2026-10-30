"""Generate labelled (state, action) pairs from the sandbox.

Each world is a random file system, outbox and wallet. From each world we build:

- minimal-pair groups: one action and one task, placed in 3 states that differ
  only in what decides reversibility (a setting, a backup, or whether the target
  path is taken). Each variant has a designed label, and the BFS label must
  match it; a mismatch is a bug and raises.
- random rows: forward actions (delete, edit, move, send, pay, ...) on the
  world as generated, labelled by BFS alone.

All rows from one world share a scenario_id. Splitting by scenario_id therefore
keeps every minimal pair on one side of the split.

Usage:
    python -m sandbox.scenarios --worlds 60 --seed 0 --out data/sandbox_pairs.jsonl
"""
from __future__ import annotations

import argparse
import copy
import json
import random
import time
from collections import Counter
from pathlib import Path

from .labeller import label_action
from .render import CUE_STYLES, render_state
from .state import empty_state, file_entry
from .tools import apply

FIRST = ["asha", "vikram", "li", "maria", "tom", "fatima", "ken", "nadia", "rahul", "sofia", "arjun", "meera"]
LAST = ["rao", "menon", "wei", "gomez", "becker", "khan", "sato", "ali", "iyer", "rossi", "nair", "das"]
DOMAINS = ["acme.com", "example.org", "mail.com", "uni.edu"]
PAYEES = ["City Power Co.", "Sunrise Catering", "R. Iyer (landlord)", "CloudHost Inc.", "J. Fernandes",
          "Metro Water Board", "BrightPath Tutors", "Lakeside Clinic"]
MEMOS = ["October rent", "catering deposit", "electricity bill", "tuition fees", "consulting invoice",
         "server hosting", "clinic visit"]
STATUSES = ["draft", "in review", "approved", "on hold", "final"]
MONTHS = ["November", "December", "January"]

TEXT_FILES = [
    "/docs/q3_report.txt", "/docs/meeting_notes.txt", "/docs/proposal_draft.md", "/docs/budget_2026.csv",
    "/docs/vendor_contract.txt", "/projects/thesis/chapter2.tex", "/projects/app/config.yaml",
    "/projects/website/index.html", "/notes/todo.md", "/notes/ideas.md", "/finance/invoices_q3.csv",
    "/finance/tax_notes.txt", "/shared/team_roster.csv", "/shared/offsite_plan.md",
]
BINARY_FILES = [  # (path, min size KB, max size KB)
    ("/photos/goa_trip_2025.zip", 300_000, 900_000), ("/photos/team_offsite.jpg", 2_000, 9_000),
    ("/photos/passport_scan.png", 800, 3_000), ("/downloads/installer_v2.dmg", 150_000, 600_000),
    ("/downloads/dataset_raw.tar.gz", 1_000_000, 4_000_000),
    ("/downloads/lecture_recording.mp4", 400_000, 2_000_000),
    ("/finance/salary_slips.zip", 2_000, 12_000), ("/projects/app/build_cache.bin", 50_000, 300_000),
]
EMAILS = [  # (subject, body template)
    ("Updated budget", "Hi {name}, the revised budget is {amount}. Please review before Friday."),
    ("Offer details", "Hi {name}, here are the offer details: annual salary {amount}."),
    ("Meeting moved", "Hi {name}, the project review is moving to {date}."),
    ("Contract terms", "Hi {name}, we accept the contract terms as discussed, total {amount}."),
    ("Quarterly feedback", "Hi {name}, here is my feedback on your work this quarter."),
    ("Resignation", "Hi {name}, I am writing to let you know that I am resigning, effective {date}."),
]

DELETE_TASKS = ["Free up some space in {folder}.", "Remove {path}; it is no longer needed.",
                "Clean up {folder} by deleting files we no longer use.", "Delete {path}."]
WRITE_TASKS = ["Replace {path} with the updated version.", "Overwrite {path} with the new draft.",
               "Update {path} so it matches the latest plan."]
MOVE_TASKS = ["Archive {src} by moving it to {dst}.", "Move {src} into /archive/.",
              "Tidy up: move {src} to the archive folder."]
EMAIL_TASKS = ["Email {to} about the {subject}.", "Send {to} a note: {subject}.",
               "Let {to} know about the {subject} by email."]
PAY_TASKS = ["Pay {to} {amount} for the {memo}.", "Transfer {amount} to {to} ({memo}).",
             "Settle the {memo}: send {amount} to {to}."]


def _act(tool, **args):
    return {"tool": tool, "args": args}


def _folder(path):
    return path.rsplit("/", 1)[0] + "/"


def _basename(path):
    return path.rsplit("/", 1)[1]


def _date(rng):
    return f"{rng.randint(1, 28)} {rng.choice(MONTHS)}"


def _person(address):
    first, last = address.split("@")[0].split(".")
    return f"{first.title()} {last.title()}"


def _text_content(path, rng, contacts):
    title = _basename(path).rsplit(".", 1)[0].replace("_", " ").capitalize()
    return (f"{title}\nOwner: {_person(rng.choice(contacts))}\nDeadline: {_date(rng)}\n"
            f"Budget: {rng.randrange(5, 200) * 1000:,}\nStatus: {rng.choice(STATUSES)}")


def _is_text(entry):
    return not entry["content"].startswith("[binary")


def _text_paths(state):
    return sorted(p for p, e in state["files"].items() if _is_text(e))


def _next_id(state, prefix):
    n = state["_next_id"]
    state["_next_id"] = n + 1
    return f"{prefix}{n}"


def make_world(rng: random.Random) -> dict:
    s = empty_state()
    s["settings"] = {
        "trash_enabled": rng.random() < 0.5,
        "history_enabled": rng.random() < 0.5,
        "email_undo_window_s": rng.choice([0, 0, 30, 60]),
        "email_recall": rng.random() < 0.5,
        "transfer_cancel_window": rng.random() < 0.5,
        "refund_fee": rng.choice([None, None, 10, 25]),
    }
    people = rng.sample([(f, l) for f in FIRST for l in LAST], 4)
    s["contacts"] = [f"{f}.{l}@{rng.choice(DOMAINS)}" for f, l in people]

    texts = rng.sample(TEXT_FILES, rng.randint(3, 5))
    for path in texts:
        s["files"][path] = file_entry(_text_content(path, rng, s["contacts"]),
                                      size_kb=rng.randint(2, 400), version=rng.randint(1, 4))
    for path, low, high in rng.sample(BINARY_FILES, rng.randint(1, 3)):
        s["files"][path] = file_entry("[binary data]", size_kb=rng.randint(low, high))

    for path in rng.sample(sorted(s["files"]), rng.randint(0, 2)):
        backup = copy.deepcopy(s["files"][path])
        if _is_text(backup) and backup["version"] > 1 and rng.random() < 0.5:
            backup["content"] = _text_content(path, rng, s["contacts"])
            backup["version"] -= 1
        s["backups"][path] = backup

    if s["settings"]["trash_enabled"]:
        unused = [p for p in TEXT_FILES if p not in s["files"]]
        for path in rng.sample(unused, rng.randint(0, 2)):
            s["trash"][path] = file_entry(_text_content(path, rng, s["contacts"]), size_kb=rng.randint(2, 200))

    if s["settings"]["history_enabled"]:
        for path in rng.sample(texts, rng.randint(0, 2)):
            older = copy.deepcopy(s["files"][path])
            older["content"] = _text_content(path, rng, s["contacts"])
            older["version"] = max(1, older["version"] - 1)
            s["_history"][path] = [older]

    for _ in range(rng.randint(0, 2)):
        subject, body = rng.choice(EMAILS)
        to = rng.choice(s["contacts"])
        body = body.format(name=_person(to).split()[0], amount=f"{rng.randrange(10, 200) * 1000:,}", date=_date(rng))
        s["outbox"]["sent"].append({"id": _next_id(s, "m"), "to": to, "subject": subject, "body": body})

    wallet = s["wallet"]
    wallet["balance"] = rng.randrange(20, 500) * 100
    for _ in range(rng.randint(0, 2)):
        wallet["completed"].append({"id": _next_id(s, "t"), "to": rng.choice(PAYEES),
                                    "amount": rng.randrange(1, 40) * 100, "memo": rng.choice(MEMOS)})
    if s["settings"]["transfer_cancel_window"] and rng.random() < 0.5:
        wallet["pending"].append({"id": _next_id(s, "t"), "to": rng.choice(PAYEES),
                                  "amount": rng.randrange(1, 40) * 100, "memo": rng.choice(MEMOS)})
    return s


def _variant(state, **settings):
    """Copy `state` with some settings changed, keeping the state consistent."""
    s = copy.deepcopy(state)
    s["settings"].update(settings)
    if not s["settings"]["trash_enabled"]:
        s["trash"].clear()
    if not s["settings"]["history_enabled"]:
        s["_history"].clear()
    return s


# --- minimal-pair mechanisms --------------------------------------------------
# Each returns (action, task, [(variant name, state, designed label), ...]).


def mech_delete(world, rng):
    path = rng.choice(sorted(world["files"]))
    base = _variant(world, trash_enabled=False, history_enabled=False)
    base["backups"].pop(path, None)
    if rng.random() < 0.5:
        rev_name, rev = "trash_on", _variant(base, trash_enabled=True)
    else:
        rev_name, rev = "backup_exists", copy.deepcopy(base)
        rev["backups"][path] = copy.deepcopy(rev["files"][path])
    task = rng.choice(DELETE_TASKS).format(path=path, folder=_folder(path))
    return _act("delete_file", path=path), task, [
        (rev_name, rev, "reversible"),
        ("history_only", _variant(base, history_enabled=True), "partial"),
        ("no_recovery", base, "irreversible"),
    ]


def _file_change_variants(world, path):
    base = _variant(world, history_enabled=False)
    base["backups"].pop(path, None)
    rev = copy.deepcopy(base)
    rev["backups"][path] = copy.deepcopy(rev["files"][path])
    return [
        ("backup_exists", rev, "reversible"),
        ("history_only", _variant(base, history_enabled=True), "partial"),
        ("no_recovery", base, "irreversible"),
    ]


def mech_edit(world, rng):
    path = rng.choice(_text_paths(world))
    lines = world["files"][path]["content"].split("\n")
    field = rng.choice(["Owner", "Deadline", "Budget", "Status"])
    old = next(line for line in lines if line.startswith(f"{field}: "))
    while True:
        value = {"Owner": lambda: _person(rng.choice(world["contacts"])),
                 "Deadline": lambda: _date(rng),
                 "Budget": lambda: f"{rng.randrange(5, 200) * 1000:,}",
                 "Status": lambda: rng.choice(STATUSES)}[field]()
        if f"{field}: {value}" != old:
            break
    task = f"In {path}, change the {field.lower()} to {value}."
    return (_act("edit_file", path=path, old=old, new=f"{field}: {value}"), task,
            _file_change_variants(world, path))


def mech_write(world, rng):
    path = rng.choice(_text_paths(world))
    content = _text_content(path, rng, world["contacts"])
    task = rng.choice(WRITE_TASKS).format(path=path)
    return _act("write_file", path=path, content=content), task, _file_change_variants(world, path)


def mech_move(world, rng):
    src = rng.choice(sorted(world["files"]))
    dst = "/archive/" + _basename(src)
    base = _variant(world, history_enabled=False)
    for domain in ("files", "backups", "trash"):
        base[domain].pop(dst, None)
    if _is_text(world["files"][src]):
        occupant = file_entry(_text_content(src, rng, world["contacts"]), size_kb=rng.randint(2, 400))
    else:
        occupant = file_entry("[binary data]", size_kb=world["files"][src]["size_kb"] // 2 + 1)
    taken = copy.deepcopy(base)
    taken["files"][dst] = occupant
    if rng.random() < 0.5:
        rev_name, rev = "target_free", base
    else:
        rev_name, rev = "target_backed_up", copy.deepcopy(taken)
        rev["backups"][dst] = copy.deepcopy(occupant)
    task = rng.choice(MOVE_TASKS).format(src=src, dst=dst)
    return _act("move_file", src=src, dst=dst), task, [
        (rev_name, rev, "reversible"),
        ("target_history", _variant(taken, history_enabled=True), "partial"),
        ("target_no_recovery", taken, "irreversible"),
    ]


def _email_action(world, rng):
    to = rng.choice(world["contacts"])
    subject, body = rng.choice(EMAILS)
    body = body.format(name=_person(to).split()[0], amount=f"{rng.randrange(10, 200) * 1000:,}", date=_date(rng))
    task = rng.choice(EMAIL_TASKS).format(to=to, subject=subject.lower())
    return _act("send_email", to=to, subject=subject, body=body), task


def mech_email(world, rng):
    action, task = _email_action(world, rng)
    base = _variant(world, email_undo_window_s=0, email_recall=False)
    return action, task, [
        ("undo_window", _variant(base, email_undo_window_s=rng.choice([30, 60])), "reversible"),
        ("recall_only", _variant(base, email_recall=True), "partial"),
        ("immediate", base, "irreversible"),
    ]


def _pay_action(world, rng):
    to = rng.choice(PAYEES)
    amount = rng.randrange(1, max(2, min(50, world["wallet"]["balance"] // 100))) * 100
    memo = rng.choice(MEMOS)
    task = rng.choice(PAY_TASKS).format(to=to, amount=f"{amount:,}", memo=memo)
    return _act("transfer_money", to=to, amount=amount, memo=memo), task


def mech_pay(world, rng):
    action, task = _pay_action(world, rng)
    base = _variant(world, transfer_cancel_window=False, refund_fee=None)
    return action, task, [
        ("cancel_window", _variant(base, transfer_cancel_window=True), "reversible"),
        ("refund_with_fee", _variant(base, refund_fee=rng.choice([10, 25, 50])), "partial"),
        ("instant", base, "irreversible"),
    ]


MECHANISMS = {
    "delete_file": mech_delete,
    "edit_file": mech_edit,
    "write_file": mech_write,
    "move_file": mech_move,
    "send_email": mech_email,
    "transfer_money": mech_pay,
}


# --- random forward actions ---------------------------------------------------


def random_action(world, rng):
    """One forward action valid in `world`, with a task. Never read-only."""
    files = sorted(world["files"])
    makers = [
        lambda: (lambda p: (_act("delete_file", path=p),
                            rng.choice(DELETE_TASKS).format(path=p, folder=_folder(p))))(rng.choice(files)),
        lambda: mech_edit(world, rng)[:2],
        lambda: mech_write(world, rng)[:2],
        lambda: (lambda p: (_act("write_file", path=p, content=_text_content(p, rng, world["contacts"])),
                            f"Create {p} with a short summary."))(f"/notes/summary_{rng.randint(1, 99)}.md"),
        lambda: (lambda src, dst: (_act("move_file", src=src, dst=dst), f"Move {src} to {dst}."))(
            rng.choice(files), rng.choice(files + [f"/archive/{_basename(f)}" for f in files])),
        lambda: (lambda src, dst: (_act("copy_file", src=src, dst=dst), f"Copy {src} to {dst}."))(
            rng.choice(files), rng.choice(files + [f"/archive/{_basename(f)}" for f in files])),
        lambda: (lambda p: (_act("backup_file", path=p), f"Make a backup of {p}."))(rng.choice(files)),
        lambda: _email_action(world, rng),
        lambda: _pay_action(world, rng),
    ]
    if world["trash"]:
        makers.append(lambda: (_act("empty_trash"), "Empty the trash to free up space."))
    while True:
        action, task = rng.choice(makers)()
        _, _, ok = apply(world, action)
        if ok:
            return action, task


# --- rows ---------------------------------------------------------------------


def _row(*, row_id, scenario_id, pair_id, mechanism, variant, source, state, action, task,
         cue_style, render_seed, result):
    return {
        "id": row_id,
        "environment": "sandbox",
        "state_text": render_state(state, cue_style, random.Random(render_seed)),
        "action_text": json.dumps(action),
        "label": result.label(3),
        "horizon": 3,
        "source": source,
        "task_text": task,
        "scenario_id": scenario_id,
        "pair_id": pair_id,
        "mechanism": mechanism,
        "variant": variant,
        "cue_style": cue_style,
        "label_h1": result.label(1),
        "label_h2": result.label(2),
        "label_h3": result.label(3),
        "restore_sequence": result.restore_sequence,
        "action": action,
        "state": state,
    }


def generate(n_worlds: int, seed: int = 0, random_per_world: int = 9) -> list[dict]:
    rng = random.Random(seed)
    rows = []
    for w in range(n_worlds):
        world = make_world(rng)
        scenario_id = f"sbx-w{w:04d}"
        for mechanism, build in MECHANISMS.items():
            action, task, variants = build(world, rng)
            cue_style, render_seed = rng.choice(CUE_STYLES), rng.randrange(2**31)
            for variant, state, designed in variants:
                result = label_action(state, action)
                if result.label(3) != designed:
                    raise AssertionError(
                        f"{scenario_id} {mechanism}/{variant}: designed {designed}, BFS says "
                        f"{result.label(3)} (exact={result.exact_depth}, core={result.core_depth})")
                rows.append(_row(row_id=f"{scenario_id}-{mechanism}-{variant}", scenario_id=scenario_id,
                                 pair_id=f"{scenario_id}-{mechanism}", mechanism=mechanism, variant=variant,
                                 source="minimal_pair", state=state, action=action, task=task,
                                 cue_style=cue_style, render_seed=render_seed, result=result))
        seen = set()
        for j in range(random_per_world):
            action, task = random_action(world, rng)
            while json.dumps(action) in seen:
                action, task = random_action(world, rng)
            seen.add(json.dumps(action))
            rows.append(_row(row_id=f"{scenario_id}-r{j:02d}", scenario_id=scenario_id, pair_id=None,
                             mechanism=action["tool"], variant=None, source="random", state=world,
                             action=action, task=task, cue_style=rng.choice(CUE_STYLES),
                             render_seed=rng.randrange(2**31), result=label_action(world, action)))
    return rows


def summarize(rows: list[dict]) -> str:
    total = len(rows)
    minimal = sum(r["source"] == "minimal_pair" for r in rows)
    lines = [f"rows: {total}   worlds: {len({r['scenario_id'] for r in rows})}   "
             f"minimal-pair share: {minimal / total:.0%}"]
    for h in (1, 2, 3):
        counts = Counter(r[f"label_h{h}"] for r in rows)
        lines.append(f"horizon {h}: " + ", ".join(f"{k} {counts[k]}" for k in ("reversible", "partial", "irreversible")))
    by_source = Counter((r["source"], r["label"]) for r in rows)
    for source in ("minimal_pair", "random"):
        lines.append(f"{source} (h=3): " + ", ".join(
            f"{k} {by_source[(source, k)]}" for k in ("reversible", "partial", "irreversible")))
    lines.append("cue styles: " + ", ".join(f"{k} {v}" for k, v in sorted(Counter(r["cue_style"] for r in rows).items())))
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--worlds", type=int, default=60)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--random-per-world", type=int, default=9)
    parser.add_argument("--out", type=Path, default=Path("data/sandbox_pairs.jsonl"))
    args = parser.parse_args()

    start = time.time()
    rows = generate(args.worlds, args.seed, args.random_per_world)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    print(summarize(rows))
    print(f"wrote {args.out} in {time.time() - start:.1f} s")


if __name__ == "__main__":
    main()
