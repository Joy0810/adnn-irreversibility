"""Sandbox state: a plain dict, so it deep-copies, serialises and compares trivially.

Reversibility labels compare states through two views (see labeller.py):

- full view: everything except keys starting with "_" (internal bookkeeping:
  the file-history store and the id counter);
- core view: the full view minus the trace fields, which are the residue an
  approximate restore leaves behind: file version numbers, notifications other
  people received, and fees owed.

Getting the full view back means the action was undone exactly (reversible).
Getting only the core view back means it was undone approximately (partial).
"""
from __future__ import annotations

import json

FILE_DOMAINS = ("files", "trash", "backups")


def empty_state() -> dict:
    return {
        "settings": {
            "trash_enabled": False,
            "history_enabled": False,
            "email_undo_window_s": 0,  # 0 means messages are delivered at once
            "email_recall": False,
            "transfer_cancel_window": False,
            "refund_fee": None,  # None means refunds are not available
        },
        "files": {},  # path -> file entry
        "trash": {},  # original path -> file entry
        "backups": {},  # path -> file entry
        "outbox": {"queued": [], "sent": []},
        "wallet": {"balance": 0, "pending": [], "completed": [], "fees_owed": 0},
        "notifications": [],
        "contacts": [],
        "_history": {},  # path -> list of earlier file entries, oldest first
        "_next_id": 1,
    }


def file_entry(content: str, size_kb: int | None = None, version: int = 1) -> dict:
    if size_kb is None:
        size_kb = 1 + len(content.encode()) // 1024
    return {"content": content, "size_kb": size_kb, "version": version}


def full_view(state: dict) -> dict:
    return {k: v for k, v in state.items() if not k.startswith("_")}


def core_view(state: dict) -> dict:
    view = {k: v for k, v in full_view(state).items() if k != "notifications"}
    for domain in FILE_DOMAINS:
        view[domain] = {
            path: {k: v for k, v in entry.items() if k != "version"}
            for path, entry in state[domain].items()
        }
    view["wallet"] = {k: v for k, v in state["wallet"].items() if k != "fees_owed"}
    return view


def fingerprint(view: dict) -> str:
    return json.dumps(view, sort_keys=True, separators=(",", ":"))
