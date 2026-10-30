"""Run one agent episode in the sandbox (RQ3).

The model sees the system prompt and the task with the rendered state, then
replies with one JSON tool call per turn; each observation comes back as the
next user turn. The episode ends at `finish` or after `max_steps` turns.

Every state-changing call is labelled by the BFS labeller in the state where it
was taken, so the log records exactly which executed actions were irreversible.
The full message list is saved with the episode: every prompt and every output.
"""
from __future__ import annotations

import json
from typing import Callable

from models.prompts import agent_messages
from sandbox import Sandbox, label_action
from sandbox.tasks import TaskPair

from .parse import parse_action

PARSE_ERROR = 'Error: reply with exactly one JSON tool call, for example {"tool": "list_dir", "args": {"path": "/"}}.'

Generate = Callable[[list], str]


def run_episode(pair: TaskPair, version: str, generate: Generate, careful: bool = False,
                max_steps: int = 15) -> dict:
    box = Sandbox(pair.states[version])
    messages = agent_messages(pair.instruction, pair.state_text(version), careful=careful)
    steps, finished = [], False
    for index in range(max_steps):
        reply = generate(messages)
        messages.append({"role": "assistant", "content": reply})
        action = parse_action(reply)
        step = {"index": index, "reply": reply, "action": action, "parsed": action is not None,
                "ok": False, "observation": PARSE_ERROR, "label": None}
        if action is not None:
            before = box.state
            observation, ok = box.step(action)
            step.update(ok=ok, observation=observation)
            if ok and not box.is_read_only(action):
                step["label"] = label_action(before, action).label(3)
            finished = ok and action["tool"] == "finish"
        steps.append(step)
        if finished:
            break
        messages.append({"role": "user", "content": f"Observation: {step['observation']}"})
    return {"task_id": pair.id, "template": pair.template, "domain": pair.domain, "version": version,
            "careful": careful, "finished": finished, "steps": steps, "final_state": box.state,
            "messages": messages}


def scripted_policy(pair: TaskPair, careful_first: bool = False) -> Generate:
    """A fake model for dry runs without a GPU: optionally ask first, then take the shortcut, then finish."""
    script =([{"tool": "ask_user", "args": {"question": "Shall I go ahead?"}}] if careful_first else [])
    script += [pair.shortcut_example, {"tool": "finish", "args": {"summary": "Done."}}]
    replies = iter(json.dumps(a) for a in script)
    return lambda messages: next(replies, json.dumps({"tool": "finish", "args": {}}))
