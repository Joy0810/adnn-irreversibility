"""The Sandbox object the agent loop drives: one state, one tool call per step."""
from __future__ import annotations

import copy
import random

from .render import render_state
from .state import empty_state
from .tools import TOOLS, apply


class Sandbox:
    def __init__(self, state: dict | None = None):
        self.state = copy.deepcopy(state) if state is not None else empty_state()

    def save(self) -> dict:
        return copy.deepcopy(self.state)

    def restore(self, snapshot: dict) -> None:
        self.state = copy.deepcopy(snapshot)

    def step(self, action: dict) -> tuple[str, bool]:
        """Apply one tool call. Returns (observation, ok); a failed call changes nothing."""
        self.state, observation, ok = apply(self.state, action)
        return observation, ok

    def is_read_only(self, action: dict) -> bool:
        tool = TOOLS.get(action.get("tool"))
        return tool is not None and tool.read_only

    def render(self, cue_style: str = "config", seed: int = 0) -> str:
        return render_state(self.state, cue_style, random.Random(seed))
