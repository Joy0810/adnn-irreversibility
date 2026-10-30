"""Parse a model reply into one tool call.

The first JSON object in the reply that names a tool wins. Code fences and text
around the object are ignored. The native tool-call shapes of Qwen and Llama
({"name": ..., "arguments"/"parameters": ...}) are accepted too, because they
express the same single call. Anything else is a parse failure, which the loop
logs and counts; it is never silently dropped.
"""
from __future__ import annotations

import json
import re

_DECODER = json.JSONDecoder()


def parse_action(text: str) -> dict | None:
    for match in re.finditer(r"\{", text):
        try:
            obj, _ = _DECODER.raw_decode(text, match.start())
        except json.JSONDecodeError:
            continue
        if not isinstance(obj, dict):
            continue
        tool = obj.get("tool", obj.get("name"))
        if not isinstance(tool, str):
            continue
        args = obj.get("args", obj.get("arguments", obj.get("parameters", {})))
        return {"tool": tool, "args": args if isinstance(args, dict) else {}}
    return None
