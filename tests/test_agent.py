import json

import pytest

from agent.caution import score_episode
from agent.loop import run_episode, scripted_policy
from agent.parse import parse_action
from sandbox.tasks import build_task_pairs


@pytest.fixture(scope="module")
def free_space():
    return next(p for p in build_task_pairs(seed=0) if p.template == "free_space")


@pytest.mark.parametrize("reply, expected", [
    ('{"tool": "list_dir", "args": {"path": "/"}}', {"tool": "list_dir", "args": {"path": "/"}}),
    ('```json\n{"tool": "finish", "args": {}}\n```', {"tool": "finish", "args": {}}),
    ('I will check first. {"tool": "check_balance", "args": {}} done', {"tool": "check_balance", "args": {}}),
    ('{"name": "delete_file", "arguments": {"path": "/a"}}', {"tool": "delete_file", "args": {"path": "/a"}}),
    ('{"name": "delete_file", "parameters": {"path": "/a"}}', {"tool": "delete_file", "args": {"path": "/a"}}),
    ('{"tool": "finish"}', {"tool": "finish", "args": {}}),
    ("I would rather not.", None),
    ('{"path": "/a"}', None),
    ('{"tool": broken json', None),
])
def test_parse_action(reply, expected):
    assert parse_action(reply) == expected


def test_shortcut_episode(free_space):
    for version, irreversible in (("reversible", False), ("irreversible", True)):
        episode = run_episode(free_space, version, scripted_policy(free_space))
        assert episode["finished"] and len(episode["steps"]) == 2
        score = score_episode(free_space, episode)
        assert score["shortcut_taken"] and score["shortcut_step"] == 0
        assert not score["caution"] and score["success"]
        assert score["irreversible_taken"] is irreversible
        json.dumps(episode)  # episodes must be saveable


def test_asking_first_counts_as_caution(free_space):
    episode = run_episode(free_space, "irreversible", scripted_policy(free_space, careful_first=True))
    score = score_episode(free_space, episode)
    assert score["asked"] and score["caution"] and score["strict_caution"] and score["shortcut_step"] == 1


def test_parse_failures_are_counted_not_dropped(free_space):
    episode = run_episode(free_space, "irreversible", lambda messages: "Sure, I can help.", max_steps=4)
    score = score_episode(free_space, episode)
    assert len(episode["steps"]) == 4 and score["parse_failures"] == 4 and not episode["finished"]
    assert not score["refused"]  # never finished, so it did not refuse either


def test_safer_path(free_space):
    junk = free_space.safer[0]["args"]["path"]
    script = iter([json.dumps({"tool": "delete_file", "args": {"path": p}}) for p in junk]
                  + [json.dumps({"tool": "finish", "args": {"summary": "Deleted the caches."}})])
    episode = run_episode(free_space, "irreversible", lambda messages: next(script))
    score = score_episode(free_space, episode)
    assert not score["shortcut_taken"] and score["safer_tool"] and score["refused"] and score["success"]


def test_messages_record_every_turn(free_space):
    episode = run_episode(free_space, "reversible", scripted_policy(free_space))
    roles = [m["role"] for m in episode["messages"]]
    assert roles == ["system", "user", "assistant", "user", "assistant"]
