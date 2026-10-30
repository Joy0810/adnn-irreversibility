import pytest

from models.extract import action_token_span
from models.prompts import agent_messages, parse_pair, parse_stated, probe_messages, stated_messages
from sandbox import TOOLS


@pytest.mark.parametrize("reply, label", [
    ("IRREVERSIBLE. Once sent, the email cannot be pulled back.", "irreversible"),
    ("Reversible: the file goes to the trash.", "reversible"),
    ("PARTIALLY - a fee applies.", "partial"),
    ("It is only partial, since a notice is sent.", "partial"),
    ("The action is irreversible.", "irreversible"),
    ("I am not sure.", None),
])
def test_parse_stated(reply, label):
    assert parse_stated(reply) == label


@pytest.mark.parametrize("reply, answer", [
    ("B. In B the trash is off.", "B"),
    ("A", "A"),
    ("Version A, because the outbox has no delay.", "A"),
    ("Answer: B", "B"),
    ("Neither.", None),
])
def test_parse_pair(reply, answer):
    assert parse_pair(reply) == answer


ROW = {"environment": "sandbox", "task_text": "Delete /a.", "state_text": "[files]\n/a  1 KB",
       "action_text": '{"tool": "delete_file", "args": {"path": "/a"}}'}


def test_probe_prompt_is_the_agent_first_turn_plus_action():
    agent = agent_messages(ROW["task_text"], ROW["state_text"])
    probe = probe_messages(ROW)
    assert probe[:2] == agent and probe[2] == {"role": "assistant", "content": ROW["action_text"]}
    assert all(name in agent[0]["content"] for name in TOOLS)


def test_careful_prompt_only_adds_one_instruction():
    neutral = agent_messages("t", "s")[0]["content"]
    careful = agent_messages("t", "s", careful=True)[0]["content"]
    assert careful.startswith(neutral) and len(careful) > len(neutral)


def test_stated_prompt_contains_action_and_state():
    content = stated_messages(ROW)[1]["content"]
    assert ROW["action_text"] in content and ROW["state_text"] in content


def test_action_token_span():
    # Tokens: "<s>", "{\"tool", "\": ", "\"x\"}", "</s>" over the text '<s>{"tool": "x"}</s>'
    offsets = [(0, 3), (3, 9), (9, 12), (12, 16), (16, 20)]
    assert action_token_span(offsets, 3, 16) == (1, 3)
    # Zero-length special tokens are skipped.
    assert action_token_span([(0, 0), (0, 5), (5, 5)], 0, 5) == (1, 1)
