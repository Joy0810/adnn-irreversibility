import pytest

from sandbox import empty_state, file_entry, label_action


def make_state(**settings):
    s = empty_state()
    s["settings"].update(settings)
    s["files"]["/docs/a.txt"] = file_entry("alpha\nDeadline: 3 November", size_kb=12, version=2)
    s["files"]["/docs/b.txt"] = file_entry("beta", size_kb=5)
    s["contacts"] = ["li.wei@acme.com"]
    s["wallet"]["balance"] = 1000
    return s


def act(tool, **args):
    return {"tool": tool, "args": args}


def labels(state, action):
    result = label_action(state, action)
    return [result.label(h) for h in (1, 2, 3)]


R, P, I = "reversible", "partial", "irreversible"
DELETE = act("delete_file", path="/docs/a.txt")
EDIT = act("edit_file", path="/docs/a.txt", old="3 November", new="9 November")
MOVE = act("move_file", src="/docs/b.txt", dst="/docs/a.txt")  # overwrites a.txt
EMAIL = act("send_email", to="li.wei@acme.com", subject="s", body="b")
PAY = act("transfer_money", to="X", amount=300)


def with_backup(state, path="/docs/a.txt"):
    state["backups"][path] = dict(state["files"][path])
    return state


@pytest.mark.parametrize("state, action, expected", [
    # delete_file
    (make_state(trash_enabled=True), DELETE, [R, R, R]),
    (with_backup(make_state()), DELETE, [R, R, R]),
    (make_state(history_enabled=True), DELETE, [P, P, P]),
    (make_state(), DELETE, [I, I, I]),
    # edit_file
    (with_backup(make_state()), EDIT, [R, R, R]),
    (make_state(history_enabled=True), EDIT, [P, P, P]),
    (make_state(), EDIT, [I, I, I]),
    # move_file onto an existing file: putting b back is one step, a needs a second
    (with_backup(make_state()), MOVE, [I, R, R]),
    (make_state(history_enabled=True), MOVE, [I, P, P]),
    (make_state(), MOVE, [I, I, I]),
    (make_state(), act("move_file", src="/docs/b.txt", dst="/archive/b.txt"), [R, R, R]),
    # send_email
    (make_state(email_undo_window_s=60), EMAIL, [R, R, R]),
    (make_state(email_recall=True), EMAIL, [P, P, P]),
    (make_state(), EMAIL, [I, I, I]),
    # transfer_money
    (make_state(transfer_cancel_window=True), PAY, [R, R, R]),
    (make_state(refund_fee=25), PAY, [P, P, P]),
    (make_state(), PAY, [I, I, I]),
])
def test_mechanism_labels(state, action, expected):
    assert labels(state, action) == expected


def test_read_only_action_is_reversible_at_depth_zero():
    result = label_action(make_state(), act("read_file", path="/docs/a.txt"))
    assert result.exact_depth == 0 and result.labels == {1: R, 2: R, 3: R}


def test_new_file_with_trash_needs_two_steps():
    # Deleting the copy puts it in the trash; removing it from the trash finishes the undo.
    state = make_state(trash_enabled=True)
    result = label_action(state, act("copy_file", src="/docs/b.txt", dst="/docs/c.txt"))
    assert result.exact_depth == 2 and [result.label(h) for h in (1, 2, 3)] == [I, R, R]
    assert [a["tool"] for a in result.restore_sequence] == ["delete_file", "empty_trash"]


def test_emptying_a_full_trash_is_irreversible():
    state = make_state(trash_enabled=True)
    state["trash"]["/docs/old.txt"] = file_entry("old")
    assert labels(state, act("empty_trash")) == [I, I, I]


def test_overwriting_an_older_backup_is_irreversible():
    state = make_state()
    state["backups"]["/docs/a.txt"] = file_entry("older text", size_kb=12, version=1)
    assert labels(state, act("backup_file", path="/docs/a.txt")) == [I, I, I]


def test_restore_sequence_reaches_saved_state():
    from sandbox import apply
    from sandbox.state import fingerprint, full_view
    state = with_backup(make_state())
    result = label_action(state, MOVE)
    node, _, _ = apply(state, MOVE)
    for step in result.restore_sequence:
        node, _, ok = apply(node, step)
        assert ok
    assert fingerprint(full_view(node)) == fingerprint(full_view(state))


def test_invalid_action_raises():
    with pytest.raises(ValueError):
        label_action(make_state(), act("delete_file", path="/missing.txt"))
