# Do AI Agents Know Which Actions Cannot Be Undone?

ADNN project, Option 1. Team: Pranav, Joy, Gatri, Nishal.

We measure whether open-weight LLMs (Qwen2.5-7B-Instruct, Llama-3.1-8B-Instruct) represent whether an action can be undone, whether they say so when asked, and whether they act on it as agents. Plan and schedule: [team plan doc](https://claude.ai/artifact/SFbppDBwMoaam5hLDpYtbh).

## Layout

| Path | What | Owner |
| --- | --- | --- |
| `sandbox/` | Simulated file system, outbox and wallet; BFS reversibility labeller; minimal-pair generator | Joy |
| `sciworld/` | ScienceWorld replay-based labeller | Gatri |
| `models/` | Model loading, prompts, hidden-state extraction, steering hooks | Pranav |
| `probes/` | Linear probes, baselines, control tasks | Pranav |
| `agent/` | Agent loop, tool-call parser, caution detector | Nishal |
| `analysis/` | Statistics and one script per figure | Nishal |
| `data/` | Labelled pairs (JSON lines), splits, hand labels | Joy, Gatri |
| `results/` | Raw outputs, agent logs, `RUNLOG.md` | everyone |
| `docs/` | Design specs | |
| `notes/` | Paper summaries (Group B) | everyone |
| `tests/` | Unit tests | |

## Quick start (sandbox, no GPU needed)

The sandbox uses only the Python standard library (3.10+). The tests need pytest.

```bash
python -m pytest -q                                   # 38 tests, under a second
python -m sandbox.scenarios --worlds 60 --seed 0      # writes data/sandbox_pairs.jsonl (1,620 rows)
```

```python
from sandbox import Sandbox, label_action

box = Sandbox(row["state"])                           # any row from data/sandbox_pairs.jsonl
print(box.render("config"))                           # state text the model sees
obs, ok = box.step({"tool": "delete_file", "args": {"path": "/docs/q3_report.txt"}})
result = label_action(row["state"], row["action"])    # BFS label at horizons 1 to 3
```

## Dataset format

One JSON object per line. The 7 fields required by the brief come first:

| Field | Meaning |
| --- | --- |
| `id`, `environment`, `state_text`, `action_text`, `label`, `horizon`, `source` | Required. `label` uses horizon 3; `source` is `minimal_pair` or `random` |
| `task_text` | The user instruction the action answers; identical across a minimal pair |
| `scenario_id` | World id. Split train/test by this field, never by row |
| `pair_id`, `mechanism`, `variant` | Minimal-pair group, the action type, and which variant this row is |
| `cue_style` | `config`, `structural` or `narrative`: how the settings are written |
| `label_h1`, `label_h2`, `label_h3` | Labels at each horizon (for the horizon ablation) |
| `restore_sequence` | Shortest undo the labeller found |
| `action`, `state` | Structured action and full state, to re-run or re-render |

## Environment

Pin exact versions on the GPU machine once everything runs: `pip install -r requirements.in`, then `pip freeze > requirements.txt` and commit it. Fixed settings for every experiment: greedy decoding, seeds fixed, every prompt and output saved under `results/`.

## Safety

Every irreversible action happens inside the simulated sandbox. Never connect the agent to real email, real files or real accounts, and use no paid APIs for reported results.
