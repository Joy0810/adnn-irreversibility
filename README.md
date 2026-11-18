# Do AI Agents Know Which Actions Cannot Be Undone?

ADNN project, Option 1. Team: Pranav, Joy, Gatri, Nishal.

We measure whether open-weight LLMs (Qwen2.5-7B-Instruct, Llama-3.1-8B-Instruct) represent whether an action can be undone (RQ1), whether they say so when asked (RQ2), whether they act on it as agents (RQ3), and whether pushing the representation changes behaviour (RQ4). Plan and schedule: [team plan doc](https://claude.ai/artifact/SFbppDBwMoaam5hLDpYtbh). Analysis rules: [ANALYSIS_PLAN.md](ANALYSIS_PLAN.md).

**All compute runs on the GPU machine, run by Joy, following the runbook below.** Nothing needs to run on a laptop.

## Layout

| Path | What |
| --- | --- |
| `sandbox/` | Simulated file system, outbox and wallet; BFS reversibility labeller; minimal-pair generator (`scenarios.py`); the 60 behaviour task pairs (`tasks.py`) |
| `sciworld/` | ScienceWorld testbed: replay-based labeller and data generator ([spec](docs/sciworld_spec.md)) |
| `models/` | Prompts shared by every experiment, model loading, hidden-state extraction (E1), steering hook |
| `probes/` | Scenario-level splits and per-layer probes with every control (E2) |
| `agent/` | Tool-call parser, episode loop, caution scoring |
| `experiments/` | Stated answers (E3), agent runs (E4, E5), steering (E6, E7) |
| `analysis/` | Statistics, and every figure and table in the report |
| `data/` | Labelled pairs (JSON lines), splits, hand labels |
| `results/` | Everything the runs write; `RUNLOG.md` |
| `docs/` | Design specs: [sandbox](docs/sandbox_spec.md), [ScienceWorld](docs/sciworld_spec.md) |
| `tests/` | Unit tests; run them first |

## Runbook (Joy, on the GPU machine)

Work through the steps in order. Do everything on Qwen first, then repeat for Llama. After each step, add a line to `results/RUNLOG.md` (date, command, output path, one-line result), then commit and push. Run long steps inside `tmux` or with `nohup ... > logs/<name>.log 2>&1 &`, so they survive a dropped connection. Every long step resumes where it stopped if you run the same command again.

### Step 0: set up (once)

```bash
git clone https://github.com/sunkenship2025/adnn-irreversibility.git
cd adnn-irreversibility
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.in
pip freeze > requirements.txt            # exact versions for the report: commit this file
huggingface-cli login                    # or `hf auth login`; the token must have Llama 3.1 access
java -version                            # ScienceWorld needs Java 1.8 or newer
nvidia-smi                               # about 50 GB of GPU memory should be free
mkdir -p logs
```

Disk: about 16 GB per model in the Hugging Face cache (set `HF_HOME` if the home quota is small), and about 3 GB of hidden states per model under `results/hidden/` (not committed).

### Step 1: tests (CPU, under a minute)

```bash
python -m pytest -q
```

Everything must pass. If anything fails, stop and send the output to Pranav before running anything else.

### Step 2: data (CPU)

```bash
python -m sandbox.scenarios --worlds 60 --seed 0          # data/sandbox_pairs.jsonl, 1,620 rows; reproduces the committed file
```

ScienceWorld, in this order. What to check in each output is listed in [docs/sciworld_spec.md](docs/sciworld_spec.md) under "Smoke test".

```bash
python -m sciworld.env --task boil --variation 0      # must say scienceworld 1.3.0 and "deterministic replay: True"; if False, stop
python -m sciworld.generate --tasks boil,find-living-thing --variations 1 --states 3 --per-state 4 \
    --limit 20 --workers 1 --target 0 --out data/sciworld_smoke.jsonl   # then read 5 rows: does restore_sequence undo the action?
python -m sciworld.generate --variations 5 --states 8 --per-state 4 --workers 16 --target 800
```

The full run writes `data/sciworld_pairs.jsonl` (800 rows, balanced) and `data/sciworld_pairs_raw.jsonl` (every row). It is estimated at about 35 minutes with 16 workers; each worker runs its own Java process and needs about 1 GB of memory, so set `--workers` to fit the machine. If it stops, rerun with `--resume`. If fast stepping fails, add `--slow-step`.

Then the splits, which every later step uses:

```bash
python -m probes.splits --rows data/sandbox_pairs.jsonl data/sciworld_pairs.jsonl     # data/splits/seed{0,1,2}.json
python -m experiments.agent_runs --model dry-run --limit 3                           # agent plumbing, no model, seconds
```

From here on, `ROWS="data/sandbox_pairs.jsonl data/sciworld_pairs.jsonl"`.

### Step 3: Qwen2.5-7B (GPU)

Run each step on a few rows first, then for real.

| Exp | Command | Writes | Rough time |
| --- | --- | --- | --- |
| E1 smoke | `python -m models.extract --model qwen7b --rows $ROWS --limit 20 --out results/hidden/smoke` | 20 rows | 2 min |
| E1 | `python -m models.extract --model qwen7b --rows $ROWS` | `results/hidden/qwen7b/` | 10-20 min |
| E2 | `python -m probes.train --model qwen7b --rows $ROWS` | `results/probes/qwen7b/` | 30-90 min, CPU |
| E3 smoke | `python -m experiments.stated --model qwen7b --rows $ROWS --limit 20 --out results/stated/smoke` | 20 rows + 120 pair questions | 5 min |
| E3 | `python -m experiments.stated --model qwen7b --rows $ROWS` | `results/stated/qwen7b/` | 1-2 h |
| E4 smoke | `python -m experiments.agent_runs --model qwen7b --limit 3 --out results/agent/smoke` | 6 episodes | 5 min |
| E4 | `python -m experiments.agent_runs --model qwen7b` | `results/agent/qwen7b/neutral/` | 1-2 h |
| E5 | `python -m experiments.agent_runs --model qwen7b --careful` | `results/agent/qwen7b/careful/` | 1-2 h |

Check after E2: `results/probes/qwen7b/best_layer.json` names a layer, and `results.csv` has rows for every layer. Check after E4: the summary printed at the end shows the shortcut rate. If the shortcut rate is under 20% in both versions, tell the team before E5 (see Risks in the plan).

### Step 4: Llama-3.1-8B (GPU)

The same commands with `--model llama8b`. Llama is gated: if loading fails with a 401 or 403 error, the licence has not been approved for the logged-in account yet.

### Step 5: horizon ablation (CPU)

```bash
for m in qwen7b llama8b; do
  python -m probes.train --model $m --rows $ROWS --label-field label_h1 --positions last
  python -m probes.train --model $m --rows $ROWS --label-field label_h2 --positions last
done
```

### Step 6: steering, RQ4 (GPU, bonus; only after the Nov 5 gate)

```bash
python -m experiments.steering --model qwen7b --mode decision                    # E6, 20-40 min
python -m experiments.steering --model qwen7b --mode rollouts --alphas -2 0 2    # E7, 5-8 h
```

Run E7 on a few pairs first (`--limit 3`). If replies stop parsing at alpha = 2, use smaller alphas (for example `-1 0 1`) and note it in the run log.

### Step 7: figures and tables (CPU)

```bash
python -m analysis.report --models qwen7b llama8b --rows $ROWS
```

This writes `results/figures/` and `results/tables/`, and can be run at any point: missing parts are skipped. Commit `results/` (except `results/hidden/`, which git ignores).

### Kaggle (2x T4) instead of the lab GPU

Data (Step 2) is made on a laptop first; Kaggle only runs the models. T4s have no bfloat16, so `models/load.py` uses float16 there.

1. Zip this folder (with `data/`) and upload it as a private Kaggle dataset, e.g. `adnn-repo`.
2. New notebook: Accelerator `GPU T4 x2`, Internet on, add the dataset, add a secret `HF_TOKEN` (for Llama: an HF token whose account has Llama 3.1 approved).
3. One cell, then "Save Version" → "Save & Run All" so it runs in the background (12 h limit) and keeps `/kaggle/working`:

```python
import os
from kaggle_secrets import UserSecretsClient
os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("HF_TOKEN")
!cp -r /kaggle/input/adnn-repo/adnn-irreversibility /kaggle/working/
%cd /kaggle/working/adnn-irreversibility
!pip install -q statsmodels
!bash kaggle_run.sh qwen7b main
```

Run `main` and `stated` as two separate notebook versions per model. Download `results/` from the notebook's Output tab into this folder, then run Step 7 on the laptop.

### If something breaks

| Symptom | Likely cause and fix |
| --- | --- |
| CUDA out of memory | Another job is on the GPU (`nvidia-smi`). Never load two models at once. For the 14B model use `--limit` and short runs |
| `the chat template changed the action text` (E1) | The tokenizer's chat template rewrote the action. Send the row id to Pranav |
| Steering error in the forward hook | This transformers version returns something new from a decoder layer. Send the traceback |
| Many parse failures in E4 | Look at a few `reply` fields in the episode files; they are counted and reported, never dropped |
| A step was interrupted | Run the same command again; E1, E3, E4, E5 and E7 resume |

## Dataset format

One JSON object per line. The 7 fields required by the brief come first:

| Field | Meaning |
| --- | --- |
| `id`, `environment`, `state_text`, `action_text`, `label`, `horizon`, `source` | Required. `label` uses horizon 3 |
| `task_text` | The instruction the action answers; identical across a minimal pair |
| `scenario_id` | World or ScienceWorld variation. Splits use this field, never single rows |
| `pair_id`, `mechanism`, `variant` | Minimal-pair group, the action verb, which variant |
| `cue_style` | `config`, `structural` or `narrative` (sandbox only) |
| `label_h1`, `label_h2`, `label_h3` | Labels at each horizon |
| `restore_sequence` | Shortest undo the labeller found |
| `action`, `state` / `replay` | Enough to re-run the row: the sandbox state, or the ScienceWorld task, variation and action prefix |
| `search` | ScienceWorld only: search depth, nodes, whether it hit the node cap, seconds |

## Safety

Every irreversible action happens inside the simulated sandbox. Never connect the agent to real email, real files or real accounts, and use no paid APIs for reported results.
