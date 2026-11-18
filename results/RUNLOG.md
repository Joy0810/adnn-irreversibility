# Run log

Log every experiment, including the ones that fail. The paper must report them. Newest first.

| Date | Who | Experiment | Model | Command or script | Output path | Result, one line | Kept in paper? |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2026-11-13 | Joy | Splits + hand-label sheets | none | `python -m probes.splits --rows ...`; `python -m analysis.human_labels sample ...` | `data/splits/`, `data/human_labels/` | test 476 / 454 / 450 rows for seeds 0-2; 100 rows per sheet | yes |
| 2026-11-13 | Joy | ScienceWorld pairs generated | none | `python -m sciworld.generate --workers 5 --target 800` (scienceworld 1.3.0, deterministic replay: True) | `data/sciworld_pairs.jsonl` | 1,167 raw, 648 balanced (h3: 267 R / 114 P / 267 I; partial too rare to reach 800); 0 searches truncated | yes (dataset) |
| 2026-11-13 | Joy | Plumbing smoke tests | Qwen2.5-0.5B (CPU) | E1/E3/E4/E6 with `--limit`; probes.train on random features | deleted | all steps run; probe on random features at chance (0.49) | no |
| 2026-10-30 | Pranav | Sandbox pairs generated | none | `python -m sandbox.scenarios --worlds 60 --seed 0` | `data/sandbox_pairs.jsonl` | 1,620 rows, 67% minimal pairs; h3: 645 reversible, 488 partial, 487 irreversible | yes (dataset) |
