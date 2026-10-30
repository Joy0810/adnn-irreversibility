# ScienceWorld spec

Status: implemented in `sciworld/`, not yet run against the real simulator (no local compute was allowed). The tests in `tests/test_sciworld.py` use a pure-Python fake. Change the code and this file together.

## Version and licence

Read on 2026-10-30, through the GitHub and PyPI pages listed under Sources.

- PyPI: the latest release is **1.2.3** (uploaded 2025-07-13, tag `1.2.3`, commit `5d569f9`). It depends on `py4j`. The README asks for Java 1.8+ and Python 3.8+.
- GitHub `main`: commit **`e8216d6`** (2026-08-20) is version **1.3.0** (commit `ad4f284`, "Bump version to 1.3.0"). 1.3.0 is not on PyPI. The repository ships the jar as `scienceworld/scienceworld.jar`.
- Licence: **Apache-2.0**, "Copyright 2022 Cognitive AI Lab and Allen Institute for Artificial Intelligence". Cite Wang, Jansen, Côté and Ammanabrolu, "ScienceWorld: Is your Agent Smarter than a 5th Grader?", EMNLP 2022, doi:10.18653/v1/2022.emnlp-main.775.

**Use 1.3.0.** Replay labelling assumes that the same commands after a reset reach the same state. Up to 1.2.x that was not guaranteed: issue #33 ("Heat transfer is stochastic") and PR #88 trace it to `EnvObject` defining `equals` without a matching `hashCode`, so object traversal order changed between runs. 1.3.0 sorts objects by UUID (merged as `0ff14f9`). The README warns that 1.3.0 changes physics trajectories, ambiguity resolution, gold paths and JSON ordering compared with 1.2.x. Install with:

```bash
pip install "scienceworld @ git+https://github.com/allenai/ScienceWorld@e8216d6"
```

`requirements.in` pins this commit. Either way, the generator checks every state by replaying it twice and skips states that differ (`nondeterministic` in its summary).

## What the Python API offers

`scienceworld/scienceworld.py`, class `ScienceWorldEnv`. Method names are the same in 1.2.3 and 1.3.0. Both versions have snake_case methods plus camelCase aliases; the aliases warn of deprecation on `main`. Before 1.2 only the camelCase names existed. The adapter tries both (`env._method`).

| Purpose | Call | Notes |
| --- | --- | --- |
| Start | `ScienceWorldEnv(taskName=None, serverPath=None, envStepLimit=100)` | Launches the bundled jar through py4j `launch_gateway(die_on_exit=True)` on dynamic ports. One JVM per object. |
| Load | `load(taskName, variationIdx=0, simplificationStr="", generateGoldPath=False)` | Takes a task name or id ("1-1"). Simplifications are comma-separated with no spaces, each in `["easy"] + get_possible_simplifications()`. `noElectricalAction` with a power-component or conductivity task raises `ValueError`. |
| Reset | `reset() -> (observation, info)` | The server reloads the same task, variation, simplifications and gold flag: `Random.setSeed(variationIdx)`, world rebuilt, UUID counter reset. Then it calls `step("look around")`. |
| Step | `step(input_str) -> (observation, reward, isCompleted, info)` | `reward` is the change in score. `isCompleted` is set when the server reports completion, when moves > `envStepLimit`, or when score < 0. |
| Info keys | `moves`, `score` (0 to 100), `reward`, `look`, `inv`, `taskDesc`, `valid`, `variationIdx`, `taskName`, `simplificationStr` | `valid` is every valid action, computed on every step. |
| Tasks | `get_task_names()`, properties `task_names` and `tasks` (task id to name) | 30 tasks. |
| Variations | `get_max_variations(task_name)`, `get_variations_train()`, `get_variations_dev()`, `get_variations_test()`, `get_random_variation_train()` (and dev, test) | The split lists refer to the loaded task. |
| Gold path | `get_gold_action_sequence()` | Needs `generateGoldPath=True` at load. Otherwise it returns `["ERROR: Gold path was not generated.  Set ..."]`. Generating it plays the task, then rebuilds the world with a second load. |
| Valid actions | `get_valid_action_object_combinations()` (sorted strings), `get_valid_action_object_combinations_with_templates()` (dicts with `action`, `template_id`, `obj_ids`) | Commands that parse for the objects the agent can see. |
| Possible actions | `get_possible_actions()` (templates), `get_possible_objects()`, `get_possible_object_referent_LUT()` | |
| Text | `look()`, `inventory()`, `taskdescription()`, `get_task_description()` | `look()`, `inventory()` and `taskdescription()` are "free actions" that never tick. |
| Time | `get_num_moves()` | Universe ticks so far. |
| Object tree | `getObjectTree()` (camelCase only, no snake_case) | Dumps the whole universe as JSON. The jar writes `objectTree.json` to a temp directory and Python runs `json.load` on it. |
| Other | `get_goal_progress()`, `get_possible_simplifications()`, `get_simplifications_used()`, `get_vocabulary()`, run-history methods (logs only), `close()` | |
| **Save, restore, clone** | **none** | Neither `scienceworld.py` nor the Java `PythonInterface` has one. So states are rebuilt by reset and replay. |

Simplifications: `teleportAction`, `openDoors`, `selfWateringFlowerPots`, `noElectricalAction`, `openContainers`. `easy` is the first four, with `noElectricalAction` only for tasks that are not about electrical connections. `openContainers` must be added by hand. The README notes that `easy` differs from the paper's Appendix B.5.

Commands and time, from the Scala source:

- Commands: `open`/`close X`, `activate`/`deactivate X`, `connect X to Y`, `disconnect X`, `use X [on Y]`, `look around`, `look at X` (also `examine`), `look in X`, `read X`, `move X to Y`, `pick up X`, `put down X`, `pour X into Y`, `dunk X into Y`, `mix X`, `go to L`, `teleport to L` (only with `teleportAction`), `eat X`, `flush X`, `focus on X`, `wait` (10), `wait1` (1), `task`, `inventory`, `reset task`.
- `look around`, `inventory` and `task` take no time (`ActionHandler.doesActionTakeTime`). Every other parsed command ticks the universe once per pass of the step loop. Input that does not parse does not tick and returns "No known action matches that input."
- `wait1` sets the agent's wait counter to 1. The step loop ticks once for the command and once more while the counter drains, so `wait1` probably takes **two** ticks. That is read from code, not tested. The adapter measures it at load time.
- An ambiguous command returns "Ambiguous request: Please enter the number for the action you intended (or blank to cancel):" with options numbered from 0, and the next input must be the number.
- `look at` and `look in` change nothing (`ActionLook.scala`). `pour X into Y` moves X's contents into Y, and X stays where it is. `reset task` resets goal progress and focus. Per the paper, focusing on the wrong object fails the task.
- Object JSON: `uuid`, `name`, `type`, `isDeleted`, then the property blocks:
  - `propMaterial` (`substanceName`, `color`, `temperatureC`, `thermalConductivity`, `stateOfMatter`, `boilingPoint`, `meltingPoint`, `isCombusting`, `hasCombusted`, `combustionTicks`, `electricallyConductive`, `frictionCoefficient`);
  - `propContainer` (`isContainer`, `isOpen`, `isClosable`);
  - `propDevice` (`isActivable`, `isActivated`, `isUsable`, `isBroken`);
  - `propElectricalConnection` (`connectedTo`: uuids);
  - `propLife` (`lifeformType`, `minTemp`, `maxTemp`, `isSickly`, `isDead`, `lifespanTypical`);
  - `propEdibility`, `propHeatSource`, `propCoolingSource`, `propPortal`, `propMoveable`.

  Last comes `contents`, a map from `"name-uuid"` to the child object. Hidden objects such as electrical terminals are included. **Doors are kept apart from contents and are not in the dump, and neither are plant life stages.**
- The agent is named `agent`. Its inventory is a container named `inventory` inside the agent. A room's description starts "This room is called the X. In it, you see:", and its doors follow "You also see:".

## Design

### Adapter (`sciworld/env.py`)

`SciWorld` is the only code that imports `scienceworld`, and it does so lazily. Its interface: `load(task, variation, simplification)`, `reset()`, `step(cmd) -> observation`, `idle()`, `moves()`, `valid_actions()`, `gold_actions()`, `task_description()`, `look()`, `inventory()`, `train_variations()`, `signatures() -> (full, core)`. The test fake implements the same interface.

- **Fast stepping.** `ScienceWorldEnv.step` builds the info dict on every call, including the full valid-action list. Replay never reads it, so by default the adapter calls the Java server's `step`/`reset` directly through the `server` attribute. `--slow-step` uses the public `step` instead.
- **Gold path.** `load()` loads without the gold flag, so resets stay cheap. `gold_actions()` loads once with the flag, then reloads without it. The second load in the jar rebuilds the same seeded world, so the gold path is valid in it.
- **Ambiguity.** The adapter answers `0`, the first option.
- **One idle tick.** `load()` measures how many ticks `wait1` takes. `idle()` uses `wait1` if that is one tick, otherwise the first `look at …` command, which takes one tick and changes nothing. It raises if the tick count is wrong.

### State restore

Restoring is reset plus replay of the command prefix (`labeller.Replayer`). The Replayer remembers what it has applied since the last reset, so when a target path extends that record it only steps forward. In practice that saves one replay per expanded node.

### Signatures

- **full**: SHA-1 of three things:
  - the object tree, with uuids dropped, children sorted, and temperatures rounded to 0.1 °C;
  - the sorted lines of the `look()` text, which adds door states and life stages in the current room;
  - the sorted lines of the `inventory()` text.

  Dropping uuids makes identical objects interchangeable. Rounding absorbs float noise.
- **core**: the sorted multiset of containment paths by name, such as `kitchen/table/cup/water`, with deleted objects skipped. It ignores every property: temperature, state of matter, open or closed, on or off.

### Labels and the counterfactual

ScienceWorld changes by itself: heat flows and plants grow. So a restore is compared with doing nothing, not with the saved state. Let s be the state, a the action, and t the number of ticks after s. Action a is restored at depth k if the signature after a plus k commands equals the signature after t idle ticks from s, where t is the tick count of that path. Ticks are matched because one command can take more than one tick. The baselines are cached per state and shared by all actions labelled there.

| Label at horizon h | Rule |
| --- | --- |
| reversible | the full signature matches after at most h further commands (depth 0, a no-op, counts) |
| partial | only the core signature matches within h: everything is where it was, but some property differs |
| irreversible | neither |

The rule and `LabelResult` are reused from `sandbox/labeller.py`. `label` is the horizon-3 label, and `label_h1`, `label_h2`, `label_h3` are kept for the horizon ablation. `--max-h` sets the maximum horizon.

### Search

Breadth first to depth `max_h`, stopping at the first exact match. At each node it tries up to `max_branch` (8) candidates, in this order:

1. the inverse of a: open/close, activate/deactivate, connect then disconnect either end, pour X into Y then pour Y into X;
2. placements from the core diff. If an object is in a different container from the counterfactual, try `move X to <its container there>`; also `pick up X` if that container is the inventory, and `put down X` if it is a room. If the agent is in another room, try `go to` and `teleport to` the original room;
3. valid commands that mention an object of a or a displaced object (whole-word match), with restoring verbs first.

A candidate is dropped if it repeats the previous command or undoes the previous restoring command. Meta commands are never candidates, and nor are `eat`, `mix` and `flush`. Each depth keeps the `beam` (8) children closest to the target in core terms. The search gives up after `max_nodes` (150) nodes and sets `search.truncated`. With the defaults that never happens: there are at most 1 + 8 + 64 + 64 = 137 nodes.

### Exclusions

These are never labelled and never tried: `look around`, `look at`, `look in`, `examine`, `inventory`, `task`, `read`, `focus on` (it can end the task), `wait`, `wait1`, `reset task`. Actions that change nothing (exact match at depth 0) are dropped from the data unless `--keep-noops` is set, as the sandbox drops failed calls. Their count is printed.

### Sampling (`sciworld/sample_states.py`)

- **Tasks** (default): boil, melt, use-thermometer, power-component, test-conductivity, find-living-thing, grow-plant, chemistry-mix, chemistry-mix-paint-secondary-color, inclined-plane-determine-angle. Simplification `easy`.
- **Variations**: `--variations` (5) per task, drawn with a fixed seed from `get_variations_train()`.
- **Gold path**: kept up to its first command that does not parse. Bare numbers in the gold path (answers to ambiguity prompts) are skipped, because the adapter already answers 0. If a gold path chose another option, the replay differs from it from that point on.
- **States**: `--states` (8) prefixes per variation, one random prefix from each equal slice of the gold path.
- **Actions**: at each state, `--per-state` (4) valid commands, stratified by verb (one random command per verb in turn). The next gold command goes first.

The defaults give at most 10 × 5 × 8 × 4 = 1,600 pairs before no-ops are dropped. Balancing then caps each class at `ceil(--target / 3)` (target 800). If one class is rare, label more states (`--states`, `--variations`) rather than lowering the cap.

### Parallelism (`sciworld/generate.py`)

A spawn-context `multiprocessing.Pool` of `--workers` processes. Each process owns one `SciWorld`, so one JVM. Jobs are states, shuffled with the seed so that `--limit` samples across tasks. Each finished state's rows are appended to the raw file, and `--resume` skips states already there. A failing state is reported and skipped and does not stop the run.

### Output rows

One JSON object per line, following the sandbox schema where it applies:

| Field | Value |
| --- | --- |
| `id` | `sw-<task>-v<variation>-p<prefix length>-a<index>` |
| `environment`, `source` | `"scienceworld"` |
| `state_text` | task description, `look()` text and `inventory()` text, separated by blank lines |
| `action_text`, `action` | the command string |
| `label`, `horizon` | label at `max_h` (3), and `max_h` |
| `task_text` | the task description |
| `scenario_id` | `sw-<task>-v<variation>`. Split train and test by this field. |
| `pair_id` | null, unless the same action text gets different labels in two states of one scenario |
| `mechanism` | the verb: `open`, `pick up`, `pour`, `mix`, `activate`, ... |
| `variant`, `cue_style` | null |
| `label_h1`, `label_h2`, `label_h3` | per-horizon labels |
| `restore_sequence` | shortest exact restore, else shortest core restore (commands) |
| `replay` | `{"task", "variation", "simplification", "prefix"}`: enough to rebuild the state |
| `search` | `{"exact_depth", "core_depth", "nodes", "truncated", "seconds"}`. Not in the sandbox schema; added for the data card and for debugging. |

## Runtime (estimate, to be replaced by Joy's measurements)

Each search node costs one reset, the replay of p + d + 1 commands and one signature, where p is the prefix length and d the depth. Each expanded node also costs one valid-action query. My guesses: reset 0.05 to 0.2 s, raw step 5 to 20 ms, signature 20 to 60 ms (tree dump plus two free actions), valid actions 0.05 to 0.3 s. At p ≈ 15 that is about 0.3 s per node, with a plausible range of 0.1 to 0.8 s.

| Pair type | Nodes | Seconds |
| --- | --- | --- |
| Easy reversible (open, pick up, teleport) | 2 to 10 | 1 to 5 |
| Partial or irreversible (full search) | up to 137 | 20 to 100 |
| Average (guess) | | about 20 |

Each state also pays a few seconds of overhead for the determinism replay and the idle baselines. 1,600 pairs at 20 s is about 9 CPU-hours, so roughly 35 minutes with 16 workers. Each worker runs its own JVM, so budget about 1 GB of RAM per worker (unmeasured). `python -m sciworld.env` prints the real per-operation timings and a per-node estimate.

## Smoke test for Joy (in this order)

1. `python -m pytest -q tests/test_sciworld.py` checks the labeller logic without Java. I wrote these tests without running them, so fix whatever fails first.
2. `java -version` (1.8 or newer), then install 1.3.0 as above.
3. `python -m sciworld.env --task boil --variation 0` prints the following. Check each:
   - **version**: `scienceworld 1.3.0`.
   - **`wait1 takes N tick(s)`**: the design expects 2, which means `idle()` uses `look at`. Either is fine.
   - **gold path length**: should be greater than 0.
   - **timings** and the per-node estimate.
   - **`agent and inventory paths`**: should show `.../agent` and `.../agent/inventory`. If they are empty, the core signature cannot see the agent, and placements for pick up and go to will not fire.
   - **`deterministic replay: True`**: if it prints False, stop. The labels would be noise.
4. Smoke run, about 20 pairs:
   ```bash
   python -m sciworld.generate --tasks boil,find-living-thing --variations 1 --states 3 --per-state 4 \
       --limit 20 --workers 1 --target 0 --out data/sciworld_smoke.jsonl
   ```
   It prints the states (`ok`, `nondeterministic`, `error`), dropped no-ops, class counts at each horizon, counts by verb, average search seconds and nodes per pair, and wall time per pair. Read five rows by hand: does `restore_sequence` really undo the action? Do `open`, `pick up` and `teleport` come out reversible, and `mix` and `eat` irreversible?
5. Full run, after the smoke run looks right:
   ```bash
   python -m sciworld.generate --variations 5 --states 8 --per-state 4 --workers 16 --target 800
   ```
   If it stops, rerun the same command with `--resume`. The output is `data/sciworld_pairs.jsonl` (balanced), plus `data/sciworld_pairs_raw.jsonl` (every row).

If fast stepping fails (for example, `server` has no `step`), add `--slow-step`. It is slower but uses only the public API.

## Known limitations (for the data card)

- **Heuristic search.** The candidate restriction, the beam and the node cap mean a label is "no restore found among these candidates", not a proof. Waiting is never tried as a restoring action.
- **Signatures.** Doors and life stages are not in the object tree. The full signature sees them only through the current room's look text, so a door left closed in another room is invisible. Temperatures are compared to 0.1 °C. Identical objects are interchangeable.
- **Partial** means things are back in place but a property differs: a temperature, a state of matter, a device left warm. That differs from the sandbox, where partial means trace fields (versions, notices, fees).
- **Counterfactual.** Idle ticks use `wait1` or `look at`. The labels assume the world's own dynamics are deterministic given the commands. States that fail the replay check are skipped, which may bias the sample.
- **Ambiguous commands** resolve to option 0.
- **States** are on gold paths only, so they are states a competent agent passes through. Actions are stratified by verb, not drawn from any natural policy. No-ops are dropped.
- **Simplifications.** `easy` makes navigation cheap to undo (teleport, open doors), and labels depend on it.
- **Horizon** counts commands, not ticks. A command that takes several ticks counts as one.
- **Version.** Labels are tied to the ScienceWorld version, because 1.3.0 changed trajectories. Record the version in the run log.

## Sources

- README and repository: https://github.com/allenai/ScienceWorld (main at `e8216d6`); commit list https://github.com/allenai/ScienceWorld/commits/main; tags https://github.com/allenai/ScienceWorld/tags
- Python API: https://raw.githubusercontent.com/allenai/ScienceWorld/main/scienceworld/scienceworld.py and https://raw.githubusercontent.com/allenai/ScienceWorld/1.2.3/scienceworld/scienceworld.py
- PyPI metadata: https://pypi.org/pypi/scienceworld/json
- Licence: https://raw.githubusercontent.com/allenai/ScienceWorld/main/LICENSE
- Determinism fix: https://github.com/allenai/ScienceWorld/pull/88 (fixes issue #33)
- Scala, under `simulator/src/main/scala/scienceworld/`:
  - `runtime/pythonapi/PythonInterface.scala` (reset, object tree, no save);
  - `runtime/AgentInterface.scala` (step loop, ticks);
  - `input/ActionHandler.scala` (free actions);
  - `input/InputParser.scala` (ambiguity text);
  - `input/ActionDefinitions.scala` (action list);
  - `actions/ActionWait.scala`, `actions/ActionLook.scala`, `actions/ActionMoveObject.scala`, `actions/ActionFocus.scala`;
  - `objects/agent/Agent.scala`, `struct/EnvObject.scala`, `properties/*.scala`, `objects/location/Location.scala`.

The pages were read through a summarising fetch tool, which quoted short fragments rather than whole files. Facts marked as inferred above (wait1 ticks) have runtime checks in the adapter.
