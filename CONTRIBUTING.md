# Contributing

The most useful thing you can send is a task. The second most useful is a
result from a model I have not run.

## Ground rule

Nothing here is copied from any course, textbook, or existing benchmark — not
code, not tests, not assignment text. Every task is written from scratch.
Inspiration is fine and redistribution is not, and the README makes that claim
in public, so a pull request that breaks it cannot be merged.

## Adding a task

Read [`TASK_FORMAT.md`](TASK_FORMAT.md) first. Then write the four parts in
this order, which is not a suggestion:

1. **`reference/`** — a correct solution. Writing it first is what surfaces the
   conventions you were about to leave unstated: output dtype, what happens on
   empty input, whether the operation is in place.
2. **`hidden_tests/`** — plain pytest, deterministic, seeded, offline. Every
   claim the prompt makes, and nothing it does not.
3. **`prompt.md`** — the specification, written the way a colleague hands work
   over. No test file names, no tolerances, no checklist of edge cases.
4. **`meta.yaml`** — the metadata block from the task format.

Then check it, both halves:

```bash
scratchbench validate --tasks <your_slug> --verbose
```

That runs the reference against the hidden tests and the untouched starter
against the same tests. The reference must pass. The starter must fail with
real test failures — an import error and a wrong answer score the same and mean
completely different things, so the starter has to be runnable from the first
second.

## The rule task authors break most often

> If a hidden test can fail for a reason `prompt.md` never states, the task is
> broken.

After writing the tests, walk every assertion and find the sentence in the
prompt that licenses it. If there is no such sentence, either add it or delete
the assertion. This one pass catches more bad tasks than everything else
combined.

Two more that follow from it:

- **A test that cannot fail is noise.** Binary scoring means a test that passes
  against an empty starter contributes nothing. Run your suite against the
  untouched starter and confirm every test fails.
- **A task that cannot catch a plausible wrong answer is worse than no task.**
  Mutate your reference into the versions a real implementation would produce —
  the off-by-one, the wrong axis, the missing rescale — and check the tests
  catch each one. Put the results in the pull request.

## Constraints that are not negotiable

- No GPU on the `laptop` tier, which is the default and the one the leaderboard
  rate is computed over: `requires_gpu: false` there is enforced by the loader.
  A task that genuinely needs hardware goes on the `accelerated` tier, declares
  which accelerator it needs, and is reported separately — see
  [`TASK_FORMAT.md`](TASK_FORMAT.md). Its reference has to have passed on that
  hardware, with the output in the pull request, before it counts; until then
  it stays out of the frozen set.

  CI cannot do this for you and deliberately does not try. A self-hosted GPU
  runner on a public repository is a hazard — a pull request can edit `runs-on`
  — so the accelerated tier's evidence is collected by hand:

  ```bash
  bash tools/verify_accelerated.sh   # CUDA: on a machine with the accelerator
  python tools/mutate_metal_task.py  # Metal: on any Apple silicon Mac
  ```

  That prints the device, the torch and triton versions, the harness suite, both
  halves of `validate`, a control run, and the mutation pass. Paste all of it.
  A transcript that names the hardware is worth more than a green tick that does
  not.

  The Metal half of that is cheaper than it sounds: `torch.mps.compile_shader`
  turns a Metal source string into a callable kernel, so the hardware a `metal`
  task needs is the laptop the rest of the benchmark already runs on.
- Under five minutes, and the time limit in `meta.yaml` is enforced as a hard
  timeout. A GPU is not a licence to need an hour.
- Deterministic: fixed seeds, no network, no wall-clock assertions.
- New dependencies need a justification in the pull request. `numpy`, `pytest`
  and `torch` are already here; anything else has to earn its place.
- English only, in code, comments, and documentation.

## Sending a result

Run the task set against a model and open a pull request with the results file
from `results/`. It already carries the harness version, the task set version
and the attempt count — a pass on the eleventh attempt is a different result
from a pass on the first, and the leaderboard says so.

Send more than one draw if you can, and name the set you are asking about:

```bash
scratchbench run --model <model> --tasks all --set v1 --repeat 5
scratchbench report --variance
```

`--set` matters now that the laptop tier holds more than one published set.
Without it a sweep covers `v1` and `warmup` together and its `pass_rate` is an
average across them, which is not a row anybody can compare to another row. The
results file records `task_set` as the list it actually covered, so a blended run
is visible rather than hidden, but a leaderboard row wants one set.

The model is sampled and the harness is not, so a single sweep is one draw. The
same model here has passed a task on one run and failed it on the next with the
same prompt and the same settings. Five draws turn a row into a range, and a
range is the only version of it two models can honestly be compared on. Attach
every draw's file; they are independent runs, not retries.

If you send a cost, it has to reproduce:

```bash
python tools/check_cost.py results/<your-file>.json
```

## Calibrating a task

A new task aimed at a numbered frozen set (`v2` and later) needs a `calibration:`
block in its `meta.yaml` and the draws behind it in `calibration/`:

```bash
scratchbench run --model <model> --tasks <your_slug> --repeat 10 --keep /tmp/draws
cp results/<model>-*.json calibration/
python tools/check_calibration.py
```

Ten draws rather than five if you can afford them: two independent five-draw
sweeps of the same model on the same task have come back 2/5 and 5/5 here.

Two models, not one. The loader refuses the task if the **top two** models you
tried both passed every draw, and it needs two entries of five draws or more
before it will read anything at all. A refusal is not a bug report — it means the
task belongs in `frozen_set: warmup`, where it keeps its block and stays out of
the headline. Five tasks are there now, the reasoning is in `docs/LESSONS.md`
L30, and why the rule reads two entries rather than one is L35.

Keep the per-draw workdirs (`--keep`) and say which tests failed, by name. `1 of
24` and `24 of 24` are the same verdict and completely different results, and
counting is not the same as naming: four identical count lines have looked like
the same test here and were not checked until they were.

## Development

```bash
pip install -e .
python -m pytest -q                 # the harness's own tests
scratchbench validate               # every task, both halves
scratchbench run --model reference  # the control run through the full harness
```

CI runs all three on every commit.
