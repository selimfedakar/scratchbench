# 22 — A name is not an identity

F07 is the finding that two draws of different things could share one error bar.
`group_draws` keyed on model, task set, and the list of slugs — careful, and
blind to the one change that matters most. Bump a task's version under the same
slug and the probe folds the two together:

```
{"probe": "different_task_versions_grouped", "group_count": 1}
```

One group, two contracts. A draw graded against version 1's tests and a draw
graded against version 2's, reported as model sampling. And `frozen_set` in
meta.yaml was standing in for a set's membership while tasks move between sets as
the frontier clears them — so "v1" had stopped naming the eight tasks its row was
measured over.

## What changed

**Grouping keys on `(slug, task_version)`.** A version bump is a new question, so
draws across it are two experiments. An unversioned legacy file is kept as `None`
rather than assumed to be version 1 — incomplete provenance is not a known one.

```
{"probe": "different_task_versions_grouped", "group_count": 2}
```

**Every run records its commit.** `results_version` is 4; `environment` now
carries `commit` and `dirty`, so a number ties back to the code that made it and
a run off uncommitted changes cannot pass as a committed one. Outside a git
checkout both are `None`, because the honest answer there is "unknown".

**Set membership is frozen.** `leaderboard/SET_MANIFESTS.yaml` records what v1 and
v2 were measured over, re-derived from the published files by a test, with
`frozen_set` demoted to the mutable evaluation label it always was. v1 keeps its
eight laptop tasks; v2 keeps `flash_attention_backward` even though it is `warmup`
today, because that is what the historical row stands on.

## Verified

```
$ python -m pytest -q -k "grouped or commit or git_provenance or measured_on or empty_list_rather"
8 passed

$ python tools/check_calibration.py
44 calibration entries re-derived from 311 draw(s)
$ python tools/check_cost.py
311 file(s) checked: every cost reproduces from its own tokens.
```

`check_calibration` is left keyed on `(model, slug)`, version-agnostic, which is
correct *only* because every published draw is version 1; the moment a version-2
draw exists it has to key on version too, and that is written down in
`ROADMAP.md` section 12 rather than assumed. The lesson is `docs/LESSONS.md` L53:
a name is a claim of sameness, and the rest of a run's identity — the resolved
model snapshot, the prompt and test and solution hashes, the dependency lock, a
replayable artifact — is debt, named in the roadmap so "I added provenance" says
which provenance it did not add.
