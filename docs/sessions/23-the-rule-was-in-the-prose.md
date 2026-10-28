# 23 — The rule was in the prose

The review's editorial half, §5.2 and §6, and one of them is not editorial at
all. The `v2` Mac-half table reported a single pass rate per model over a sweep
that spanned the laptop task `flash_attention_backward` and the two Metal tasks.
`claude-opus-5` read **90%, 27/30** — a laptop result and a GPU result averaged
into one number, under a heading three lines below where the page says tiers are
never folded.

## The tier that was folded anyway

`render_leaderboard` pooled `solved / measured` over every graded task in a draw
and knew nothing about tiers. The rule was in the prose and not in the renderer.
The fix is a tier-filtered rate table rendered from the same files:

```
| Model | Pass rate | Solved |
| `claude-opus-5`   | 85% | 17/20 |
| `claude-sonnet-5` | 65% | 13/20 |
| `claude-haiku-4-5`|  6% |  1/18 |
| `gpt-5.6-sol`     | 70% | 14/20 |
| `gpt-5.6-terra`   | 90% | 18/20 |
| `gpt-5.6-luna`    | 50% |  7/14 |
```

The mixed table stays, relabelled as the blend it is, with
`flash_attention_backward` named as the laptop member set aside. These numbers
are re-derived by `tools/render_leaderboard.py`, not typed: `--check` is green.

## The claims that were sentences nobody checked

§6, corrected on the front page to what is true:

- "No GPU, no cloud bill" → the CPU tier is free; a real model is a paid API call
  and three tasks want a GPU, both kept off the laptop rate.
- "Nobody is measuring…" → "what I could not find is a benchmark for…", with the
  neighbours named rather than dismissed.
- "Passing tests prove the reference is right" → proves it satisfies the contract
  the tests cover; only mutants prove they would catch a wrong version.
- "Every set is frozen" → set membership is frozen in `SET_MANIFESTS.yaml`;
  `frozen_set` is the mutable label.
- "Every row is one draw" → a row is the summary of however many draws it says.
- `251 files / 35 entries` → `311 / 44`.

And three cleanups the review flagged: the duplicate `ResponseError` in
`adapters/model_api.py` is gone, `CLAUDE.md` no longer calls the OpenAI adapter a
skeleton (it is real and lives in `openai_api.py`), and `TASK_FORMAT.md`'s meta
example carries the now-required `hidden_tests` key.

## Verified

```
$ python tools/render_leaderboard.py --check
7 rendered region(s) match their results files.
$ python tools/render_readme.py --check
2 rendered region(s) match the task directories.
```

The lesson is `docs/LESSONS.md` L54: the rate was wrong in the one direction the
whole page swears it never is, because the rule lived where a reader believes it
and not where it would be enforced. What remains of §5.2/§6 — trimming the
6,000-word front page, the full scope-comparison section — is editorial polish,
noted in `ROADMAP.md` §12, not correctness.
