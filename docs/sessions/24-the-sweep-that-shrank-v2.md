# 24 — The sweep that shrank v2

The last measurement before launch. Rent a CUDA box, finish
`fused_rmsnorm_kernel` at ten draws per model across both vendors, re-sweep the
two version-2 batchnorm tasks that rode with it, and ship. Three of those four
things happened. The fourth — shipping `v2` as three accelerated tasks — did
not, because the sweep removed one of the three.

## The box was API-bound, so it ran twelve ways at once

The job is model calls, not kernels: the rented RTX 4090 sat at ~7% utilisation
because every draw spends its time waiting on an API and milliseconds grading.
So the sweep is one process per (task, model) — twelve in parallel, each writing
to its own `--results` directory so two draws of the same model on different
tasks cannot collide on a filename. It finished in about seven minutes instead
of the two to three hours a serial run would have taken, at the same cost.

The tree was shipped to the box with `tar` over `ssh` rather than a `git clone`,
on purpose: the review fixes (sessions 19–23) are not pushed yet, so a clone
would have graded the old grader and the version-1 batchnorm tasks. The box ran
the working tree as it stood. Hardware was proven before any money moved —
`tools/verify_accelerated.sh` named the 4090 and ran **155 passed** — and a
one-draw smoke test confirmed the adapter, the key and the grading path end to
end before committing to 120 draws.

## What the draws said

Clean, ten graded draws per cell, re-derived locally:

```
fused_rmsnorm_kernel   opus 5/10  sonnet 10/10  haiku 5/10
                       gpt-5.6-sol 10/10  terra 10/10  luna 10/10
batchnorm_backward     opus 10/10  sonnet 10/10  haiku 0/10
batchnorm_reduction    opus 8/10   sonnet 9/10   haiku 0/10
```

`fused_rmsnorm_kernel` saturated at the top. Opus's five losses are all
`test_a_single_column`, the Triton single-column specialization edge, the same
signature as August; Haiku's five are the wrapper contract, not the kernel. But
Sonnet and all three OpenAI models went 10/10, so the top two measured rates are
both 100% and `_check_admission` refuses the task. **`v2` is two Metal tasks
now** — no laptop member, no CUDA member. This is flash-attention's ejection a
second time, `docs/LESSONS.md` L55, and the uncomfortable half is L46: the
cheapest model in the field clearing the task is half the reason it is refused.

`batchnorm_reduction` is the near-miss. Its two new version-2 tests cost Opus one
draw and Sonnet one, so the rule would now *admit* it — a `v2` laptop member at
last. It stays `warmup` by judgement, because every frontier loss fails all six
chunk sizes including `chunk_sizes0`, the single-chunk split where the named
mistake is correct by construction: the admission-clearing draws never touched
the mechanism. L45 a second time, written up as L56.

## The credit ran out, and the machinery said so

Anthropic's balance emptied partway through, so sixteen batchnorm draws came back
`adapter_error` — the same `invalid_request_error` the 2026-08-09 and 2026-09-06
sweeps hit. They measured nothing and entered neither the numerator nor the
denominator. Topped up, the five short cells were re-run to a clean ten graded
draws each. The OpenAI half was untouched; those draws finished before the
Anthropic key went dry.

## Ingestion, and what is verified

The version-1 batchnorm draws were removed from `calibration/` (they measured a
different test suite) and replaced with the version-2 draws; the rmsnorm draws
were added cumulatively to the August evidence. Three `meta.yaml` blocks and the
`SET_MANIFESTS.yaml` note were rewritten, `fused_rmsnorm_kernel` relabelled
`warmup`, and the README's three stale present-tense claims about the CUDA task
corrected. Every check re-run from this session's output:

```
full suite (nogpu plugin)   155 passed
validate                    14 task(s) validated
check_calibration           48 entries / 362 draws
check_cost                  362 file(s) checked
render_readme --check       2 regions match
render_leaderboard --check  7 regions match
```

Spend this session: **$4.44** over 170 draws on a rented 4090. What is left for
launch is editorial and the Show HN post; the last blocking measurement is
closed. `docs/ROADMAP.md` §13.
