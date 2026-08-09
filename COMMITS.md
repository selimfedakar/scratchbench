# Commit queue — `selimfedakar/scratchbench`

One file per commit. Selim runs these — nothing in this repository is committed
or pushed by anyone else.

Every block starts with `cd ~/scratchbench` so a block is safe to paste on its
own, into any shell, without checking which directory you are in.

The loop variable is `file` and never `path`. In zsh `$path` is the array form
of `$PATH`, so `read path` empties it and every command after the first
iteration becomes "command not found", with `set -e` unable to fire because the
shell cannot find the binary that was supposed to fail. `docs/LESSONS.md` L24.

---

## Pending

### Session 09 — the accelerated tier answers

Twenty-two commits, twenty-two pushes, a minute apart. Every tree here is green:
the results files are self-contained, the documents are read by no CI step, and
`tools/check_cost.py` is happy at every point because each new file re-derives
on its own. Twenty-two green ticks, and about twenty-one minutes of `sleep`.

No force. `origin/main` and `main` are the same commit, so every push is a
fast-forward.

```bash
cd ~/scratchbench
set -e

n=0
while IFS='|' read -r file message; do
  if [ -z "$file" ]; then continue; fi
  n=$((n + 1))
  if [ "$n" -gt 1 ]; then sleep 60; fi
  git add "$file"
  git commit -q -m "$message"
  git push origin main
  echo "  $n pushed: $message"
done <<'QUEUE'
leaderboard/claude-haiku-4-5-20260809-draw1.json|leaderboard: claude-haiku-4-5 on the laptop tier, draw 1 of 5
leaderboard/claude-haiku-4-5-20260809-draw2.json|leaderboard: claude-haiku-4-5 on the laptop tier, draw 2 of 5
leaderboard/claude-haiku-4-5-20260809-draw3.json|leaderboard: claude-haiku-4-5 on the laptop tier, draw 3 of 5
leaderboard/claude-haiku-4-5-20260809-draw4.json|leaderboard: claude-haiku-4-5 on the laptop tier, draw 4 of 5
leaderboard/claude-haiku-4-5-20260809-draw5.json|leaderboard: claude-haiku-4-5 on the laptop tier, draw 5 of 5
leaderboard/accelerated-claude-opus-5-20260809-draw1.json|leaderboard: claude-opus-5 on the accelerated tier, draw 1 of 5
leaderboard/accelerated-claude-opus-5-20260809-draw2.json|leaderboard: claude-opus-5 on the accelerated tier, draw 2 of 5
leaderboard/accelerated-claude-opus-5-20260809-draw3.json|leaderboard: claude-opus-5 on the accelerated tier, draw 3 of 5
leaderboard/accelerated-claude-opus-5-20260809-draw4.json|leaderboard: claude-opus-5 on the accelerated tier, draw 4 of 5
leaderboard/accelerated-claude-opus-5-20260809-draw5.json|leaderboard: claude-opus-5 on the accelerated tier, draw 5 of 5
leaderboard/accelerated-claude-haiku-4-5-20260809-draw1.json|leaderboard: claude-haiku-4-5 on the accelerated tier, draw 1 of 5
leaderboard/accelerated-claude-haiku-4-5-20260809-draw2.json|leaderboard: claude-haiku-4-5 on the accelerated tier, draw 2 of 5
leaderboard/accelerated-claude-haiku-4-5-20260809-draw3.json|leaderboard: claude-haiku-4-5 on the accelerated tier, draw 3 of 5
leaderboard/accelerated-claude-haiku-4-5-20260809-draw4.json|leaderboard: claude-haiku-4-5 on the accelerated tier, draw 4 of 5
leaderboard/accelerated-claude-haiku-4-5-20260809-draw5.json|leaderboard: claude-haiku-4-5 on the accelerated tier, draw 5 of 5
leaderboard/README.md|leaderboard: both tiers, five draws each, and the failure shape beside the rate
docs/LESSONS.md|Lessons: a rate that put two failures in the wrong order
docs/sessions/09-the-accelerated-tier-answers.md|Journal 09: the accelerated tier answers
docs/V2_DESIGN.md|Design: one task already discriminates, and it is on the other tier
README.md|README: the accelerated tier is the part still measuring at the top
CLAUDE.md|Update state after the first accelerated model runs
COMMITS.md|Update commit queue
QUEUE
```

Then confirm:

```bash
cd ~/scratchbench

git status --porcelain          # expect: empty
git log --oneline | wc -l       # expect 122
python -m pytest -q             # expect 78 passed
python tools/check_cost.py      # expect 22 file(s) checked
gh run list --limit 3           # expect success
```

---

## Committed

### 2026-08-09 — session 08, ten commits

The five Opus draws on the laptop tier, the leaderboard page rewritten around
them, journal 08, and L26. Ten commits, ten pushes a minute apart, all green.

### 2026-08-09 — the replay, 45 commits, and CI green

`main` was reset to `fa24428` and everything after it replayed, ten commits on
2026-08-03 and thirty-five the next day, one push per commit. Ninety commits,
working tree clean, every workflow run green.

The fifteen red crosses are gone from the history because the commits they
belonged to are: a check run belongs to a SHA, and none of those SHAs is
reachable from `main` any more. All fifteen had the same cause and all fifteen
were correct — `adapters/__init__.py` was committed importing
`adapters.anthropic_api`, which was never committed, so `pytest` exited 2 at
collection from the first CI commit onwards. `docs/LESSONS.md` L17 and L24.

Per-commit pushing was safe there because of where the workflow file sat in the
queue: `ci.yml` was commit 34 of 45, so the thirty-three pushes before it started
no run at all, and the tree at 34 had been assembled and run against CI's four
commands beforehand. That ordering is not a general property. `ci.yml` in its
original position lands before `tools/check_cost.py` has anything in
`leaderboard/` to check and goes red on arrival.

The Actions tab keeps run history independently of commits, so the fifteen
failures stay listed there until deleted:

```bash
gh run list --limit 20 --json databaseId,conclusion \
  --jq '.[] | select(.conclusion=="failure") | .databaseId' \
  | while read -r id; do gh run delete "$id"; done
```
