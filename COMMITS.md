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

### Session 08 — the five draws, and what they say

Ten commits, ten pushes, a minute apart. Every tree here is green, so unlike the
replay this needs no ordering care: the results files are self-contained and the
documents are read by no CI step. Ten workflow runs, ten green ticks.

No force this time. `origin/main` and `main` are the same commit, so every push
is a fast-forward.

Safe to run while a sweep is going in another terminal. The sweep writes to
`results/`, which `.gitignore` covers, and none of the ten files below is
touched by it.

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
leaderboard/claude-opus-5-20260809-draw1.json|leaderboard: claude-opus-5 on task set v1, draw 1 of 5
leaderboard/claude-opus-5-20260809-draw2.json|leaderboard: claude-opus-5 on task set v1, draw 2 of 5
leaderboard/claude-opus-5-20260809-draw3.json|leaderboard: claude-opus-5 on task set v1, draw 3 of 5
leaderboard/claude-opus-5-20260809-draw4.json|leaderboard: claude-opus-5 on task set v1, draw 4 of 5
leaderboard/claude-opus-5-20260809-draw5.json|leaderboard: claude-opus-5 on task set v1, draw 5 of 5
leaderboard/README.md|leaderboard: five draws, and the error bar moves to the column that needed one
docs/LESSONS.md|Lessons: the error bar went on the column that did not need it
docs/sessions/08-five-draws.md|Journal 08: five draws
README.md|README: the top row is five draws now, and it is a ceiling
COMMITS.md|Update commit queue
QUEUE
```

Then confirm:

```bash
cd ~/scratchbench

git status --porcelain          # expect: empty
git log --oneline | wc -l       # expect 100
python -m pytest -q             # expect 78 passed
python tools/check_cost.py      # expect 7 file(s) checked
gh run list --limit 3           # expect success
```

---

## Committed

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
