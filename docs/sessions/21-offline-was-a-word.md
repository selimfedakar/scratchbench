# 21 — "Offline" was a word

The review's other P0 is not a wrong number; it is a wrong claim about what
running the benchmark does. `pytest_environment` was documented as "a
deterministic, offline environment", and the sandbox module talks about
guarantees. Neither the word nor the implication held.

```
{"probe": "environment", "dummy_secret_inherited": true}
{"probe": "outside_workdir_read", "sentinel_read": true}
```

A graded solution inherited the whole parent environment — including the API key
the adapter was called with — and read a sentinel file a directory above its
workdir. There is no network block, no filesystem jail, no resource limit. The
solver-cannot-read-a-test-not-yet-on-disk guarantee is real, and it is about
information, not safety; I had let it stand in for a sandbox it never was.

## The line between a fix and a document

One thing here is fixable cheaply and one is not, and conflating them would be
the original mistake again.

**Fixed.** The adapter runs in the *parent* process, where the key lives and the
API call is made. The grading child only runs pytest over a file already on
disk, so it has no use for a credential. `pytest_environment` now strips
credential-shaped variables before the child starts:

```
{"probe": "environment", "dummy_secret_inherited": false}
```

That closes the one secret the harness itself introduces, and it breaks nothing,
because nothing downstream of the adapter needed the key.

**Documented, not fixed.** Everything else. `outside_workdir_read` still returns
true, and it should, because env-cleaning is not isolation and a denylist cannot
see a filesystem. `grade`'s docstring and a new README "Running generated code"
note now state plainly that the solution runs in the same interpreter with the
host's filesystem and network, and that a real model's output should be run like
any untrusted program — on a machine or disposable container you are willing to
hand it — until an isolated grading path exists. F02 is told the truth about
rather than closed.

## Verified

```
$ python review-probes.py
{"probe": "environment", "dummy_secret_inherited": false}
{"probe": "outside_workdir_read", "sentinel_read": true}   # deliberately still true
```

One regression test: a credential-named variable is stripped from the graded
environment, a benign one is kept, and the deterministic pins stay set. The
lesson is `docs/LESSONS.md` L52: "offline" was a word doing the work the code
was not, the same shape as an exit code read as a verdict (L50) and an invariant
read as a fact about every vendor (L49).
