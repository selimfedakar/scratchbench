"""Mutation check for the v2 laptop tasks.

Passing tests prove the reference is right. Only mutants prove the tests would
catch anything else, and `CONTRIBUTING.md` asks for that evidence in the pull
request rather than on trust. This applies, to each task, the wrong
implementations a real attempt would plausibly produce, and reports for each
whether the hidden tests noticed.

It lives in the repository for the same reason `mutate_rmsnorm.py` does: a check
nobody can re-run is a claim, not a check. That one stays separate because it
only runs on a CUDA machine and is driven by `verify_accelerated.sh`; everything
here runs on the laptop tier, which is to say anywhere.

Usage, from anywhere:

    python tools/mutate_v2_tasks.py                       # every task below
    python tools/mutate_v2_tasks.py --repo /path/to/repo --task <slug>

Every mutant must come back CAUGHT. A SURVIVED line is a hole in the task.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

# slug -> (module file, [(name, what it does wrong, old text, new text), ...])
TASKS: dict[str, tuple[str, list[tuple[str, str, str, str]]]] = {
    "speculative_decoding_verify": (
        "speculative_decoding.py",
        [
            (
                "ratio upside down",
                "accepts with min(1, draft / target) instead of the other way round",
                "if rng.random() * proposed < wanted:",
                "if rng.random() * wanted < proposed:",
            ),
            (
                "resample from the target",
                "draws the replacement straight from the target distribution",
                """        residual = np.maximum(target_probs[position] - draft_probs[position], 0.0)
        emitted.append(_sample(residual / residual.sum(), rng))""",
                """        emitted.append(_sample(target_probs[position], rng))""",
            ),
            (
                "absolute difference",
                "renormalises |target - draft| instead of its positive part",
                "residual = np.maximum(target_probs[position] - draft_probs[position], 0.0)",
                "residual = np.abs(target_probs[position] - draft_probs[position])",
            ),
            (
                "residual over the rejection probability of one token",
                "divides the residual by 1 - target/draft rather than by its own mass",
                "emitted.append(_sample(residual / residual.sum(), rng))",
                "emitted.append(_sample(residual / (1.0 - wanted / proposed), rng))",
            ),
            (
                "never accepts",
                "rejects every proposal, which emits the right distribution and no speedup",
                "if rng.random() * proposed < wanted:",
                "if rng.random() < 0.0:",
            ),
            (
                "no free token",
                "stops at the end of the proposal instead of taking the token after it",
                """    emitted.append(_sample(target_probs[len(draft_tokens)], rng))
    return np.array(emitted, dtype=np.int64)""",
                """    return np.array(emitted, dtype=np.int64)""",
            ),
            (
                "free token off by one",
                "takes the extra token from the last proposal position, not the one after",
                "emitted.append(_sample(target_probs[len(draft_tokens)], rng))",
                "emitted.append(_sample(target_probs[len(draft_tokens) - 1], rng))",
            ),
            (
                "keeps going after a rejection",
                "verifies the rest of the proposal instead of ending the round",
                """        emitted.append(_sample(residual / residual.sum(), rng))
        return np.array(emitted, dtype=np.int64)""",
                """        emitted.append(_sample(residual / residual.sum(), rng))
        continue""",
            ),
        ],
    ),
    "flash_attention_backward": (
        "flash_backward.py",
        [
            (
                "no row correction",
                "differentiates the softmax as if its Jacobian were diagonal",
                "d_scores = weights * (d_weights - delta[..., None])",
                "d_scores = weights * d_weights",
            ),
            (
                "correction over this block only",
                "computes the row term from the columns in hand instead of the whole row",
                "delta = (do * o).sum(dim=-1)",
                "delta = (weights * d_weights).sum(dim=-1)",
            ),
            (
                "query gradient unscaled",
                "leaves 1/sqrt(head_dim) off dq",
                "return (d_scores @ k_block) * scale, (d_scores.transpose(-1, -2) @ q) * scale, dv_block",
                "return d_scores @ k_block, (d_scores.transpose(-1, -2) @ q) * scale, dv_block",
            ),
            (
                "key gradient unscaled",
                "leaves 1/sqrt(head_dim) off dk",
                "return (d_scores @ k_block) * scale, (d_scores.transpose(-1, -2) @ q) * scale, dv_block",
                "return (d_scores @ k_block) * scale, d_scores.transpose(-1, -2) @ q, dv_block",
            ),
            (
                "value gradient against the output",
                "weights the output instead of its gradient",
                "dv_block = weights.transpose(-1, -2) @ do",
                "dv_block = weights.transpose(-1, -2) @ o",
            ),
            (
                "weights without the log-sum-exp",
                "exponentiates the scores instead of recovering the softmax from lse",
                "weights = torch.exp(scores - lse[..., None])",
                "weights = torch.exp(scores)",
            ),
            (
                "a query may not read its own key",
                "masks at or before the query position as strictly before it",
                "scores = scores.masked_fill(key_positions > query_positions[:, None], -math.inf)",
                "scores = scores.masked_fill(key_positions >= query_positions[:, None], -math.inf)",
            ),
            (
                "block placed at the start of the sequence",
                "masks with the block's own indices instead of its offset ones",
                "key_positions = key_offset + torch.arange(k_block.shape[-2], device=q.device)",
                "key_positions = torch.arange(k_block.shape[-2], device=q.device)",
            ),
            (
                "queries placed at the start of the sequence",
                "ignores the query offset inside the block function",
                "query_positions = query_offset + torch.arange(q.shape[-2], device=q.device)",
                "query_positions = torch.arange(q.shape[-2], device=q.device)",
            ),
            (
                "queries not put at the end of the key range",
                "drives the blocks with a query offset of zero",
                "query_offset = n_keys - q.shape[-2]",
                "query_offset = 0",
            ),
        ],
    ),
    "activation_checkpointing_rng": (
        "checkpointed_mlp.py",
        [
            (
                "recompute without rewinding",
                "recomputes the block against whatever randomness comes next",
                """        generator.set_state(states[index])
        h, mask, dropped, _ = _block_forward(u, w1, w2, p, generator)""",
                """        h, mask, dropped, _ = _block_forward(u, w1, w2, p, generator)""",
            ),
            (
                "generator left where the recompute finished",
                "restores before each recompute and never puts it back afterwards",
                """    generator.set_state(after_forward)
    return boundaries[-1], d_out, list(reversed(grads))""",
                """    return boundaries[-1], d_out, list(reversed(grads))""",
            ),
            (
                "state captured after the block rather than before",
                "remembers where each block ended instead of where it started",
                """        states.append(generator.get_state())
        _, _, _, out = _block_forward(boundaries[-1], w1, w2, p, generator)""",
                """        _, _, _, out = _block_forward(boundaries[-1], w1, w2, p, generator)
        states.append(generator.get_state())""",
            ),
            (
                "no draw when there is nothing to drop",
                "skips the mask at p = 0, so the stream depends on a hyper-parameter",
                """    keep = torch.rand(tuple(shape), generator=generator, dtype=torch.float64) >= p
    return keep.to(torch.float64) / (1.0 - p)""",
                """    if p == 0.0:
        return torch.ones(tuple(shape), dtype=torch.float64)
    keep = torch.rand(tuple(shape), generator=generator, dtype=torch.float64) >= p
    return keep.to(torch.float64) / (1.0 - p)""",
            ),
            (
                "mask the wrong way round",
                "keeps the positions whose draw is below p",
                "torch.rand(tuple(shape), generator=generator, dtype=torch.float64) >= p",
                "torch.rand(tuple(shape), generator=generator, dtype=torch.float64) < p",
            ),
            (
                "dropout not inverted",
                "drops without rescaling what survives",
                "return keep.to(torch.float64) / (1.0 - p)",
                "return keep.to(torch.float64)",
            ),
            (
                "single precision draw",
                "draws the uniforms in float32, which is a different amount of stream",
                "generator=generator, dtype=torch.float64) >= p",
                "generator=generator, dtype=torch.float32) >= p",
            ),
            (
                "residual missing from the backward",
                "sends the gradient through the block and not around it",
                "d_out = d_h @ w1.T + d_out",
                "d_out = d_h @ w1.T",
            ),
            (
                "relu gradient before the mask",
                "forgets that the dropped positions contributed nothing",
                """        d_a = d_dropped * mask
        d_h = d_a * (h > 0)""",
                """        d_h = d_dropped * (h > 0)""",
            ),
            (
                "second weight gradient from the undropped activation",
                "differentiates against relu(h) rather than what actually reached w2",
                "d_w2 = dropped.T @ d_out",
                "d_w2 = torch.relu(h).T @ d_out",
            ),
        ],
    ),
}


def graded_environment() -> dict[str, str]:
    """The same pinned environment the runner grades in."""
    env = dict(os.environ)
    env.update(
        {
            "PYTHONHASHSEED": "0",
            "PYTHONDONTWRITEBYTECODE": "1",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "TOKENIZERS_PARALLELISM": "false",
        }
    )
    env.pop("PYTHONPATH", None)
    return env


def run_tests(source: str, task_dir: Path, module: str) -> tuple[int, str]:
    workdir = Path(tempfile.mkdtemp(prefix="mutate-v2-"))
    try:
        (workdir / module).write_text(source, encoding="utf-8")
        for test_file in sorted((task_dir / "hidden_tests").glob("test_*.py")):
            shutil.copy2(test_file, workdir / test_file.name)
        completed = subprocess.run(
            [sys.executable, "-m", "pytest", "-q", "--tb=line", "-p", "no:cacheprovider"],
            cwd=workdir,
            env=graded_environment(),
            capture_output=True,
            text=True,
            timeout=1800,
        )
        return completed.returncode, (completed.stdout + completed.stderr).strip()
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


def last_line(output: str) -> str:
    lines = [line for line in output.splitlines() if line.strip()]
    return lines[-1] if lines else "(no output)"


def check(repo: Path, slug: str) -> int:
    module, mutants = TASKS[slug]
    task_dir = repo / "tasks" / slug
    reference = (task_dir / "reference" / module).read_text(encoding="utf-8")
    starter = (task_dir / "starter" / module).read_text(encoding="utf-8")

    print(f"=== {slug} ===")
    print("--- reference ---")
    code, output = run_tests(reference, task_dir, module)
    print(last_line(output))
    if code != 0:
        print("\nthe reference does not pass; nothing below means anything")
        print(output)
        return 1

    print("--- untouched starter ---")
    code, output = run_tests(starter, task_dir, module)
    print(last_line(output))
    if code == 0:
        print("\nthe untouched starter passes; the tests prove nothing")
        return 1

    print("--- mutants ---")
    survivors = 0
    for name, description, old, new in mutants:
        if old not in reference:
            print(f"SKIPPED   {name}: the reference no longer contains the patched text")
            survivors += 1
            continue
        code, output = run_tests(reference.replace(old, new, 1), task_dir, module)
        verdict = "SURVIVED" if code == 0 else "CAUGHT  "
        survivors += int(code == 0)
        print(f"{verdict}  {name} ({description})")
        print(f"          {last_line(output)}")

    print()
    if survivors:
        print(f"{slug}: {survivors} mutant(s) survived — the task has a hole in it")
        return 1
    print(f"{slug}: all {len(mutants)} mutants caught")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default=".", help="path to the scratchbench checkout")
    parser.add_argument("--task", action="append", choices=sorted(TASKS), help="one slug")
    arguments = parser.parse_args()

    repo = Path(arguments.repo).resolve()
    failures = 0
    for slug in arguments.task or sorted(TASKS):
        failures += check(repo, slug)
        print()
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
