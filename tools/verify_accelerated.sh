#!/usr/bin/env bash
# Everything the accelerated tier needs proved, on a machine that has the
# hardware. Run it from the repository root on a CUDA box:
#
#     bash tools/verify_accelerated.sh
#
# It prints five things, and all five belong in the pull request that moves an
# accelerated task into the frozen set:
#
#   1. which GPU, which torch, which triton — a result without these is not
#      reproducible by anyone;
#   2. the harness's own suite, on Linux with a CUDA build of torch;
#   3. the reference passing the hidden tests, and the untouched starter
#      failing them with real test failures;
#   4. the same task through the full harness, the way a model would be graded;
#   5. every mutant caught.
#
# The first step is the one that makes the rest mean anything. `validate`
# prints UNCHECKED and exits zero on a machine with no accelerator, so without
# it a box that quietly lost its GPU would produce a clean transcript against a
# task nothing had run.

set -euo pipefail

echo "=== 1. hardware ==="
python - <<'PY'
import sys

import torch

if not torch.cuda.is_available():
    sys.exit("no CUDA device here: everything below would be a transcript of nothing")
import triton

print(torch.cuda.get_device_name(0), "| torch", torch.__version__, "| triton", triton.__version__)
PY

echo
echo "=== 2. harness suite ==="
python -m pytest -q

echo
echo "=== 3. validate (reference passes, untouched starter fails) ==="
python -m runner.cli validate --tasks fused_rmsnorm_kernel --verbose

echo
echo "=== 4. control run through the full harness ==="
python -m runner.cli run --model reference --tasks fused_rmsnorm_kernel --tier accelerated --no-write

echo
echo "=== 5. mutants ==="
python tools/mutate_rmsnorm.py .
