import os
import sys

import pytest
import torch

# Prefer the local flash-linear-attention checkout so tests compare against
# the development version of the Triton kernels.
_FLA_REPO = os.environ.get("FLA_REPO", "/root/flash-linear-attention")
if os.path.isdir(_FLA_REPO) and _FLA_REPO not in sys.path:
    sys.path.insert(0, _FLA_REPO)

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
