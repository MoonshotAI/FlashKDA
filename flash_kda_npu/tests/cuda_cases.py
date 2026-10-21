"""CUDA test matrices, mirrored locally without importing CUDA-only helpers.

Keep these values identical to tests/test_fwd_full.py so NPU tests cover the
same shapes and state combinations without depending on CUDA-only helpers.
"""

D = 128
LOWER_BOUND = -5.0
STATE_IO = [(True, True), (True, False), (False, True), (False, False)]
STATE_IO_IDS = ["in+out", "in_only", "out_only", "no_state"]
STATE_DTYPES = ["bf16", "fp32"]
H_VALUES = [1, 4, 32, 96]
T_VALUES = [16, 64, 256, 1024, 4096, 8192, 17, 37, 97]
VARLEN_CASES = [
    [16],
    [16, 16],
    [4, 8, 12],
    [64, 128, 256],
    [1300, 547, 2048, 963, 271, 3063],
    [1024] * 8,
    [20, 50, 100],
    [17, 33, 65],
]
VARLEN_H = [1, 4, 96]
BATCH_CASES = [(2, 64), (2, 1024), (4, 256), (8, 128)]
BATCH_H = [1, 4, 96]
LONG_T_VALUES = [131072, 1048576]
LONG_VARLEN_CASES = [[131072], [524288, 524288]]
