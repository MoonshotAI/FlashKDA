"""Explicit fixed-shape NPU graph capture for repeated KDA forward calls."""
from __future__ import annotations

import os

import torch
import torch_npu

from .kda_fwd import _check_public_inputs, _run_ascendc


class CapturedForward:
    """Replay against the tensors supplied to :func:`capture_fwd`.

    Shapes, addresses and scalar parameters are fixed. Update input values in
    place, ordered on the replay stream, then call ``replay()``. The object
    retains the tensors and the graph's private scratch pool. Use distinct
    objects and output buffers for concurrent executions.
    """

    def __init__(self, graph, tensors, out, final_state):
        self._graph = graph
        self._tensors = tensors
        self._device = out.device
        self.out = out
        self.final_state = final_state

    def replay(self) -> None:
        """Enqueue the captured forward on the device's current stream."""
        with torch.npu.device(self._device):
            self._graph.replay()


def capture_fwd(
    q, k, v, g, beta, scale, out, A_log, dt_bias, lower_bound=-5.0,
    initial_state=None, final_state=None, *, warmup=3,
) -> CapturedForward:
    """Warm up and capture a fixed-length forward with caller-owned buffers.

    Initial and final state must use separate storage. Capture executes the
    forward and synchronizes the device. Compilation and
    capture are one-time setup costs, excluded from replay timing. As with
    ``fwd``, output tensors are written in place. Packed ``cu_seqlens`` and
    changing shapes/addresses require the regular forward API. Stage profiling
    must be disabled (``KDA_PROFILE=0``) during capture.
    """
    if os.environ.get("KDA_PROFILE", "0") == "1":
        raise ValueError("capture requires KDA_PROFILE=0 (stage profiling synchronizes)")
    if warmup < 1:
        raise ValueError("warmup must be positive")
    if initial_state is not None and final_state is not None:
        if initial_state.untyped_storage().data_ptr() == final_state.untyped_storage().data_ptr():
            raise ValueError("capture requires separate initial and final state storage")
    _check_public_inputs(q, k, v, g, beta, out, A_log, dt_bias,
                         initial_state, final_state, None)
    tensors = (q, k, v, g, beta, out, A_log, dt_bias, initial_state, final_state)

    def run():
        # Use the execution layer directly: public fwd may enable automatic
        # capture, which would recursively enter this capture helper.
        _run_ascendc(q, k, v, g, beta, float(scale), out, A_log, dt_bias,
                     float(lower_bound), initial_state, final_state, None)

    with torch.npu.device(q.device):
        for _ in range(warmup):
            run()
        torch.npu.synchronize()
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            run()
    return CapturedForward(graph, tensors, out, final_state)
