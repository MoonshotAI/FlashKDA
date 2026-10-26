"""Bounded, thread/stream-local graph reuse for the regular forward API.

Each signature advances from eager warmup to capture, then replay. A False
return delegates the current call to the public entry point's eager path.
The cache owns tensor references until eviction or explicit clearing.
"""
from collections import OrderedDict
import os
import threading

import torch
import torch_npu

_LOCAL = threading.local()
# Include these environment values in the cache key to avoid reusing a graph
# across configuration changes. Import-time options still require a fresh process.
_OPTIONS = ('KDA_PRE_UNROLL', 'KDA_PERSIST_LOOP_BLOCKS', 'KDA_PERSIST_HEAD_MAP',
            'KDA_SOLVE_SLICE_CHUNKS', 'KDA_SOLVE_SLICE_TARGET', 'KDA_SOLVE_OVERLAP')


def clear_auto_graph_cache():
    """Release this thread's cached graphs after their pending work completes."""
    cache = getattr(_LOCAL, 'cache', None)
    if cache:
        for entry in cache.values():
            entry['stream'].synchronize()
        cache.clear()


def _signature(tensor):
    """Identify storage and layout; values may be updated in place between calls."""
    if tensor is None:
        return None
    return (tensor.data_ptr(), tuple(tensor.shape), tuple(tensor.stride()),
            tensor.dtype, tensor.device)


def try_auto_graph(args):
    """Handle validated public fwd arguments; return whether output was produced.

    Packed inputs and profiling use eager execution. External capture also
    bypasses this cache so that the caller owns the entire captured graph.
    Capture/replay failures propagate rather than silently retrying a call.
    """
    (q, k, v, g, beta, scale, out, a_log, bias, lower_bound,
     initial, final, cu_seqlens) = args
    if cu_seqlens is not None or os.environ.get('KDA_PROFILE', '0') == '1':
        return False
    tensors = (q, k, v, g, beta, out, a_log, bias, initial, final)
    # Repeated setup/recording would change inputs when buffers alias outputs.
    output_stores = {x.untyped_storage().data_ptr() for x in (out, final) if x is not None}
    if any(x is not None and x.untyped_storage().data_ptr() in output_stores
           for x in (q, k, v, g, beta, a_log, bias, initial)):
        return False
    with torch.npu.device(q.device):
        if torch.npu.is_current_stream_capturing():
            return False
        stream = torch.npu.current_stream()
        capacity = int(os.environ.get('KDA_AUTO_GRAPH_CACHE_SIZE', '8'))
        warmup = int(os.environ.get('KDA_AUTO_GRAPH_WARMUP', '3'))
        if capacity < 1 or warmup < 1:
            raise ValueError('automatic graph cache size and warmup must be positive')
        cache = getattr(_LOCAL, 'cache', None)
        if cache is None:
            cache = _LOCAL.cache = OrderedDict()
        key = (int(stream.npu_stream), tuple(_signature(x) for x in tensors),
               scale, lower_bound, tuple(os.environ.get(name) for name in _OPTIONS))
        # Refresh recency so eviction preserves frequently reused graphs.
        entry = cache.pop(key, None)
        if entry is None:
            # Keep tensor references so reused addresses cannot match unrelated tensors.
            entry = dict(calls=0, graph=None, tensors=tensors, stream=stream)
        cache[key] = entry
        while len(cache) > capacity:
            _, retired = cache.popitem(last=False)
            # Graph storage may still be referenced by a pending replay.
            retired['stream'].synchronize()
        if entry['graph'] is not None:
            entry['graph'].replay()
            return True
        if entry['calls'] < warmup:
            entry['calls'] += 1
            return False
        from .graphs import capture_fwd
        # Eager calls initialize compilation and masks before capture; one final
        # warmup suffices for this already exercised signature.
        entry['graph'] = capture_fwd(
            q, k, v, g, beta, scale, out, a_log, bias, lower_bound,
            initial_state=initial, final_state=final, warmup=1,
        )
        entry['graph'].replay()
        return True
