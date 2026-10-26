"""Ascend NPU backend for FlashKDA."""

from .kda_fwd import fwd, is_available

__all__ = ["fwd", "is_available", "capture_fwd", "clear_auto_graph_cache"]


# Keep torch_npu optional when importing the package on other platforms.
def capture_fwd(*args, **kwargs):
    """Capture fixed-shape forward; see :mod:`flash_kda_npu.graphs`."""
    from .graphs import capture_fwd as capture
    return capture(*args, **kwargs)


def clear_auto_graph_cache():
    """Synchronize and release automatic graphs cached by this thread."""
    from .auto_graph import clear_auto_graph_cache as clear
    clear()
