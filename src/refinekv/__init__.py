"""RefineKV: query-adaptive block refinement for cached attention."""

from .attention import AttentionResult, select_blocks, sparse_attention
from .config import RefineKVConfig
from .summary import BlockSummary, build_summary

__version__ = "0.1.2"
__all__ = [
    "AttentionResult",
    "BlockSummary",
    "RefineKVConfig",
    "build_summary",
    "select_blocks",
    "sparse_attention",
    "RefineKV",
    "make_visible_static_cache",
]


def __getattr__(name):
    if name in ("RefineKV", "make_visible_static_cache"):
        from .integrations.transformers import RefineKV, make_visible_static_cache

        return {"RefineKV": RefineKV, "make_visible_static_cache": make_visible_static_cache}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
