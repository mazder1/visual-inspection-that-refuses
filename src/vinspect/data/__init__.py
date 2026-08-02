"""Dataset indexing, loading, grouping and splitting.

Re-exports are lazy so that ``python -m vinspect.data.mvtec`` does not import
the module twice, which would otherwise emit a runpy warning on every run.
"""

from typing import Any

_EXPORTS = {
    "CATEGORIES": "vinspect.data.mvtec",
    "OBJECT_CATEGORIES": "vinspect.data.mvtec",
    "TEXTURE_CATEGORIES": "vinspect.data.mvtec",
    "MVTecDataset": "vinspect.data.mvtec",
    "MVTecLayoutError": "vinspect.data.mvtec",
    "MVTecRecord": "vinspect.data.mvtec",
    "format_inventory": "vinspect.data.mvtec",
    "index_mvtec": "vinspect.data.mvtec",
    "summarise": "vinspect.data.mvtec",
    "GroupingResult": "vinspect.data.grouping",
    "group_records": "vinspect.data.grouping",
    "group_by_hash": "vinspect.data.grouping",
    "group_by_keypoints": "vinspect.data.grouping",
    "inlier_distribution": "vinspect.data.grouping",
    "dump_clusters": "vinspect.data.grouping",
    "SplitError": "vinspect.data.splits",
    "assign_grouped": "vinspect.data.splits",
    "assign_random": "vinspect.data.splits",
    "load_split": "vinspect.data.splits",
    "records_for_split": "vinspect.data.splits",
    "verify_no_leakage": "vinspect.data.splits",
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str) -> Any:
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    return getattr(importlib.import_module(_EXPORTS[name]), name)
