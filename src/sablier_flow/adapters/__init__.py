"""Engine adapters — universal DataFrame + per-engine helpers.

Built-in:
  - :func:`as_dataframes`        — list of pd.DataFrame, same schema as input
  - :func:`as_array`             — 3-D ndarray (n_paths, horizon, n_features)
  - :func:`as_backtrader_feeds`  — list of bt.feeds.PandasData (extra: backtrader)
  - :func:`as_vectorbt_panel`    — wide (T, n_paths) DataFrame (extra: vectorbt)

Community-contributed adapters can register themselves via the
``sablier_flow.adapters`` entry-point group in their own package's
pyproject.toml::

    [project.entry-points."sablier_flow.adapters"]
    nautilus = "sablier_nautilus:as_nautilus_catalog"

Once installed, the name resolves via attribute lookup
(``sablier_flow.adapters.as_nautilus_catalog``) and listing via
:func:`available_adapters`.
"""

from importlib.metadata import entry_points
from typing import Any

from sablier_flow.adapters.dataframe import as_array, as_dataframes

__all__ = [
    "as_array",
    "as_backtrader_feeds",
    "as_dataframes",
    "as_vectorbt_panel",
    "available_adapters",
    "write_lean_csv_universe",
]


_ENTRY_POINT_GROUP = "sablier_flow.adapters"


def _discover_entry_points() -> dict[str, Any]:
    """Map adapter name → loaded callable for every package that registers
    under the ``sablier_flow.adapters`` entry-point group.

    Looked up lazily on each access so adapters installed in the same
    process post-import (rare, but happens in tests) are picked up.
    """
    discovered: dict[str, Any] = {}
    # entry_points(group=...) was added in Python 3.10 — we require 3.10+
    # per pyproject so this is safe without a pre-3.10 fallback.
    eps = entry_points(group=_ENTRY_POINT_GROUP)
    for ep in eps:
        try:
            discovered[ep.name] = ep.load()
        except Exception:
            continue
    return discovered


def available_adapters() -> list[str]:
    """Return the names of all adapters available right now — built-ins
    plus anything registered via the ``sablier_flow.adapters`` entry-point
    group. Useful for the ``sablier-flow adapters list`` CLI command
    and for diagnostics.
    """
    builtins = ["as_array", "as_dataframes", "as_backtrader_feeds", "as_vectorbt_panel"]
    return sorted(set(builtins) | set(_discover_entry_points()))


def __getattr__(name: str) -> Any:
    """Lazy-load engine-specific adapters so the base import never pulls in
    backtrader / vectorbt unless the customer actually uses them. Falls
    back to entry-point discovery for community-contributed adapters."""
    if name == "as_backtrader_feeds":
        from sablier_flow.adapters.backtrader import as_backtrader_feeds
        return as_backtrader_feeds
    if name == "as_vectorbt_panel":
        from sablier_flow.adapters.vectorbt import as_vectorbt_panel
        return as_vectorbt_panel
    if name == "write_lean_csv_universe":
        # 1.0.21 — `from sablier_flow.adapters import write_lean_csv_universe`
        # used to raise ImportError because the lazy __getattr__ + __all__
        # only routed backtrader / vectorbt. The lean adapter has no extra
        # deps (writes CSVs with stdlib), so this is a pure namespace fix.
        from sablier_flow.adapters.lean import write_lean_csv_universe
        return write_lean_csv_universe
    discovered = _discover_entry_points()
    if name in discovered:
        return discovered[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
