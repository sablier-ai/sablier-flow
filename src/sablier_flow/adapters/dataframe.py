"""Universal DataFrame / NumPy materializers.

These work with any backtest engine that consumes pandas DataFrames or
NumPy arrays — which is every engine in the universe of relevance
(vectorbt, backtesting.py, raw pandas, in-house C++/KDB ingest layers,
LEAN's BaseData adapter, …).

Given a :class:`~sablier_flow.GenerationResult`, you
get N DataFrames with the same schema as the input. Drop into your
existing backtest loop unchanged.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from sablier_flow.types import GenerationResult

__all__ = [
    "as_array",
    "as_dataframes",
]


def as_dataframes(
    result: GenerationResult,
    *,
    index: pd.DatetimeIndex | None = None,
) -> list[pd.DataFrame]:
    """Convert a GenerationResult into a list of DataFrames.

    Each DataFrame has:
      - the same columns as the customer's input (internal cyclical
        embeddings stripped)
      - shape ``(horizon, n_features)``
      - a DatetimeIndex if ``index`` is provided; otherwise a default
        integer index (the customer should overlay their own date index
        for direct ingest by their backtest)

    Use case::

        fit = sf.fit(real_data, train_split=0.8)
        paths = sf.generate(fit.model_id, n_paths=1000, like=backtest_window)
        dfs = as_dataframes(paths)                  # uses backtest_window.index
        # Or equivalently, since like= sets paths_index:
        #     dfs = paths.as_dataframes()
        results = [my_backtest(df) for df in dfs]   # your existing backtest, unchanged
    """
    if index is not None and len(index) != result.horizon:
        raise ValueError(
            f"index has {len(index)} rows; expected {result.horizon} "
            f"(GenerationResult.horizon)"
        )
    paths_prices = result.paths_prices  # (n_paths, horizon, n_features)
    out: list[pd.DataFrame] = []
    for i in range(paths_prices.shape[0]):
        df = pd.DataFrame(
            paths_prices[i],
            columns=result.feature_names,
            index=index if index is not None else None,
        )
        out.append(df)
    return out


def as_array(result: GenerationResult) -> np.ndarray:
    """Return the 3-D price-level array directly.

    Shape: ``(n_paths, horizon, n_real_features)``. This is what vectorbt
    and other vectorized backtest engines consume natively (they treat
    the path dimension as a parameter sweep).
    """
    return result.paths_prices
