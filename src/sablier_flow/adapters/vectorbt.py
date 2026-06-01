"""vectorbt adapter — stack synthetic paths into a multi-column DataFrame.

Requires the ``[adapters-vectorbt]`` extra
(``pip install 'sablier-flow[adapters-vectorbt]'``). vectorbt is the
fastest open-source backtest engine for parameter sweeps and multi-path
analysis, so it's a natural fit for evaluating N synthetic alternative
histories in one call.

Usage::

    import vectorbt as vbt

    fit = sf.fit(real_data, train_split=0.8)
    paths = sf.generate(fit.model_id, n_paths=1000, like=backtest_window)
    df = as_vectorbt_panel(paths, ticker_column='AAPL')
    # df shape: (horizon, n_paths) — each column is one alternative history

    # Vectorized backtest across all 1000 paths at once
    fast_ma = vbt.MA.run(df, window=10)
    slow_ma = vbt.MA.run(df, window=30)
    entries = fast_ma.ma_above(slow_ma, crossover=True)
    exits = fast_ma.ma_below(slow_ma, crossover=True)
    pf = vbt.Portfolio.from_signals(df, entries, exits)
    print(pf.sharpe_ratio())   # one Sharpe per synthetic path
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from sablier_flow.types import GenerationResult

__all__ = ["as_vectorbt_panel"]


def as_vectorbt_panel(
    result: GenerationResult,
    *,
    ticker_column: str | None = None,
    index: pd.DatetimeIndex | None = None,
    column_prefix: str = "path",
) -> pd.DataFrame:
    """Convert a GenerationResult into a wide DataFrame for vectorbt.

    Parameters
    ----------
    result
        A :class:`~sablier_flow.GenerationResult`.
    ticker_column
        Which feature to stack across paths. ``None`` uses the first
        feature in ``result.feature_names``.
    index
        Optional DatetimeIndex of length ``result.horizon``. Default:
        business days from today.
    column_prefix
        Column-name prefix for each synthetic path. Default ``"path"``
        gives columns ``path_0000``, ``path_0001``, ….

    Returns
    -------
    DataFrame
        Shape ``(horizon, n_paths)``. Each column is one synthetic
        alternative history of the selected ticker. Drop directly into
        ``vbt.Portfolio.from_signals(df, ...)``.

    Notes
    -----
    For multi-asset strategies, call this function once per asset and
    feed the resulting DataFrames into vectorbt's multi-asset API. FLOW
    generates joint paths so the per-asset DataFrames are *aligned* by
    path index (path_0042 for AAPL and path_0042 for MSFT come from the
    same correlated draw).
    """
    if ticker_column is None:
        ticker_column = result.feature_names[0]
    if ticker_column not in result.feature_names:
        raise ValueError(
            f"ticker_column={ticker_column!r} not in feature_names "
            f"{result.feature_names}"
        )

    col_idx = result.feature_names.index(ticker_column)
    prices = result.paths_prices[:, :, col_idx]  # (n_paths, horizon)

    if index is None:
        index = pd.bdate_range(start=pd.Timestamp.today().normalize(), periods=result.horizon)
    elif len(index) != result.horizon:
        raise ValueError(
            f"index has {len(index)} rows; expected {result.horizon}"
        )

    n_paths = prices.shape[0]
    width = max(4, len(str(n_paths - 1)))
    columns = [f"{column_prefix}_{i:0{width}d}" for i in range(n_paths)]

    # vectorbt wants a (T, N) DataFrame — transpose from (n_paths, T) to (T, n_paths)
    df = pd.DataFrame(prices.T, index=index, columns=columns)
    return df
