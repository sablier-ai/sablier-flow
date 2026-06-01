"""Backtrader adapter — wrap synthetic paths as `bt.feeds.PandasData`.

Requires the ``[adapters-backtrader]`` extra
(``pip install 'sablier-flow[adapters-backtrader]'``). Backtrader is the
most widely-deployed open-source backtest engine in the Python ecosystem
outside QuantConnect / LEAN; this adapter is the day-1 promise.

Usage::

    fit = sf.fit(real_data, train_split=0.8)
    paths = sf.generate(fit.model_id, n_paths=1000, like=backtest_window)
    feeds = as_backtrader_feeds(paths, ticker_column='AAPL')
    for feed in feeds:
        cerebro = bt.Cerebro()
        cerebro.adddata(feed)
        cerebro.addstrategy(MyStrategy)
        cerebro.run()
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pandas as pd

if TYPE_CHECKING:
    from sablier_flow.types import GenerationResult

__all__ = ["as_backtrader_feeds"]


def as_backtrader_feeds(
    result: GenerationResult,
    *,
    ticker_column: str | None = None,
    index: pd.DatetimeIndex | None = None,
) -> list[Any]:
    """Convert a GenerationResult into a list of backtrader PandasData feeds.

    Parameters
    ----------
    result
        A :class:`~sablier_flow.GenerationResult`.
    ticker_column
        Which feature to use as the close-price column. ``None`` uses the
        first feature in ``result.feature_names``. If only one feature
        is present, this is unambiguous.
    index
        Optional DatetimeIndex of length ``result.horizon``. Each feed
        gets this same index; if absent, a synthetic business-day range
        starting from today is used.

    Returns
    -------
    list of ``backtrader.feeds.PandasData``
        One feed per synthetic path. Each has columns
        ``[open, high, low, close, volume, openinterest]`` derived from
        the chosen feature column (open=high=low=close, vol=0).

    Notes
    -----
    Backtrader expects OHLCV columns; FLOW outputs close-like price
    levels. We synthesize OHLC by repeating the close (intraday
    structure is out of scope for the daily/weekly horizon FLOW models).
    For full intraday support, write your own adapter that maps your
    synthetic generator's multi-feature output to OHLC.
    """
    try:
        import backtrader as bt
    except ImportError as exc:
        raise ImportError(
            "backtrader is not installed. "
            "Install with: pip install 'sablier-flow[adapters-backtrader]'"
        ) from exc

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
        # Default: business days starting today
        index = pd.bdate_range(start=pd.Timestamp.today().normalize(), periods=result.horizon)
    elif len(index) != result.horizon:
        raise ValueError(
            f"index has {len(index)} rows; expected {result.horizon}"
        )

    feeds: list[Any] = []
    for i in range(prices.shape[0]):
        close = prices[i]
        df = pd.DataFrame(
            {
                "open": close,
                "high": close,
                "low": close,
                "close": close,
                "volume": 0.0,
                "openinterest": 0.0,
            },
            index=index,
        )
        feeds.append(bt.feeds.PandasData(dataname=df))
    return feeds
