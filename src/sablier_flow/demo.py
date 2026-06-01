"""First-touch demo datasets bundled in the wheel.

Lets an evaluator run the SDK end-to-end with zero data setup::

    import sablier_flow

    real = sablier_flow.demo_data()         # SPY/QQQ/IWM/TLT, 2010-2024
    @sablier_flow.augment(n_paths=100, features=list(real.columns))
    def my_backtest(df):
        return {"sharpe": ...}
    print(my_backtest(real).summary())

The bundled data is a frozen historical slice — **not** market-realtime,
**not** a production data source. It exists purely so a first-time
evaluator can install the SDK and see a real verdict before deciding
whether to wire up their own data pipeline.

For production use, source data through whatever pipeline the customer's
fund already runs (ArcticDB, Bloomberg BQuant, kdb+/pykx, internal
warehouse, Polars, etc.) and pass the resulting DataFrame to the SDK.
See ``docs/SDK.md`` recipes section.
"""

from __future__ import annotations

from importlib import resources
from typing import Literal

import pandas as pd

__all__ = [
    "DEMO_DATA_TYPES",
    "DemoDatasetName",
    "available_demo_datasets",
    "demo_data",
]


DemoDatasetName = Literal[
    "us_equities_2010_2024",
    "us_equities_macro_2010_2024",
    "us_equities_macro_5min_3mo",
]


# Canonical per-column ``data_type`` annotations for every bundled demo.
#
# These are attached to ``df.attrs['data_types']`` so an evaluator can pass
# the dict straight through to ``sf.fit(..., data_types=df.attrs['data_types'])``
# without having to classify the columns themselves. Allowed values must
# stay in lockstep with the SDK's canonical allowed set:
#     {'price', 'return', 'rate', 'index', 'volatility'}
#
# Choices:
#   - SPY / QQQ / IWM / TLT / VIXY / IEF / UUP    -> 'price'  (tradeable ETFs)
#   - VIX                                         -> 'volatility'
#   - TNX (10Y constant-maturity yield, %)        -> 'rate'
#   - DXY (price-weighted dollar-index basket)    -> 'index'
DEMO_DATA_TYPES: dict[str, dict[str, str]] = {
    "us_equities_2010_2024": {
        "SPY": "price",
        "QQQ": "price",
        "IWM": "price",
        "TLT": "price",
    },
    "us_equities_macro_2010_2024": {
        "SPY": "price",
        "QQQ": "price",
        "IWM": "price",
        "TLT": "price",
        "VIX": "volatility",
        "TNX": "rate",
        "DXY": "index",
    },
    "us_equities_macro_5min_3mo": {
        # Intraday substitutes for the daily-only macros (VIX/TNX/DXY have no
        # intraday endpoint on common data providers, so we ship tradeable
        # proxies — every column in this set is a tradeable ETF price).
        "SPY": "price",
        "QQQ": "price",
        "IWM": "price",
        "TLT": "price",
        "VIXY": "price",
        "IEF": "price",
        "UUP": "price",
    },
}


def available_demo_datasets() -> list[str]:
    """Return the names of bundled demo datasets that ``demo_data`` accepts."""
    return [
        "us_equities_2010_2024",
        "us_equities_macro_2010_2024",
        "us_equities_macro_5min_3mo",
    ]


def demo_data(name: DemoDatasetName = "us_equities_macro_2010_2024") -> pd.DataFrame:
    """Load a bundled demo DataFrame, no network access required.

    Parameters
    ----------
    name
        Which bundled dataset to load. Available options:

          - ``"us_equities_macro_2010_2024"`` (default) — SPY, QQQ, IWM, TLT
            equity ETFs + VIX, TNX (10Y yield), DXY (dollar index) macro
            series. 3522 rows × 7 columns, daily, 2010-01-04 → 2023-12-28.
            Recommended starting point because the structural-validation
            metrics need regime context to pass — equity prices alone
            don't carry enough signal for the model to calibrate
            uncertainty.
          - ``"us_equities_2010_2024"`` — SPY/QQQ/IWM/TLT only, no macros
            (4 columns). Kept for backwards compat / minimal-input demos;
            validate() typically flunks calibration metrics on this set
            because the model has no regime context.
          - ``"us_equities_macro_5min_3mo"`` — SPY, QQQ, IWM, TLT + VIXY
            (vol-future ETF), IEF (10Y Treasury ETF), UUP (dollar bull
            ETF) at 5-minute granularity. 5781 rows × 7 columns spanning
            ~3 calendar months. Use this to exercise the intraday code
            path (auto-detected via ``pd.infer_freq``) and to drive
            intraday backtests. VIX/TNX/DXY have no intraday endpoint on
            common data providers, so we substitute tradeable proxies
            with similar regime signal.

        Call :func:`available_demo_datasets` for the full list.

    Returns
    -------
    pd.DataFrame
        DatetimeIndex (business-day frequency), one float32 column per
        feature. Equity ETFs are price series; VIX/TNX/DXY are index
        levels. ``df.attrs['data_types']`` carries the per-column
        ``data_type`` annotation the SDK requires, so the DataFrame
        is drop-in for :func:`sablier_flow.fit`::

            real = sablier_flow.demo_data()
            fit = sablier_flow.fit(
                real,
                features=['SPY', 'QQQ', 'IWM', 'TLT', 'VIX', 'TNX', 'DXY'],
                data_types=real.attrs['data_types'],
                train_split=0.8,
            )
    """
    # Import pandas lazily so ``import sablier_flow`` stays cheap.
    import pandas as pd

    valid = available_demo_datasets()
    if name not in valid:
        raise ValueError(
            f"unknown demo dataset {name!r} — available: {valid}"
        )

    pkg = "sablier_flow._demo_data"
    fname = f"{name}.parquet"
    # importlib.resources.files() lands the file regardless of whether
    # the package is installed from a wheel, a zipfile, or source.
    path = resources.files(pkg).joinpath(fname)
    with resources.as_file(path) as local_path:
        df = pd.read_parquet(local_path)
    # Ensure DatetimeIndex; round-trip through Parquet preserves it but
    # some readers strip the frequency metadata.
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    df.index.name = "date"
    df = df.sort_index()
    # Attach the canonical per-column ``data_type`` map so the DataFrame is
    # drop-in for the ``data_types=`` contract on ``fit / generate /
    # validate``. We restrict to columns actually present so a future
    # subset (e.g. ``df[['SPY', 'QQQ']]``) can pull the right slice.
    type_map = DEMO_DATA_TYPES.get(name, {})
    df.attrs["data_types"] = {
        col: type_map[col] for col in df.columns if col in type_map
    }
    return df
