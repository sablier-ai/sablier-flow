"""LEAN / QuantConnect adapter — write synthetic paths to LEAN's CSV
universe format.

LEAN ingests bar data from a per-symbol directory tree of compressed
CSVs. ``write_lean_csv_universe`` writes one synthetic alternative
history out as a LEAN-ingestable bundle: each path becomes its own
"date" subdirectory (so the same backtest can be re-run against each
synthetic world by pointing LEAN's data folder at the right slice).

Directory layout produced (matches LEAN's data/equity/usa/daily/<ticker>.zip
convention but uses unzipped CSV for simpler iteration):

    out_dir/
      path_0000/
        equity/usa/daily/AAPL.csv
        equity/usa/daily/MSFT.csv
      path_0001/
        equity/usa/daily/...
      ...

Each CSV row is the LEAN daily-bar wire format:
    yyyymmdd HH:MM,open,high,low,close,volume

Bars are synthesized from the path's price series:
  - open/close come from consecutive points
  - high/low are min/max of (open, close)
  - volume is a placeholder (1_000_000)

The intent is to drop these into the LEAN Cloud Data folder + run the
exact same algorithm N times. Output is reproducible from the same
:class:`GenerationResult`.

Day-30 deliverable on a follow-up release; ships in v0.0.2.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from sablier_flow.types import GenerationResult


__all__ = ["write_lean_csv_universe"]


def write_lean_csv_universe(
    result: GenerationResult,
    out_dir: Path | str,
    *,
    index: pd.DatetimeIndex | None = None,
    volume_placeholder: int = 1_000_000,
    overwrite: bool = False,
) -> Path:
    """Write the synthetic paths out as a LEAN-compatible CSV bundle.

    Parameters
    ----------
    result
        The :class:`GenerationResult` from :func:`generate_paths`.
    out_dir
        Directory to write into. Subdirectories ``path_NNNN/equity/usa/daily/``
        are created. Files named ``<TICKER>.csv``.
    index
        Optional DatetimeIndex to stamp every bar with. Must have
        length ``result.horizon``. Defaults to a business-day index
        starting today.
    volume_placeholder
        LEAN requires a volume column; we synthesize a constant.
        Bump if your algorithm depends on absolute volume.
    overwrite
        If False (default), raise if ``out_dir`` already exists.
        If True, files are overwritten in place.

    Returns
    -------
    Path
        ``out_dir`` (the root of the LEAN bundle).
    """
    out = Path(out_dir)
    if out.exists() and not overwrite:
        raise FileExistsError(
            f"{out} already exists. Pass overwrite=True to clobber."
        )
    out.mkdir(parents=True, exist_ok=overwrite)

    if index is None:
        index = pd.bdate_range(pd.Timestamp.today().normalize(), periods=result.horizon)
    if len(index) != result.horizon:
        raise ValueError(
            f"index length {len(index)} != result.horizon {result.horizon}"
        )

    prices = np.asarray(result.paths_prices)
    if prices.ndim != 3:
        raise ValueError(
            f"paths_prices must be (n_paths, horizon, n_features); got shape {prices.shape}"
        )
    n_paths, _horizon, n_features = prices.shape
    if n_features != len(result.feature_names):
        raise ValueError(
            f"feature_names length {len(result.feature_names)} != prices last-dim {n_features}"
        )

    for p in range(n_paths):
        path_dir = out / f"path_{p:04d}" / "equity" / "usa" / "daily"
        path_dir.mkdir(parents=True, exist_ok=overwrite)
        for f, ticker in enumerate(result.feature_names):
            csv_path = path_dir / f"{ticker.upper()}.csv"
            close = prices[p, :, f]
            # LEAN daily bars don't carry intraday H/L for synthetic data —
            # we use close-to-close for open/high/low/close, leaving high/low
            # as the per-bar range (small but non-zero so LEAN accepts it).
            opens = np.concatenate([[close[0]], close[:-1]])
            highs = np.maximum(opens, close) * 1.0001
            lows = np.minimum(opens, close) * 0.9999
            df = pd.DataFrame(
                {
                    "open": opens,
                    "high": highs,
                    "low": lows,
                    "close": close,
                    "volume": volume_placeholder,
                },
                index=index,
            )
            # LEAN daily format: yyyymmdd HH:MM,open,high,low,close,volume
            df.index = df.index.strftime("%Y%m%d 00:00")
            df.to_csv(csv_path, header=False, float_format="%.6f")
    return out
