"""``sablier-flow`` console entry point.

Why a CLI: the Python SDK only reaches Python users. R / MATLAB / Q-KDB+ /
shell-pipeline / CI/CD users all need a non-Python on-ramp. Parquet in,
Parquet out, exit code = success. No language SDK required.

Commands:

    sablier-flow version
    sablier-flow adapters list
    sablier-flow generate --input prices.parquet --n 100 --out ./paths/
    sablier-flow robustness --real real.json --synthetic synth.json [--metric sharpe]

The implementation uses argparse only (no new dependencies). Each
subcommand is a small function and is unit-tested by calling it
directly with the parsed namespace.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

__all__ = ["build_parser", "main"]


# ============================================================================
# Parser construction (also used by tests so they don't shell out)
# ============================================================================


def build_parser() -> argparse.ArgumentParser:
    """Build the top-level parser. Public so tests / docs can introspect it."""
    p = argparse.ArgumentParser(
        prog="sablier-flow",
        description=(
            "Synthetic alternative-history generation for backtest "
            "overfitting detection. See https://docs.sablier.ai."
        ),
    )
    sub = p.add_subparsers(dest="command", required=True, metavar="<command>")

    # version
    sub.add_parser("version", help="Print SDK version and exit.")

    # notebook — copies the bundled getting-started notebook into the
    # current directory (or --out) so a customer can `jupyter notebook
    # 00_getting_started.ipynb` without cloning anything.
    nb = sub.add_parser(
        "notebook",
        help="Copy the bundled getting-started notebook into the current directory.",
    )
    nb.add_argument(
        "--out", "-o", type=Path, default=Path("00_getting_started.ipynb"),
        help="Where to write the notebook (default: ./00_getting_started.ipynb).",
    )
    nb.add_argument(
        "--force", action="store_true",
        help="Overwrite an existing file at --out.",
    )

    # adapters list
    adapters = sub.add_parser("adapters", help="Engine adapter introspection.")
    adapters_sub = adapters.add_subparsers(dest="subcommand", required=True)
    adapters_sub.add_parser("list", help="List available engine adapters.")

    # generate
    gen = sub.add_parser(
        "generate",
        help="Generate N synthetic alternative-history datasets from a real one.",
    )
    gen.add_argument("--input", "-i", required=True, type=Path,
                     help="Path to input dataset (Parquet or CSV).")
    gen.add_argument("--n", "-n", required=True, type=int,
                     help="Number of synthetic paths to generate.")
    gen.add_argument("--out", "-o", required=True, type=Path,
                     help="Output directory; one Parquet file per path will be written.")
    gen.add_argument("--horizon", type=int, default=None,
                     help="Generation horizon in periods. Defaults to len(input).")
    gen.add_argument("--seed", type=int, default=None,
                     help="Deterministic seed (otherwise the server picks one).")
    gen.add_argument("--features", default=None,
                     help="Comma-separated feature names; defaults to all input columns.")
    gen.add_argument("--api-key", default=None,
                     help="Override SABLIER_FLOW_API_KEY env var.")
    gen.add_argument("--idempotency-key", default=None,
                     help="Stable key to deduplicate retried requests on the server.")

    # robustness
    rb = sub.add_parser(
        "robustness",
        help="Compute a robustness verdict from a real backtest + N synthetic backtests.",
    )
    rb.add_argument("--real", required=True, type=Path,
                    help="JSON file containing the real backtest result "
                         "(scalar or dict of metrics).")
    rb.add_argument("--synthetic", required=True, type=Path,
                    help="JSON file containing a list of synthetic backtest results.")
    rb.add_argument("--metric", default=None,
                    help="Primary metric (defaults to 'sharpe' or first key).")
    rb.add_argument("--lower-is-better", action="store_true",
                    help="Flip orientation (use for drawdown-style metrics).")
    rb.add_argument("--html", type=Path, default=None,
                    help="If set, also write an HTML report to this path.")

    return p


# ============================================================================
# Command implementations
# ============================================================================


def cmd_version(_ns: argparse.Namespace) -> int:
    from sablier_flow import __version__
    print(__version__)
    return 0


def cmd_notebook(ns: argparse.Namespace) -> int:
    """Copy the bundled getting-started notebook to ns.out."""
    import shutil

    import sablier_flow

    src = Path(sablier_flow.__file__).parent / "_resources" / "getting_started.ipynb"
    if not src.exists():
        print(
            f"error: notebook resource missing at {src} — your sablier-flow "
            f"install may be incomplete. Try `pip install --upgrade --force-reinstall sablier-flow`.",
            file=sys.stderr,
        )
        return 2
    out: Path = ns.out
    if out.exists() and not ns.force:
        print(
            f"error: {out} already exists. Pass --force to overwrite, or "
            f"use --out to write somewhere else.",
            file=sys.stderr,
        )
        return 2
    out.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, out)
    print(f"wrote {out}")
    print(f"open with: jupyter notebook {out}")
    return 0


def cmd_adapters_list(_ns: argparse.Namespace) -> int:
    from sablier_flow.adapters import available_adapters
    for name in available_adapters():
        print(name)
    return 0


def cmd_generate(ns: argparse.Namespace) -> int:
    import pandas as pd

    api_key = ns.api_key or os.environ.get("SABLIER_FLOW_API_KEY")
    if not api_key:
        print("error: no API key — pass --api-key or set SABLIER_FLOW_API_KEY", file=sys.stderr)
        return 2

    real = _load_dataset(ns.input)

    features = (
        [s.strip() for s in ns.features.split(",")] if ns.features else None
    )

    from sablier_flow.adapters.dataframe import as_dataframes
    from sablier_flow.client.client import Client

    client = Client(api_key=api_key)
    fit_res = client.fit(
        real,
        features=features,
        horizon=ns.horizon,
        seed=ns.seed,
    )
    result = client.generate(
        fit_res.model_id,
        n_paths=ns.n,
        horizon=ns.horizon,
        seed=ns.seed,
        idempotency_key=ns.idempotency_key,
    )

    out_dir: Path = ns.out
    out_dir.mkdir(parents=True, exist_ok=True)

    # Try to preserve the real data's DatetimeIndex on synthetic outputs when
    # horizons match. Otherwise leave the default integer index.
    overlay_index = real.index if (
        isinstance(real.index, pd.DatetimeIndex) and len(real.index) == result.horizon
    ) else None

    dfs = as_dataframes(result, index=overlay_index)
    for i, df in enumerate(dfs):
        out_path = out_dir / f"path_{i:05d}.parquet"
        df.to_parquet(out_path)

    print(f"wrote {len(dfs)} synthetic datasets to {out_dir}")
    if result.memorization_risk:
        print(f"memorization risk: {result.memorization_risk}")
    return 0


def cmd_robustness(ns: argparse.Namespace) -> int:
    from sablier_flow.analytics.robustness import robustness

    real = json.loads(ns.real.read_text())
    synthetic = json.loads(ns.synthetic.read_text())
    if not isinstance(synthetic, list):
        print("error: --synthetic must be a JSON list of scalars or dicts", file=sys.stderr)
        return 2

    report = robustness(
        real,
        synthetic,
        primary_metric=ns.metric,
        higher_is_better=not ns.lower_is_better,
    )

    print(f"verdict        : {report.verdict}")
    print(f"overfit_score  : {report.overfit_score:.3f}")
    print(f"primary_metric : {report.primary_metric}")
    print(f"real value     : {report.real_value:+.4f}")
    print(f"synth median   : {report.synthetic_median:+.4f}")
    print(f"synth 5/95 pct : {report.synthetic_p5:+.4f} / {report.synthetic_p95:+.4f}")
    print(f"n synthetic    : {report.n_synthetic}")
    for note in report.notes:
        print(f"note: {note}")

    if ns.html is not None:
        report.to_html(str(ns.html))
        print(f"html report    : {ns.html}")
    return 0


# ============================================================================
# Entry point
# ============================================================================


_COMMANDS: dict[tuple[str, str | None], Any] = {
    ("version", None): cmd_version,
    ("notebook", None): cmd_notebook,
    ("adapters", "list"): cmd_adapters_list,
    ("generate", None): cmd_generate,
    ("robustness", None): cmd_robustness,
}


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    ns = parser.parse_args(argv)
    sub = getattr(ns, "subcommand", None)
    handler = _COMMANDS.get((ns.command, sub))
    if handler is None:
        parser.error(f"unknown command: {ns.command} {sub or ''}".strip())
    return int(handler(ns))


# ============================================================================
# Helpers
# ============================================================================


def _load_dataset(path: Path) -> Any:
    """Load a DataFrame from .parquet or .csv. Detects format by suffix.

    Returns ``Any`` rather than ``pd.DataFrame`` because pandas is a deferred
    import inside the function — the CLI top-level should stay importable
    without pandas installed (e.g. for ``sablier-flow version`` / ``adapters
    list`` which don't need it).
    """
    import pandas as pd

    if not path.exists():
        raise FileNotFoundError(path)
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix == ".csv":
        return pd.read_csv(path, index_col=0, parse_dates=True)
    raise ValueError(f"unsupported input format: {suffix} (use .parquet or .csv)")


if __name__ == "__main__":
    sys.exit(main())
