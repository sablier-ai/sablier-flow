"""sablier-flow — synthetic alternative-history generation for backtest overfitting detection.

Customer-facing thin client + post-hoc analytics. The model architecture,
training loop, and validation methodology are not part of this wheel —
they run inside Sablier's hosted service and are invoked through
:class:`Client` (or the :func:`fit` / :func:`generate` / :func:`validate`
module-level shortcuts).

Canonical workflow::

    import sablier_flow as sf
    import numpy as np

    # 1. Auth
    sf.login()

    # 2. Your backtest
    def my_backtest(prices):
        rets = prices['SPY'].pct_change().dropna()
        return {'sharpe': float(rets.mean() / rets.std() * np.sqrt(252))} if rets.std() > 0 else {'sharpe': 0.0}

    # 3. Load data + define backtest window
    df = sf.demo_data()
    backtest_window = df.iloc[-252:]

    # 4. Fit + generate (with like= for shape match)
    fit   = sf.fit(df, features=list(df.columns), data_types=df.attrs['data_types'], horizon=252)
    paths = sf.generate(fit.model_id, n_paths=200, like=backtest_window)

    # 5. Score robustness
    real   = my_backtest(backtest_window)
    synth  = [my_backtest(p) for p in paths.as_dataframes()]
    report = sf.robustness(real, synth, primary_metric='sharpe')
    print(report.summary())

Public API:

    sablier_flow.Client                   — remote client talking to the hosted service
    sablier_flow.fit                      — train; auto 80/20 split + embargo for OOS
    sablier_flow.generate                 — emit synthetic paths (use ``like=window``)
    sablier_flow.validate                 — run validation on OOS held-out at fit time

Post-hoc analytics (client-side, no GPU):

    sablier_flow.robustness               — robustness report from real vs synthetic
    sablier_flow.deflated_sharpe          — DSR (Bailey–LdP) + realistic null variant
    sablier_flow.evaluate_family          — CSCV across a strategy family
    sablier_flow.probability_of_backtest_overfitting  — PBO directly
    sablier_flow.consistency_check        — live-vs-baseline drift signal

Demo + attestation helpers:

    sablier_flow.demo_data                — bundled SPY/QQQ/IWM/TLT plus 3 macro features (VIX, TNX, DXY) 2010-2023
    sablier_flow.AttestationVerifier      — checks the TEE's quote vs SDK-pinned digest
    sablier_flow.envelope_encrypt         — X25519+AES-GCM envelope encryption

Optional engine adapters (in :mod:`sablier_flow.adapters`):

    as_dataframes, as_array               — universal (DataFrame / NumPy)
    as_backtrader_feeds                   — requires [adapters-backtrader] extra
    as_vectorbt_panel                     — requires [adapters-vectorbt] extra
"""

from __future__ import annotations

from typing import Any

__version__ = "1.0.17"

__all__ = [
    "ALLOWED_DATA_TYPES",
    "ALLOWED_FREQUENCIES",
    "AttestationQuote",
    "AttestationVerificationError",
    "AttestationVerifier",
    "AuthenticationError",
    "Client",
    "ConsistencyReport",
    "CreditsBalance",
    "DeflatedSharpeReport",
    "EnvelopeEncrypted",
    "FamilyReport",
    "FitResult",
    "GenerationResult",
    "JobHandle",
    "JobNotFoundError",
    "Model",
    "ModelNotFoundError",
    "PredictiveRankReport",
    "RemoteJobError",
    "RobustnessReport",
    "SablierClientError",
    "TransportError",
    "UsageEvent",
    "UsageSummary",
    "ValidationReport",
    "__version__",
    "available_demo_datasets",
    "cancel_job",
    "consistency_check",
    "credits",
    "deflated_sharpe",
    "delete_model",
    "demo_data",
    "envelope_decrypt",
    "envelope_encrypt",
    "estimate_cost",
    "evaluate_family",
    "fetch_result",
    "fit",
    "fit_async",
    "generate",
    "generate_async",
    "get_model",
    "list_jobs",
    "list_models",
    "login",
    "logout",
    "ping",
    "predictive_rank_score",
    "probability_of_backtest_overfitting",
    "resume_job",
    "robustness",
    "usage",
    "usage_summary",
    "validate",
    "validate_async",
    "validate_data",
    "whoami",
]


def __getattr__(name: str) -> Any:
    """Lazy-load to keep ``import sablier_flow`` cold-start light.

    Customer-only surface: everything resolved here lives in either
    ``sablier_flow.client.*`` (Client + transport + attestation + crypto),
    ``sablier_flow.analytics.*`` (post-hoc verdicts),
    ``sablier_flow.types`` (wire dataclasses), ``sablier_flow.demo``
    (bundled dataset), or ``sablier_flow.adapters.*`` (engine helpers).

    The model architecture, training loop, and validation methodology
    live server-side and are intentionally **not** importable from this
    wheel.
    """
    # Data-layer contract constants — useful for customer
    # introspection (`assert col_type in sf.ALLOWED_DATA_TYPES`)
    # and round-trip checks in agent harnesses.
    if name == "ALLOWED_DATA_TYPES":
        from sablier_flow.client.client import ALLOWED_DATA_TYPES
        return ALLOWED_DATA_TYPES
    if name == "ALLOWED_FREQUENCIES":
        from sablier_flow.client.client import ALLOWED_FREQUENCIES
        return ALLOWED_FREQUENCIES

    # Client + connection-shape entry points
    if name == "Client":
        from sablier_flow.client.client import Client
        return Client
    if name == "fit":
        from sablier_flow.client.client import fit
        return fit
    if name == "generate":
        from sablier_flow.client.client import generate
        return generate
    if name == "validate":
        from sablier_flow.client.client import validate
        return validate
    if name == "list_models":
        from sablier_flow.client.client import list_models
        return list_models
    if name == "get_model":
        from sablier_flow.client.client import get_model
        return get_model
    if name == "delete_model":
        from sablier_flow.client.client import delete_model
        return delete_model
    if name == "ping":
        from sablier_flow.client.client import ping
        return ping
    if name == "whoami":
        from sablier_flow.client.client import whoami
        return whoami
    if name == "credits":
        from sablier_flow.client.client import credits
        return credits
    if name == "usage":
        from sablier_flow.client.client import usage
        return usage
    if name == "usage_summary":
        from sablier_flow.client.client import usage_summary
        return usage_summary
    if name == "estimate_cost":
        from sablier_flow.client.client import estimate_cost
        return estimate_cost
    if name == "validate_data":
        from sablier_flow.client.client import validate_data
        return validate_data
    if name == "fit_async":
        from sablier_flow.client.client import fit_async
        return fit_async
    if name == "generate_async":
        from sablier_flow.client.client import generate_async
        return generate_async
    if name == "validate_async":
        from sablier_flow.client.client import validate_async
        return validate_async
    if name == "fetch_result":
        from sablier_flow.client.client import fetch_result
        return fetch_result
    if name == "list_jobs":
        from sablier_flow.client.client import list_jobs
        return list_jobs
    if name == "cancel_job":
        from sablier_flow.client.client import cancel_job
        return cancel_job
    if name == "resume_job":
        from sablier_flow.client.client import resume_job
        return resume_job
    if name == "login":
        from sablier_flow.client.login import login
        return login
    if name == "logout":
        from sablier_flow.client.login import logout
        return logout

    # Public wire dataclasses
    if name in {
        "CreditsBalance",
        "FitResult",
        "GenerationResult",
        "JobHandle",
        "Model",
        "UsageEvent",
        "UsageSummary",
        "ValidationReport",
    }:
        from sablier_flow import types as _types
        return getattr(_types, name)

    # Post-hoc analytics (client-side, no GPU)
    if name == "robustness":
        from sablier_flow.analytics.robustness import robustness
        return robustness
    if name == "RobustnessReport":
        from sablier_flow.analytics.robustness import RobustnessReport
        return RobustnessReport
    if name == "deflated_sharpe":
        from sablier_flow.analytics.deflated_sharpe import deflated_sharpe
        return deflated_sharpe
    if name == "DeflatedSharpeReport":
        from sablier_flow.analytics.deflated_sharpe import DeflatedSharpeReport
        return DeflatedSharpeReport
    if name == "evaluate_family":
        from sablier_flow.analytics.family import evaluate_family
        return evaluate_family
    if name == "FamilyReport":
        from sablier_flow.analytics.family import FamilyReport
        return FamilyReport
    if name == "probability_of_backtest_overfitting":
        from sablier_flow.analytics.family import probability_of_backtest_overfitting
        return probability_of_backtest_overfitting
    if name == "consistency_check":
        from sablier_flow.analytics.consistency import consistency_check
        return consistency_check
    if name == "ConsistencyReport":
        from sablier_flow.analytics.consistency import ConsistencyReport
        return ConsistencyReport
    if name == "predictive_rank_score":
        from sablier_flow.analytics.predictive_rank import predictive_rank_score
        return predictive_rank_score
    if name == "PredictiveRankReport":
        from sablier_flow.analytics.predictive_rank import PredictiveRankReport
        return PredictiveRankReport

    # Bundled demo dataset
    if name == "demo_data":
        from sablier_flow.demo import demo_data
        return demo_data
    if name == "available_demo_datasets":
        from sablier_flow.demo import available_demo_datasets
        return available_demo_datasets

    # Attestation + crypto primitives
    if name in {"AttestationVerifier", "AttestationQuote", "AttestationVerificationError"}:
        from sablier_flow.client import attestation
        return getattr(attestation, name)
    if name == "AuthenticationError":
        from sablier_flow.client.transport import AuthenticationError
        return AuthenticationError
    if name in {
        "JobNotFoundError",
        "ModelNotFoundError",
        "RemoteJobError",
        "SablierClientError",
        "TransportError",
    }:
        from sablier_flow.client import transport as _t
        return getattr(_t, name)
    if name in {"envelope_encrypt", "envelope_decrypt", "EnvelopeEncrypted"}:
        from sablier_flow.client import crypto
        return getattr(crypto, name)

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
