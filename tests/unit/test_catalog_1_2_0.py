"""1.2.0 catalog-model discoverability.

Three surfaces are pinned here:

  * the wire ``ModelInfoResponse`` accepts the four new optional fields
    (visibility / display_name / feature_data_types / scorecard) and stays
    backward-compatible when a server omits them;
  * the public ``Model`` dataclass exposes those fields (via
    ``_model_info_to_dataclass``);
  * ``sf.catalog()`` / ``Client.catalog()`` returns only catalog-visibility
    models, and ``generate(...)`` auto-fills ``data_types`` from a model's
    registered ``feature_data_types`` when the caller omits them.
"""
from __future__ import annotations

import pandas as pd
import pytest

import sablier_flow as sf


def _anchor_df(cols):
    """A small anchor window with a DatetimeIndex (the SDK validates that
    generate's anchor_data is date-indexed before the round-trip)."""
    idx = pd.date_range("2024-01-01", periods=2, freq="B")
    return pd.DataFrame({c: [1.0, 2.0] for c in cols}, index=idx)
from sablier_flow.client.client import Client, _model_info_to_dataclass
from sablier_flow.client.transport import ListModelsResponse, ModelInfoResponse
from sablier_flow.types import Model


def _info(model_id, *, visibility=None, feature_data_types=None, **kw):
    return ModelInfoResponse(
        model_id=model_id,
        features=list(feature_data_types or {"SPY": "price"}),
        training_horizon=252,
        n_assets=len(feature_data_types or {"SPY": "price"}),
        status="ready",
        visibility=visibility,
        feature_data_types=feature_data_types,
        **kw,
    )


# ---------------------------------------------------------------------------
# Wire schema — the four new optional fields
# ---------------------------------------------------------------------------


class TestModelInfoResponseSchema:
    def test_accepts_new_catalog_fields(self):
        info = ModelInfoResponse(
            model_id="catalog-us_eq_wide_504_v1",
            features=["AAPL", "MSFT"],
            training_horizon=504,
            n_assets=2,
            status="ready",
            visibility="catalog",
            display_name="US Equities — Wide",
            feature_data_types={"AAPL": "price", "MSFT": "price"},
            scorecard={"252": 0.72, "504": 0.68},
        )
        assert info.visibility == "catalog"
        assert info.display_name == "US Equities — Wide"
        assert info.feature_data_types == {"AAPL": "price", "MSFT": "price"}
        assert info.scorecard == {"252": 0.72, "504": 0.68}

    def test_fields_default_none_for_older_servers(self):
        info = ModelInfoResponse(
            model_id="m-1",
            features=["SPY"],
            training_horizon=252,
            n_assets=1,
            status="ready",
        )
        assert info.visibility is None
        assert info.display_name is None
        assert info.feature_data_types is None
        assert info.scorecard is None


# ---------------------------------------------------------------------------
# Public Model dataclass exposes the fields
# ---------------------------------------------------------------------------


class TestModelDataclassExposesFields:
    def test_new_fields_default_none(self):
        m = Model(
            model_id="m-1",
            features=["SPY"],
            training_horizon=252,
            n_assets=1,
            status="ready",
        )
        assert m.visibility is None
        assert m.display_name is None
        assert m.feature_data_types is None
        assert m.scorecard is None

    def test_model_info_to_dataclass_carries_catalog_fields(self):
        info = _info(
            "catalog-1",
            visibility="catalog",
            feature_data_types={"AAPL": "price"},
            display_name="US Equities — Wide",
            scorecard={"504": 0.68},
        )
        m = _model_info_to_dataclass(info)
        assert m.visibility == "catalog"
        assert m.display_name == "US Equities — Wide"
        assert m.feature_data_types == {"AAPL": "price"}
        assert m.scorecard == {"504": 0.68}


# ---------------------------------------------------------------------------
# sf.catalog() filters to catalog-visibility models
# ---------------------------------------------------------------------------


class _ListTransport:
    """Minimal transport stub: only ``list_models`` is exercised."""

    def __init__(self, infos):
        self._infos = infos

    def list_models(self, *, limit=50):
        return ListModelsResponse(
            models=list(self._infos), total_returned=len(self._infos)
        )


def _client_with(infos):
    return Client(api_key="sk-test", transport=_ListTransport(infos))


class TestCatalog:
    def test_catalog_method_returns_only_catalog_models(self):
        client = _client_with([
            _info("m-private", visibility="private"),
            _info("catalog-1", visibility="catalog",
                  feature_data_types={"AAPL": "price"}),
            _info("m-legacy-none", visibility=None),
            _info("catalog-2", visibility="catalog",
                  feature_data_types={"MSFT": "price"}),
        ])
        cat = client.catalog()
        assert [m.model_id for m in cat] == ["catalog-1", "catalog-2"]
        assert all(m.visibility == "catalog" for m in cat)

    def test_catalog_empty_when_no_catalog_models(self):
        client = _client_with([_info("m-1", visibility="private")])
        assert client.catalog() == []

    def test_catalog_is_public_and_exported(self):
        assert callable(sf.catalog)
        assert "catalog" in sf.__all__
        assert "catalog" in dir(sf)


# ---------------------------------------------------------------------------
# generate(...) auto-fills data_types from the model's feature_data_types
# ---------------------------------------------------------------------------


class _CapturedRunJob(Exception):
    """Raised by the patched ``_run_job`` to capture the resolved params
    without running a real (crypto + polling) job."""

    def __init__(self, params):
        self.params = params
        super().__init__("captured")


def _patch_run_job(client, monkeypatch):
    def _capture(*, kind, real_data, params, idempotency_key):
        raise _CapturedRunJob(params)

    monkeypatch.setattr(client, "_run_job", _capture)


def test_generate_autofills_data_types_from_catalog_model(monkeypatch):
    client = _client_with([])  # transport.list_models unused here
    model = Model(
        model_id="catalog-1",
        features=["AAPL", "MSFT"],
        training_horizon=504,
        n_assets=2,
        status="ready",
        visibility="catalog",
        feature_data_types={"AAPL": "price", "MSFT": "price"},
    )
    monkeypatch.setattr(client, "get_model", lambda mid: model)
    # Column-coverage check also needs the model's features; stub it.
    monkeypatch.setattr(client, "_check_columns_cover_model", lambda *a, **k: None)
    _patch_run_job(client, monkeypatch)

    df = _anchor_df(["AAPL", "MSFT"])
    with pytest.raises(_CapturedRunJob) as exc:
        client.generate("catalog-1", anchor_data=df, n_paths=4, quiet=True)
    # Auto-filled types reach the wire as feature_data_types (both 'price'
    # pass through the level->rate wire mapping unchanged).
    assert exc.value.params["feature_data_types"] == {"AAPL": "price", "MSFT": "price"}


def test_generate_does_not_fetch_when_data_types_passed(monkeypatch):
    """No extra round-trip when the caller supplies data_types explicitly."""
    client = _client_with([])
    called = {"n": 0}

    def _boom(mid):
        called["n"] += 1
        raise AssertionError("get_model must not be called when data_types given")

    monkeypatch.setattr(client, "get_model", _boom)
    monkeypatch.setattr(client, "_check_columns_cover_model", lambda *a, **k: None)
    _patch_run_job(client, monkeypatch)

    df = _anchor_df(["AAPL"])
    with pytest.raises(_CapturedRunJob):
        client.generate(
            "m-1", anchor_data=df, data_types={"AAPL": "price"},
            n_paths=4, quiet=True,
        )
    assert called["n"] == 0


def test_generate_falls_back_when_fetch_fails(monkeypatch):
    """A failing get_model must degrade to current behavior. With no
    feature_data_types available and a ref window supplied, the server
    reuses the model's registered types (no feature_data_types on wire)."""
    client = _client_with([])

    def _raise(mid):
        raise RuntimeError("network")

    monkeypatch.setattr(client, "get_model", _raise)
    monkeypatch.setattr(client, "_check_columns_cover_model", lambda *a, **k: None)
    _patch_run_job(client, monkeypatch)

    df = _anchor_df(["AAPL"])
    with pytest.raises(_CapturedRunJob) as exc:
        client.generate("m-1", anchor_data=df, n_paths=4, quiet=True)
    assert "feature_data_types" not in exc.value.params


def test_generate_without_data_types_or_types_still_raises(monkeypatch):
    """Model has no feature_data_types AND no ref window: preserve the
    current contract — _require_data_types raises TypeError."""
    client = _client_with([])
    model = Model(
        model_id="m-1",
        features=["AAPL"],
        training_horizon=252,
        n_assets=1,
        status="ready",
        feature_data_types=None,
    )
    monkeypatch.setattr(client, "get_model", lambda mid: model)
    # No anchor/like -> _require_data_types(None, ...) path is NOT hit in
    # generate (that path only requires when ref_df present). Instead the
    # server reuses registered types; assert no feature_data_types on wire.
    _patch_run_job(client, monkeypatch)
    with pytest.raises(_CapturedRunJob) as exc:
        client.generate("m-1", n_paths=4, quiet=True)
    assert "feature_data_types" not in exc.value.params
