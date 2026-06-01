"""1.1.0 data-types vocabulary collapse — every test in this file enforces
the contract that the SDK customer-facing surface accepts only
``{'price', 'level', 'return'}`` and rejects the pre-1.1 5-string vocab
with a friendly migration hint.

The wire-mapping shim (``_to_wire_data_types``) translates
``'level' → 'rate'`` before sending to the current Cloud Run backend; tests
here exercise both the customer-facing API and the wire shim.
"""
from __future__ import annotations

import pytest

import sablier_flow as sf
from sablier_flow.client.client import (
    _WIRE_DATA_TYPE_MAPPING,
    ALLOWED_DATA_TYPES,
    _require_data_types,
    _to_wire_data_types,
)


class TestAllowedDataTypes:
    def test_exactly_three_strings(self):
        assert frozenset({"price", "level", "return"}) == ALLOWED_DATA_TYPES

    def test_exposed_via_public_module(self):
        assert sf.ALLOWED_DATA_TYPES is ALLOWED_DATA_TYPES

    def test_allowed_frequencies_is_gone(self):
        with pytest.raises(AttributeError):
            sf.ALLOWED_FREQUENCIES  # noqa: B018  pre-1.1 export, removed

    @pytest.mark.parametrize("dtype", ["price", "level", "return"])
    def test_each_dtype_is_accepted_in_isolation(self, dtype):
        out = _require_data_types({"col": dtype}, expected_columns=["col"])
        assert out == {"col": dtype}


class TestLegacyAliasesRejected:
    @pytest.mark.parametrize(
        "legacy,new",
        [("rate", "level"), ("volatility", "level"), ("index", "price")],
    )
    def test_each_legacy_alias_raises_with_migration_hint(self, legacy, new):
        with pytest.raises(ValueError) as exc:
            _require_data_types(
                {"col": legacy}, expected_columns=["col"]
            )
        msg = str(exc.value)
        # The error MUST name the new vocabulary so an LLM agent or human
        # gets a single-glance fix.
        assert "retired" in msg.lower()
        assert f"{new!r}" in msg, f"migration hint missing for {legacy!r}: {msg}"

    def test_unknown_value_rejected(self):
        with pytest.raises(ValueError) as exc:
            _require_data_types(
                {"col": "definitely_not_a_type"}, expected_columns=["col"]
            )
        assert "definitely_not_a_type" in str(exc.value)

    def test_missing_column_rejected(self):
        with pytest.raises(ValueError) as exc:
            _require_data_types(
                {"SPY": "price"}, expected_columns=["SPY", "VIX"]
            )
        assert "VIX" in str(exc.value)

    def test_none_rejected_with_typeerror_pointing_at_dict(self):
        with pytest.raises(TypeError) as exc:
            _require_data_types(None, expected_columns=["SPY"])
        msg = str(exc.value)
        # Old-vocab examples in the error string would mislead; the
        # error example should use the new vocabulary only.
        assert "'price'" in msg
        assert "'level'" in msg
        assert "'volatility'" not in msg
        assert "'rate'" not in msg


class TestWireMapping:
    def test_mapping_table_is_complete(self):
        assert set(_WIRE_DATA_TYPE_MAPPING) == ALLOWED_DATA_TYPES

    def test_price_passes_through(self):
        assert _WIRE_DATA_TYPE_MAPPING["price"] == "price"

    def test_level_translates_to_rate(self):
        # back-compat with current Cloud Run backend; same DIFFERENCE
        # transform server-side. Will become identity when sablier-backend
        # goes live on AWS with native 'level' support.
        assert _WIRE_DATA_TYPE_MAPPING["level"] == "rate"

    def test_return_passes_through(self):
        assert _WIRE_DATA_TYPE_MAPPING["return"] == "return"

    def test_to_wire_translates_full_dict(self):
        wire = _to_wire_data_types({
            "SPY": "price",
            "VIX": "level",
            "FF_MOM": "return",
        })
        assert wire == {
            "SPY": "price",
            "VIX": "rate",       # ← translated
            "FF_MOM": "return",
        }

    def test_to_wire_empty_dict(self):
        assert _to_wire_data_types({}) == {}


class TestDemoDatasetUsesNewVocab:
    def test_demo_columns_use_new_vocab_only(self):
        df = sf.demo_data()
        for col, dtype in df.attrs["data_types"].items():
            assert dtype in ALLOWED_DATA_TYPES, (
                f"demo column {col!r} has legacy dtype {dtype!r}; "
                f"must be in {sorted(ALLOWED_DATA_TYPES)}"
            )

    def test_demo_vix_tnx_are_level(self):
        df = sf.demo_data()
        assert df.attrs["data_types"]["VIX"] == "level"
        assert df.attrs["data_types"]["TNX"] == "level"

    def test_demo_dxy_is_price(self):
        df = sf.demo_data()
        assert df.attrs["data_types"]["DXY"] == "price"

    def test_demo_tradeable_etfs_are_price(self):
        df = sf.demo_data()
        for col in ("SPY", "QQQ", "IWM", "TLT"):
            assert df.attrs["data_types"][col] == "price"


class TestClientSurfaceNoFrequencyKwarg:
    """The frequency= kwarg was removed in 1.1.0 — row cadence is auto-detected
    from the DataFrame index."""

    def test_client_fit_has_no_frequency_param(self):
        import inspect
        sig = inspect.signature(sf.Client.fit)
        assert "frequency" not in sig.parameters

    def test_client_generate_has_no_frequency_param(self):
        import inspect
        sig = inspect.signature(sf.Client.generate)
        assert "frequency" not in sig.parameters

    def test_client_validate_has_no_frequency_param(self):
        import inspect
        sig = inspect.signature(sf.Client.validate)
        assert "frequency" not in sig.parameters

    def test_module_level_fit_shortcut_has_no_frequency(self):
        import inspect
        sig = inspect.signature(sf.fit)
        assert "frequency" not in sig.parameters
