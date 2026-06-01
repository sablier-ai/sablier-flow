"""1.1.0 intraday gate lift — the SDK accepts any uniform-cadence
DatetimeIndex (5-min, 1-min, hourly, daily, weekly, monthly, quarterly)
and surfaces the detected cadence in the pre-flight info line.

The wire payload always sends ``'daily'`` for intraday cadences so the
current Cloud Run backend (which only knows the 4 canonical families)
treats the data as a uniform sequence without triggering the legacy
``FREQUENCY_DATA_TYPE_TRANSFORMS`` override path.
"""
from __future__ import annotations

import pandas as pd
import pytest

from sablier_flow.client.client import (
    _detect_row_cadence,
    _resolve_wire_frequency,
)


class TestRowCadenceDetection:
    @pytest.mark.parametrize(
        "freq,expected_label_contains",
        [
            ("1min",   "1-min"),
            ("5min",   "5-min"),
            ("15min",  "15-min"),
            ("1h",     "1.0h"),
            ("D",      "daily"),
            ("W",      "weekly"),
            ("MS",     "monthly"),
            ("QS",     "quarterly"),
        ],
    )
    def test_uniform_cadence_detected(self, freq, expected_label_contains):
        # Use enough rows that the median Δt is unambiguous.
        idx = pd.date_range("2024-01-01", periods=200, freq=freq)
        label, median = _detect_row_cadence(idx)
        assert expected_label_contains in label, (
            f"cadence label for freq={freq!r} is {label!r}; "
            f"expected substring {expected_label_contains!r}"
        )
        assert isinstance(median, pd.Timedelta)
        assert median > pd.Timedelta(0)

    def test_short_index_rejected(self):
        idx = pd.DatetimeIndex(["2024-01-01"])
        with pytest.raises(ValueError) as exc:
            _detect_row_cadence(idx)
        assert "at least 2 rows" in str(exc.value)

    def test_non_datetimeindex_rejected(self):
        with pytest.raises(ValueError) as exc:
            _detect_row_cadence(pd.Index([0, 1, 2, 3]))
        assert "DatetimeIndex" in str(exc.value)

    def test_irregular_index_rejected(self):
        # Build an index whose 95th-percentile gap is >> 3× the median.
        timestamps = [pd.Timestamp("2024-01-01") + pd.Timedelta(days=i) for i in range(20)]
        timestamps.append(pd.Timestamp("2025-01-01"))  # massive gap at the end
        idx = pd.DatetimeIndex(timestamps)
        with pytest.raises(ValueError) as exc:
            _detect_row_cadence(idx)
        msg = str(exc.value)
        assert "irregular" in msg.lower()
        assert "uniform" in msg.lower()


class TestWireFrequencyMapping:
    """The wire-frequency value sent to the current Cloud Run backend is always
    one of {'daily', 'weekly', 'monthly', 'quarterly'} so the legacy
    FREQUENCY_DATA_TYPE_TRANSFORMS overrides never accidentally fire on the
    customer's at-cadence data."""

    @pytest.mark.parametrize("intraday_label", [
        "intraday (5-min)",
        "intraday (1-min)",
        "intraday (15-min)",
        "intraday (1.0h)",
        "intraday (30s)",
    ])
    def test_intraday_collapses_to_daily_on_wire(self, intraday_label):
        assert _resolve_wire_frequency(intraday_label) == "daily"

    @pytest.mark.parametrize("native_label", ["daily", "weekly", "monthly", "quarterly"])
    def test_native_labels_pass_through(self, native_label):
        assert _resolve_wire_frequency(native_label) == native_label


class TestNoIntradayRejectionMessage:
    """The pre-1.1 ``'intraday is deferred to 1.1.0'`` error message must
    not surface from any of the cadence helpers; the gate is fully lifted."""

    def test_5min_index_does_not_raise(self):
        idx = pd.date_range("2024-01-01 09:30", periods=200, freq="5min")
        # Should NOT raise — pre-1.1 this raised with 'intraday is deferred'.
        label, _ = _detect_row_cadence(idx)
        assert "intraday" in label

    def test_1min_index_does_not_raise(self):
        idx = pd.date_range("2024-01-01 09:30", periods=500, freq="1min")
        label, _ = _detect_row_cadence(idx)
        assert "intraday" in label
