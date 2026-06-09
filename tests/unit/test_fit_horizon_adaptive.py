"""Data-adaptive fit-horizon cap.

The flat 504 cap was replaced with a cap that scales with the customer's
history length: a longer training window is only allowed when there are
enough *non-overlapping* spans of that length to train it without
memorizing. Two invariants matter most:

  1. Non-breaking — any horizon that passed the old flat-504 check still
     passes (the cap floors at 504 for every input).
  2. Data-rich customers can now train beyond 504, bounded by their data
     and never above the server backstop (5000).
"""
from __future__ import annotations

import pytest

from sablier_flow.client.client import (
    _FIT_HORIZON_CEILING,
    _FIT_HORIZON_FLOOR,
    _check_fit_horizon_bounds,
    _fit_horizon_max,
)


# --------------------------------------------------------------------------
# _fit_horizon_max — the pure adaptive-cap function
# --------------------------------------------------------------------------


def test_unknown_rows_falls_back_to_flat_floor():
    assert _fit_horizon_max(None, 0.8) == _FIT_HORIZON_FLOOR
    assert _fit_horizon_max(0, 0.8) == _FIT_HORIZON_FLOOR


def test_thin_data_stays_at_floor():
    # ~2y daily (504 rows): data_max is tiny, so the cap floors at 504.
    assert _fit_horizon_max(504, 0.8) == _FIT_HORIZON_FLOOR


def test_data_rich_raises_cap_above_floor():
    # 20y daily (5040 rows), train_split 0.8:
    #   train_rows = 4032, usable = 4032 - 200 = 3832, data_max = 3832 // 2 = 1916
    cap = _fit_horizon_max(5040, 0.8)
    assert cap == 1916
    assert _FIT_HORIZON_FLOOR < cap < _FIT_HORIZON_CEILING


def test_clamped_to_ceiling():
    # Absurdly long history must still clamp to the server backstop.
    assert _fit_horizon_max(20_000, 1.0) == _FIT_HORIZON_CEILING


def test_train_split_none_treated_as_full_history():
    # train_split=None means "train on everything" → use all rows.
    assert _fit_horizon_max(3000, None) == _fit_horizon_max(3000, 1.0)


def test_cap_never_below_floor_for_any_input():
    for n in (10, 250, 504, 1000, 2520, 5040, 50_000):
        for ts in (None, 0.5, 0.8, 1.0):
            assert _fit_horizon_max(n, ts) >= _FIT_HORIZON_FLOOR


# --------------------------------------------------------------------------
# _check_fit_horizon_bounds — the guard used by fit() / fit_async()
# --------------------------------------------------------------------------


def test_legacy_flat_behavior_without_rows():
    # No n_rows → flat floor. <=504 ok, >504 rejected (old contract).
    _check_fit_horizon_bounds(504)
    _check_fit_horizon_bounds(1)
    with pytest.raises(ValueError):
        _check_fit_horizon_bounds(505)


def test_horizon_at_or_below_floor_always_accepted():
    # Even on thin data, anything the old cap allowed still passes.
    for n in (250, 504, 1000):
        _check_fit_horizon_bounds(504, n_rows=n, train_split=0.8)
        _check_fit_horizon_bounds(120, n_rows=n, train_split=0.8)


def test_long_horizon_rejected_on_thin_data():
    with pytest.raises(ValueError, match="fit-horizon cap"):
        _check_fit_horizon_bounds(600, n_rows=504, train_split=0.8)


def test_long_horizon_accepted_on_rich_data():
    # 10y daily (2520 rows): train_rows=2016, usable=1816, data_max=908.
    _check_fit_horizon_bounds(900, n_rows=2520, train_split=0.8)
    with pytest.raises(ValueError):
        _check_fit_horizon_bounds(1000, n_rows=2520, train_split=0.8)


def test_nonpositive_and_noninteger_rejected():
    with pytest.raises(ValueError):
        _check_fit_horizon_bounds(0, n_rows=5040, train_split=0.8)
    with pytest.raises(ValueError):
        _check_fit_horizon_bounds(-5, n_rows=5040, train_split=0.8)
    with pytest.raises(ValueError):
        _check_fit_horizon_bounds("abc")
