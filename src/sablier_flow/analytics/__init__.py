"""Post-hoc analytics — robustness, DSR, PBO, drift checks.

These run entirely client-side against the customer's backtest results
(real + synthetic). No GPU, no server roundtrip. Published in the
public wheel because the methodology is academic
(Bailey–López de Prado for the deflated Sharpe ratio, Bailey et al.
for PBO via CSCV) and the trust signal comes from the customer being
able to read what's computed on their numbers.
"""
