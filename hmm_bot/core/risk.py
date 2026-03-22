"""
core/risk.py — Position sizing and drawdown protection.

v4.0 Enhancements:
    - Half-Kelly position sizing with volatility scaling
    - ATR-based sizing (unified, replaces mixed vol/ATR approach)
    - Regime-adaptive risk budget
    - Cost-aware minimum lot computation

Pure calculation functions. No MT5 order logic here.
"""

from __future__ import annotations

import math
import numpy as np


def calculate_position_size(
    balance: float,
    risk_pct: float,
    sl_distance: float,
    symbol: str = "EURUSD",
    atr: float = 0.0,
    win_rate: float = 0.48,
    avg_rr: float = 2.0,
) -> float:
    """
    Half-Kelly position sizing with ATR-based volatility adjustment.

    The standard fixed-fractional approach sizes purely on SL distance.
    This enhanced version applies a Kelly-fraction adjustment:

        f* = (p * b - q) / b     where p=win_rate, q=1-p, b=avg RR
        position_risk = min(risk_pct, f*/2)   (half-Kelly for safety)

    Then volatility scaling:
        If ATR is provided, scale down when vol is high relative to normal.
        target_vol = 0.0008 (EURUSD M5 typical 14-bar ATR)
        vol_scalar = min(1.0, target_vol / current_atr)

    Args:
        balance:     Account balance in USD.
        risk_pct:    Base risk fraction from config (e.g. 0.01 = 1%).
        sl_distance: Stop-loss distance in price units.
        symbol:      Trading instrument.
        atr:         Current ATR value (for volatility scaling). 0 = no scaling.
        win_rate:    Estimated strategy win rate (default 0.48).
        avg_rr:      Estimated average risk-reward ratio (default 2.0).

    Returns:
        Lot size (minimum 0.01, maximum 2.00).
    """
    MIN_SL_DISTANCE = 0.00050   # 5-pip floor
    MAX_LOTS        = 2.00      # hard cap
    MIN_LOTS        = 0.01

    if sl_distance < MIN_SL_DISTANCE:
        sl_distance = MIN_SL_DISTANCE
    if sl_distance <= 0:
        return MIN_LOTS

    # ── Half-Kelly fraction ───────────────────────────────────────────────────
    # Kelly criterion: f* = (p * b - q) / b
    # Half-Kelly for safety margin against estimation error
    q = 1.0 - win_rate
    if avg_rr > 0:
        kelly_full = (win_rate * avg_rr - q) / avg_rr
    else:
        kelly_full = 0.0

    kelly_half = max(kelly_full / 2.0, 0.001)  # minimum tiny risk
    effective_risk = min(risk_pct, kelly_half)

    # ── Volatility scaling ────────────────────────────────────────────────────
    # Scale position down when ATR is elevated above typical levels.
    # This implements natural volatility targeting.
    vol_scalar = 1.0
    if atr > 0:
        target_atr = 0.00080  # EURUSD M5 typical 14-bar ATR
        vol_scalar = min(1.0, target_atr / max(atr, 1e-8))
        vol_scalar = max(vol_scalar, 0.25)  # don't reduce more than 75%

    risk_amount = balance * effective_risk * vol_scalar

    # ── Try MT5 first (live trading) ─────────────────────────────────────────
    try:
        import MetaTrader5 as mt5
        symbol_info = mt5.symbol_info(symbol)
        if symbol_info:
            contract_size = symbol_info.trade_contract_size
            loss_per_lot  = sl_distance * contract_size
            if loss_per_lot > 0:
                lot_size = risk_amount / loss_per_lot
                step     = symbol_info.volume_step
                lot_size = round(lot_size / step) * step
                lot_size = max(lot_size, symbol_info.volume_min)
                lot_size = min(lot_size, symbol_info.volume_max)
                lot_size = min(lot_size, MAX_LOTS)
                return lot_size
    except Exception:
        pass

    # ── Fallback for backtest (no MT5 connection) ─────────────────────────────
    CONTRACT_SIZE = 100_000.0
    loss_per_lot  = sl_distance * CONTRACT_SIZE

    if loss_per_lot <= 0:
        return MIN_LOTS

    lot_size = risk_amount / loss_per_lot

    # Round to 0.01 step
    lot_size = round(lot_size / 0.01) * 0.01
    lot_size = max(lot_size, MIN_LOTS)
    lot_size = min(lot_size, MAX_LOTS)

    return lot_size


def check_daily_drawdown(
    initial_balance: float,
    current_equity: float,
    max_dd_pct: float,
) -> bool:
    """
    Return True if the daily drawdown limit has been breached.

    Args:
        initial_balance: Balance at start of the trading day.
        current_equity:  Current account equity (unrealised P&L included).
        max_dd_pct:      Maximum allowed daily drawdown fraction (e.g. 0.02 = 2%).
    """
    if initial_balance <= 0:
        return False
    dd = (initial_balance - current_equity) / initial_balance
    return dd >= max_dd_pct


def compute_half_kelly(
    win_rate: float,
    avg_win: float,
    avg_loss: float,
) -> float:
    """
    Compute the half-Kelly fraction for position sizing.

    f* = (p * b - q) / b where b = avg_win / avg_loss (risk-reward ratio)
    Returns f*/2 (half-Kelly for safety).

    Args:
        win_rate: Historical win rate (0-1).
        avg_win:  Average winning trade PnL (positive).
        avg_loss: Average losing trade PnL (positive, absolute value).

    Returns:
        Half-Kelly fraction (0-0.25 range). Returns 0 if edge is negative.
    """
    if avg_loss <= 0 or win_rate <= 0:
        return 0.0

    b = avg_win / avg_loss  # risk-reward ratio
    q = 1.0 - win_rate

    kelly_full = (win_rate * b - q) / b
    if kelly_full <= 0:
        return 0.0

    return min(kelly_full / 2.0, 0.25)  # capped at 25% half-Kelly
