"""
core/risk.py — Position sizing and drawdown protection.

Pure calculation functions. No MT5 order logic here.
"""

import MetaTrader5 as mt5


def calculate_position_size(
    balance: float,
    risk_pct: float,
    sl_distance: float,
    symbol: str = "EURUSD",
) -> float:
    """
    Compute lot size so that a stop-loss hit loses exactly `risk_pct` of balance.

    Args:
        balance:     Account balance in account currency.
        risk_pct:    Fraction of balance to risk (e.g. 0.01 = 1%).
        sl_distance: Price distance to stop-loss (e.g. 0.0020 for 20 pips on EURUSD).
        symbol:      Instrument ticker (used to fetch contract specs from MT5).

    Returns:
        Normalised lot size clamped to broker min/max/step.
    """
    if sl_distance <= 0:
        return 0.01

    symbol_info = mt5.symbol_info(symbol)
    if not symbol_info:
        return 0.01

    contract_size = symbol_info.trade_contract_size  # Usually 100,000 for FX
    risk_amount   = balance * risk_pct

    # Loss per 1.0 lot for given SL distance = distance × contract_size
    loss_per_lot = sl_distance * contract_size
    if loss_per_lot == 0:
        return 0.01

    lot_size = risk_amount / loss_per_lot

    # Normalise to broker step
    step     = symbol_info.volume_step
    lot_size = round(lot_size / step) * step

    # Clamp to min/max
    lot_size = max(lot_size, symbol_info.volume_min)
    lot_size = min(lot_size, symbol_info.volume_max)

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
