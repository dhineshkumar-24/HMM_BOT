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
    Compute lot size so SL hit = exactly risk_pct of balance.
    Works in backtest (no MT5) and live (with MT5).
    """
    MIN_SL_DISTANCE = 0.00050   # 5 pip floor
    MAX_LOTS        = 2.00      # hard cap
    MIN_LOTS        = 0.01

    if sl_distance < MIN_SL_DISTANCE:
        sl_distance = MIN_SL_DISTANCE
    if sl_distance <= 0:
        return MIN_LOTS

    risk_amount = balance * risk_pct  # e.g. $100 at 1%

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
    # EURUSD standard: contract = 100,000 units
    # loss_per_lot = sl_distance * 100,000
    # e.g. 5 pip SL = 0.0005 * 100000 = $50 per lot
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
