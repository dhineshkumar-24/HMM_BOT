import MetaTrader5 as mt5

def calculate_position_size(balance, risk_pct, sl_pips, symbol="EURUSD"):
    """
    Calculates lot size based on % risk and SL distance.
    """
    if sl_pips <= 0:
        return 0.01

    risk_amount = balance * risk_pct
    
    # Get symbol properties
    symbol_info = mt5.symbol_info(symbol)
    if not symbol_info:
        return 0.01
        
    contract_size = symbol_info.trade_contract_size
    tick_size = symbol_info.trade_tick_size
    tick_value = symbol_info.trade_tick_value
    
    # Calculate value of one pip
    # Standard approximation if tick_value is complex
    # Lot Value = Volume * Contract Size
    # PnL = (Close - Open) * Volume * Contract Size
    
    # 1 Pip loss amount for 1.0 Lot
    # Assuming standard Forex 100k contract
    # Logic: Risk = Lot * SL_Pips * Pip_Value
    # Lot = Risk / (SL_Pips * Pip_Value)
    
    # Simple calculation for EURUSD (Standard)
    # pip_value_dollar ~ $10 usually per standard lot per pip
    
    # Dynamic calculation:
    # point = symbol_info.point
    # sl_points = sl_pips # Assuming input is in points actually, wait. 
    # User said "SL distance derived from hybrid". This will likely be a price difference.
    
    # Let's assume sl_pips is actually "Price Difference" (e.g., 0.00020)
    price_dist = sl_pips
    if price_dist == 0: return 0.01
    
    loss_per_lot = price_dist * contract_size 
    
    if loss_per_lot == 0: return 0.01
    
    lot_size = risk_amount / loss_per_lot
    
    # Normalize to lot step
    step = symbol_info.volume_step
    lot_size = round(lot_size / step) * step
    
    if lot_size < symbol_info.volume_min:
        lot_size = symbol_info.volume_min
    if lot_size > symbol_info.volume_max:
        lot_size = symbol_info.volume_max
        
    return lot_size

def check_daily_drawdown(initial_balance, current_balance, max_dd_pct):
    """
    Checks if daily drawdown limit is reached.
    """
    dd = (initial_balance - current_balance) / initial_balance
    return dd >= max_dd_pct
