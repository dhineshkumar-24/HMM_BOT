import numpy as np
import pandas as pd
from utils.indicators import *
from utils.helpers import check_trading_session, check_news_impact
from utils.logger import setup_logger

logger = setup_logger("Strategy")

class MeanReversionStrategy:
    def __init__(self, config):
        self.config = config
        self.z_trigger = config['strategy']['z_score_trigger']       
        self.entry_target = config['strategy']['entry_zone_target']  
        self.hurst_limit = config['strategy']['hurst_threshold']     
        
        # RSI Limits
        self.rsi_period = config['strategy']['rsi_period']
        self.rsi_ob = config['strategy']['rsi_overbought']
        self.rsi_os = config['strategy']['rsi_oversold']
        
        # State Tracking
        self.pending_signal = None  # None, "BUY", "SELL"
        self.wait_counter = 0

    def calculate_indicators(self, df):
        """
        Calculates all required indicators and adds them to the DataFrame.
        """
        df = df.copy()
        df['returns'] = np.log(df['close'] / df['close'].shift(1))
        df['volatility'] = compute_volatility(df['returns'])
        # Use configured window (e.g. 20) for Rolling VWAP matching Z-Score window
        vwap_win = self.config['strategy'].get('vwap_window', 20)
        df['vwap'] = compute_vwap(df, window=vwap_win)
        
        # Z-Score: (Close - VWAP) / Rolling STD
        # Note: If VWAP is length 20, and STD is length 20, Z-Score is consistent.
        df['z_score'] = (df['close'] - df['vwap']) / (df['close'].rolling(20).std()) 
        df['vol_slope'] = compute_volatility_slope(df['volatility'])
        df['auto_corr'] = compute_autocorrelation(df['returns'])
        df['hurst'] = compute_hurst(df['close']) 
        df['rsi'] = compute_rsi(df['close'], period=self.rsi_period) # New
        return df

    def analyze_market(self, df):
        """
        Executes the optimization logic:
        1. Check Filters (Session, News, Slope, Hurst, AC)
        2. Check Z-Score Trigger (> Z) -> Set State
        3. Check Retracement (< Target) -> Execute Entry
        """
        if len(df) < 100: return None
        if not check_trading_session(self.config): return None
        if not check_news_impact(): return None

        last = df.iloc[-2] # Closed Candle

        last = df.iloc[-2] # Closed Candle

        # --- Regime Filters ---
        if last.vol_slope > 0.0001: return None
        if last.hurst > self.hurst_limit: return None

        signal = None
        z = last.z_score
        rsi = last.rsi
        
        # STATE MACHINE LOGIC
        if self.pending_signal:
            self.wait_counter += 1
            if self.wait_counter > 20: 
                self.pending_signal = None
                self.wait_counter = 0

        # TRANSITION 1: Trigger (Z-Score + RSI CHECK)
        if abs(z) >= self.z_trigger:
            if z > 0 and rsi > self.rsi_ob: # Sell Overbought
                self.pending_signal = "SELL_WAIT"
                self.wait_counter = 0
            elif z < 0 and rsi < self.rsi_os: # Buy Oversold
                self.pending_signal = "BUY_WAIT"
                self.wait_counter = 0
            
        # TRANSITION 2: Execution (Z drops to Target)
        if self.pending_signal == "SELL_WAIT":
            if z <= self.entry_target: 
                signal = "SELL"
                self.pending_signal = None
                
        elif self.pending_signal == "BUY_WAIT":
            if z >= -self.entry_target:
                signal = "BUY"
                self.pending_signal = None

        if signal:
            return {
                "bias": signal,
                "z_score": z,
                "rsi": rsi,
                "vwap": last.vwap,
                "volatility": last.volatility,
                "close": last.close
            }
            
        return None
