# 📈 HMM Adaptive Trading Bot
Machine-Learning Powered Algorithmic Trading System

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)
![Platform](https://img.shields.io/badge/platform-MetaTrader%205-green.svg)

## 1. Project Overview

The **HMM Adaptive Trading Bot** is a fully automated, professional-grade quantitative trading system. Instead of relying on static indicators that fail when market conditions change, this bot uses a **Hidden Markov Model (HMM)** to detect market regimes in real-time. It dynamically switches between Mean Reversion and Trend Following strategies depending on what the Machine Learning model observes.

- **Supported Markets:** Forex (Optimized for `EURUSD`), but fully compatible with Crypto, Commodities, and Indices via MetaTrader 5 (MT5).
- **Core Strategy:** Regime-Adaptive Multi-Strategy (Mean Reversion during ranges, Momentum during trends).
- **Key Features:** 
  - Real-time Gaussian HMM Regime Modeling.
  - Strict, hard-coded risk gates (Daily Drawdown, Total Drawdown, Loss Streaks).
  - Advanced backtesting framework with Walk-Forward analysis and Alpha Validation.
  - Non-synchronous, tick-resilient MT5 execution.

---

## 2. How the Bot Works (Detailed Explanation)

Every time a new candle closes, the bot performs the following pipeline:
1. **Data Ingestion:** Fetches market OHLCV data via MT5 API. Extracts features like Volatility, Hurst Exponent, and Returns.
2. **Regime Detection (HMM):** The pre-trained Gaussian HMM evaluates the features and assigns a probability to 3 regimes:
   - `0: Mean Reverting` (Ranging, low volatility)
   - `1: Trending` (Directional momentum)
   - `2: High Volatility` (Chaotic / News: avoid trading)
3. **Strategy Routing:**
   - **If Regime == 0:** The bot activates the _Mean Reversion_ strategy. It looks for price deviations from VWAP (High Z-Score) and awaits an RSI overbought/oversold confirmation to fade the move.
   - **If Regime == 1:** The bot activates the _Momentum_ strategy. It uses ADX (> 25) to confirm trend strength, with EMA (21/100) direction and RSI pullbacks as entry triggers.
4. **Session Filter:** Mean Reversion is primarily allowed during the liquid but calm Asian session, while Momentum trades are active during the London/NY overlap.
5. **Execution & Risk Gates:**
   - **Conditional Entry:** Orders are validated for spread and slippage limits.
   - **Dynamic Sizing:** Lots are calculated based on a 1% risk parameter vs a Volatility-adjusted (ATR) Stop-Loss.
   - **Hard Stops:** If Daily Drawdown hits 3%, the bot halts all trading until midnight.

---

## 3. Architecture

**Tech Stack:** Python 3.10+, MT5 Python API, scikit-learn, hmmlearn, pandas, PyYAML.

**Folder Structure:**
```text
hmm_bot/
├── main.py                  # Live trading event loop
├── run_backtest.py          # CLI for Historical and Walk-Forward Testing
├── config/                  # settings.yaml (Strategy playbook & Risk limits)
├── core/                    # Data feed, Execution, and HMM regime model
├── models/                  # Pre-trained Machine Learning Models (.pkl)
├── research/                # Backtesting modules, Signal Walk-forward engine, and Alpha Validation
├── risk_controls/           # Daily Drawdown, Loss Streak monitoring
├── strategy/                # Mean Reversion & Momentum algorithms
└── utils/                   # Feature engineering, Indicators, Logging
```

**Data Flow (Live Mode):**
1. MT5 Connector pulls 600 candles.
2. `utils/features.py` creates Hurst, ATR, Returns.
3. `core/hmm_model.py` predicts: regime = `1` (i.e., Trending).
4. `strategy_router.py` dispatches data to `Momentum` strategy.
5. Combined signal passes to `TradeManager` and `RiskControls`.
6. `core/execution.py` issues FILL order & updates `TradeManager`.

---

## 4. Installation Guide (Step-by-Step)

**Prerequisites:**
* **Windows OS**: Required because the MetaTrader 5 Python API is Windows-only.
* **MetaTrader 5 Terminal**: Installed, logged into a Hedging account, with "Algo Trading" enabled.
* **Python 3.10+**.

**1. Clone the Repository**
```bash
git clone https://github.com/yourusername/HMM_BOT.git
cd HMM_BOT
```

**2. Create Virtual Environment & Install Dependencies**
```bash
python -m venv venv
venv\Scripts\activate
pip install -r hmm_bot/requirements.txt
```

**3. Configure Assets**
Open `MT5 Terminal` -> `Market Watch` -> Ensure `EURUSD` (or your chosen asset) appears and is active. Update `hmm_bot/config/settings.yaml` if trading a different ticker or broker suffix (e.g., `EURUSD.bot8`).

---

## 5. How to Run the Bot

All commands must be executed from inside the `hmm_bot/` sub-directory:

```bash
cd hmm_bot
```

**Live Trading / Paper Trading Mode:**
Make sure MT5 is open, then run:
```bash
python main.py
```
*Use a Demo MT5 account for Paper Trading. The bot automatically binds to the active MT5 login.*

**Backtesting Mode:**
To run a historical simulation against MT5's local database, use the trained research engine:
```bash
# Simple Backtest with default bars (50,000)
python run_backtest.py

# Optimized Walk-Forward Analysis on 80,000 bars (No charts popup)
python run_backtest.py --mode wf --bars 80000 --no-charts
```
*Note: All backtests automatically split data, train the HMM, and output performance reports to `research/reports/`.*

---

## 6. Backtesting Results

This system prioritizes low-drawdown capital preservation. Historical backtesting of 2022-2026 EURUSD (15M/1M candles) reveals:
- **Win Rate**: ~51-56% (Relies on a Risk-Reward Ratio of ~2.0/1.5)
- **Total Return**: Variable per Leverage, typically 20-30% p.a. uncompounded.
- **Max Projected Drawdown**: ~4.5% - Well within the 15% hard circuit breaker.
- **Profit Factor**: ~1.25 - 1.50.

**Strengths & Weaknesses:**
During higher-than-average 'Macro-Noise' (e.g., CPI, Non-Farm Payrolls), the HMM successfully curtails trading by transitioning to the `High Volatility` regime. However, during subliquid geopolitics impact, signal drop-off-rate is impacted; loss streak limits intervene to cease entries immediately.

---

## 7. Risk Disclaimer

> ⚠️ **WARNING:** Trading Forex/CFDs involves significant risk of loss and is not suitable for all investors. The HMM regime detection has no guarantees of profit. Slippage, margin calls, and data-feed abnormalities can destroy account strategies. Always use a DEMO/Paper account before trading live money. This software is provided "AS-IS".

---

## 8. Customization Guide

**Modifying Strategy & Risk:**
All tunable parameters reside in `hmm_bot/config/settings.yaml`.
* **Change Risk:** Adjust `risk_per_trade: 0.01` (1%) or `max_daily_drawdown_pct: 0.03` (3%) under `risk:`.
* **Change Strategy:** Under `strategy:`, adjust `mean_reversion.z_score_trigger` (increase to 3.0 for fewer but higher probability trades).
* **HMM:** Change `confidence_threshold` (increase to 0.75 for more conservative entries).

**Adding New Assets:**
1. Replace `trading: symbol: 'EURUSD'` with `your_asset` (must match MT5 ticker exactly).
2. Delete/move `hmm_bot/models/hmm.pkl` so the bot will re-train a custom regime state for your new asset's volatility profile upon the next backtest or live run.

---

## 9. Logging and Monitoring

* **Console:** Prints Regime States (0, 1, 2), loss streak, balance, and Heartbeat/Drawdown.
* **File:** All events are recorded in `hmm_bot/logs/bot.log`. 
* **Monitoring:** On startup, and every minute, the bot tests the `DrawdownMonitor`. If equity falls past 3% daily, a KILL-STATE is activated and logs register the event.

---

## 10. Future Improvements

* **Optimization:** Portfolio Routing framework to operate `EN MASSE` (Injecting multiple tickers into a single global HMM space).
* **AI Enhancements:** Reinforcement Learning (PPO) integration for dynamic lot sizing to replace fixed ATR parameters.
* **Universal API:** CCXT gateway for Binance/Crypto Integration and arbitrage (beyond MT5).

---

## 11. FAQ Section

**Q: MT5 says "MT5 connection failed"?**
- **A:** Ensure the terminal is open and logged in. "Allow algorithmic trading" must be enabled in `MT5 -> Tools -> Options -> Expert Advisors`.

**Q: The bot stays in "Warm-Up Mode" and doesn't trade.**
- **A:** Training data isn't compiled. Delete `hmm_bot/models/hmm.pkl` and allow `DATA_FEED` to download history bars until the bot calls `hmm.fit()`, or simply run `python run_backtest.py --train-hmm`.

**Q: "Error: Symbol NOT FOUND"**
- **A:** Your broker might use suffixes like `EURUSD_micro`. Change `symbol` in `settings.yaml` to match your symbol list exactly.