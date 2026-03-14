# Advanced Mean Reversion MT5 Trading Bot

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)
![Platform](https://img.shields.io/badge/platform-MetaTrader5-green.svg)

A professional-grade, fully automated trading bot designed for MetaTrader 5 (MT5). This system implements a sophisticated Mean Reversion strategy, leveraging statistical indicators like Z-Score, Hurst Exponent, and Volatility adjustments to identify high-probability reversal points. It features robust risk management, session filtering, and a modular architecture.

## 🚀 Key Features

*   **Statistical Strategy**: Uses Z-Score deviations from a Rolling VWAP to find overextended price levels.
*   **Regime Filtering**: Incorporates Hurst Exponent and Volatility Slope to filter out trending markets and trade only during favorable mean-reverting conditions.
*   **Dynamic Entry/Exit**: Waits for confirmation ("Signal State Machine") before entering (e.g., waiting for Z-Score to hook back) and targets volatility-adjusted take-profits.
*   **Risk Management**:
    *   Dynamic Lot Sizing based on Account Risk % and Volatility-based Stop Loss.
    *   Daily Drawdown Hard Stop (stops trading if daily loss exceeds a set limit).
    *   Session Filtering (Trades only during specific sessions like Asian or London).
*   **Robust Architecture**:
    *   **Modular Design**: Separated concerns (Data Feed, Strategy, Execution, Risk, Validation).
    *   **Resilient**: Handles connection drops, data delays, and daily resets automatically.
    *   **Safe**: Checks for existing positions on startup and prevents double-entry.

## 📂 Project Structure

```text
.
├── config/                 # Configuration files
│   ├── settings.yaml       # Main bot settings (Strategy, Risk, Sessions)
│   └── symbols.yaml        # Symbol specific specs (contract size, lots)
├── core/                   # Core system logic
│   ├── mt5_connector.py    # MT5 connection handler
│   ├── data_feed.py        # Fetches live candle data
│   ├── execution.py        # Order placement and execution
│   └── risk.py             # Position sizing and drawdown logic
├── strategy/               # Trading Logic
│   ├── mean_reversion.py   # Main Mean Reversion strategy implementation
│   └── strategy_base.py    # Abstract base class
├── services/               # Background services
│   ├── trade_manager.py    # Manages open trades (trailing SL, partials)
│   └── order_validator.py  # Validates trades before execution
├── utils/                  # Helper functions
│   ├── indicators.py       # TA Library (Z-Score, RSI, Hurst, VWAP)
│   ├── session_filter.py   # Time-based session filters
│   └── logger.py           # Logging configuration
├── logs/                   # Log output directory
├── main.py                 # Entry point
└── requirements.txt        # Python dependencies
```

## 🛠️ Installation & Setup

### Prerequisites
1.  **Windows OS** (MT5 Python integration is Windows-native).
2.  **MetaTrader 5 Terminal** installed and logged into a hedging account.
3.  **Python 3.9+** installed.

### Steps
1.  **Clone the Repository**:
    ```bash
    git clone https://github.com/yourusername/mt5-trading-bot.git
    cd mt5-trading-bot
    ```

2.  **Create Virtual Environment** (Recommended):
    ```bash
    python -m venv venv
    .\venv\Scripts\activate
    ```

3.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

4.  **Configure MT5**:
    *   Open MetaTrader 5.
    *   Go to **Tools** -> **Options** -> **Expert Advisors**.
    *   Enable "Allow algorithmic trading".

## ⚙️ Configuration

The bot is fully configurable via `config/settings.yaml`.

**Key Parameters:**
*   **Trading**: Symbol (`EURUSD`), Timeframe (`M1`), Risk Per Trade (e.g., `0.01` for 1%).
*   **Strategy**:
    *   `z_score_trigger`: Deviation required to look for a trade (e.g., `2.5`).
    *   `hurst_threshold`: Max Hurst value to allow trading (filters trends).
    *   `rsi_overbought/oversold`: RSI filters for entry confirmation.
*   **Sessions**: Define start/end times for trading windows (Asian/London).
*   **Risk**: `max_daily_drawdown_pct` sets the daily loss limit.

## ▶️ Usage

1.  **Start MetaTrader 5**: Ensure your terminal is running and connected to the internet.
2.  **Run the Bot**:
    ```bash
    python main.py
    ```
3.  **Operation**:
    *   The bot will connect to MT5.
    *   It will wait for the next candle close to process data.
    *   Logs will be printed to the console and saved in `logs/bot.log`.
    *   Use `Ctrl+C` to stop the bot safely.

## 📊 Strategy Details

The **Mean Reversion Strategy** operates on a refined state machine:
1.  **Filter**: Checks if the market is in a "ranging" regime (Hurst < Threshold, Vol Slope low).
2.  **Signal**: Checks if Price deviates significantly from VWAP (Z-Score > Trigger).
3.  **Confirm**: Uses RSI to ensure the move is overextended.
4.  **Wait**: Enters a "Pending" state, waiting for price to hook back towards the mean.
5.  **Execute**: Enters trade when Z-Score starts returning to the mean (crossing Entry Target).
6.  **Exit**: Target Profit is dynamic (Vol-Adjusted), or Hard Stop Loss is hit.

## ⚠️ Disclaimer

Trading Forex/CFDs involves significant risk. This software is for educational purposes only. Do not trade with money you cannot afford to lose. Always Backtest and Paper Trade before going live.
