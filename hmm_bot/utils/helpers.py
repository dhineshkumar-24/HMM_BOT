"""
utils/helpers.py — Session detection and utility helper functions.

All session checks use BROKER candle time, not local wall-clock time.
"""

from __future__ import annotations
import datetime


# ─────────────────────────────────────────────────────────────────────────────
# Session label constants
# ─────────────────────────────────────────────────────────────────────────────
SESSION_NONE   = "none"
SESSION_ASIAN  = "asian"
SESSION_LONDON = "london"
SESSION_NY     = "newyork"


# ─────────────────────────────────────────────────────────────────────────────
# Core time utility
# ─────────────────────────────────────────────────────────────────────────────

def is_time_in_range(
    start_str: str,
    end_str: str,
    current_time: datetime.datetime,
) -> bool:
    """
    Return True if current_time.time() is within [start_str, end_str].
    Times are "HH:MM" strings. Handles overnight ranges.
    """
    start = datetime.datetime.strptime(start_str, "%H:%M").time()
    end   = datetime.datetime.strptime(end_str,   "%H:%M").time()
    t     = current_time.time()

    if start <= end:
        return start <= t <= end
    else:               # overnight (e.g. 22:00 → 02:00)
        return t >= start or t <= end


# ─────────────────────────────────────────────────────────────────────────────
# Per-session checks
# ─────────────────────────────────────────────────────────────────────────────

def is_asian_session(config: dict, candle_time: datetime.datetime) -> bool:
    """Return True if the candle falls inside the Asian session window."""
    s = config["sessions"]
    return is_time_in_range(s["asian_start"], s["asian_end"], candle_time)


def is_london_session(config: dict, candle_time: datetime.datetime) -> bool:
    """Return True if the candle falls inside the London session window."""
    s = config["sessions"]
    return is_time_in_range(s["london_start"], s["london_end"], candle_time)


def is_newyork_session(config: dict, candle_time: datetime.datetime) -> bool:
    """Return True if the candle falls inside the New York session window."""
    s = config["sessions"]
    return is_time_in_range(s["newyork_start"], s["newyork_end"], candle_time)


def detect_session(config: dict, candle_time: datetime.datetime) -> str:
    """
    Identify which trading session the candle belongs to.

    Priority: Asian → London → New York → none.
    Overlapping windows (London/NY overlap) → London takes priority.

    Returns one of: SESSION_ASIAN, SESSION_LONDON, SESSION_NY, SESSION_NONE.
    """
    if is_asian_session(config, candle_time):
        return SESSION_ASIAN
    if is_london_session(config, candle_time):
        return SESSION_LONDON
    if is_newyork_session(config, candle_time):
        return SESSION_NY
    return SESSION_NONE


# ─────────────────────────────────────────────────────────────────────────────
# Combined session guard (backward-compatible)
# ─────────────────────────────────────────────────────────────────────────────

def check_trading_session(config: dict, candle_time: datetime.datetime) -> bool:
    """
    Return True if the candle is in any active trading session
    and is not on a weekend.

    Backward-compatible — used by main.py.
    """
    if candle_time.weekday() >= 5:      # Sat=5, Sun=6
        return False
    return detect_session(config, candle_time) != SESSION_NONE


def check_news_impact() -> bool:
    """
    News event filter — placeholder, always returns True (safe to trade).
    Replace with live economic calendar API when ready.
    """
    return True
