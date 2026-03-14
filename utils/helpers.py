import datetime

def is_time_in_range(start_time_str, end_time_str, current_time):
    """
    Checks if current_time is between start_time and end_time (hh:mm).
    """
    start_time = datetime.datetime.strptime(start_time_str, "%H:%M").time()
    end_time = datetime.datetime.strptime(end_time_str, "%H:%M").time()
    current_time_time = current_time.time()

    if start_time <= end_time:
        return start_time <= current_time_time <= end_time
    else: # Over midnight
        return start_time <= current_time_time or current_time_time <= end_time

def check_trading_session(config, candle_time):
    """
    Session filter using BROKER CANDLE TIME ONLY.
    """

    current_time = candle_time

    # STEP 1: Weekday filter
    # 0 = Monday, 6 = Sunday
    # if current_time.weekday() == 0:      # Monday
    #     return False
    if current_time.weekday() >= 5:      # Saturday / Sunday
        return False

    # STEP 2: Session windows (BROKER TIME)
    is_asian = is_time_in_range(
        config['sessions']['asian_start'],
        config['sessions']['asian_end'],
        current_time
    )

    is_london = is_time_in_range(
        config['sessions']['london_mid_start'],
        config['sessions']['london_mid_end'],
        current_time
    )

    return is_asian or is_london

def check_news_impact():
    """
    Placeholder news filter.
    Always returns True (safe to trade).
    """
    return True

