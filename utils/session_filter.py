from datetime import datetime
import pytz


def is_valid_session():
    utc = datetime.utcnow().replace(tzinfo=pytz.utc)
    hour = utc.hour

    asian = 0 <= hour <= 6
    london_mid = 9 <= hour <= 12

    return asian or london_mid
