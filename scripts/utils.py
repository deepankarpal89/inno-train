from datetime import datetime
from typing import Optional, Union
from zoneinfo import ZoneInfo

# IST timezone
IST = ZoneInfo("Asia/Kolkata")

def ist_now_isoformat() -> str:
    """Get current time in IST timezone as ISO format string."""
    return ist_now().isoformat()

def ist_now() -> datetime:
    """Get current time in IST timezone as datetime object."""
    return datetime.now(IST).replace(microsecond=0)


def parse_timestamp(timestamp_str: str) -> Optional[datetime]:
    """Parse timestamp string to IST-aware datetime object.

    Note: Input timestamps are assumed to already be in IST timezone.
    """
    if not timestamp_str:
        return None
    try:
        dt = datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S")
        dt = dt.replace(tzinfo=IST)
        return dt
    except Exception:
        return None


def calculate_duration(start_time: Union[datetime, str], end_time: Union[datetime, str]) -> Union[float, None]:
    """Calculate duration in minutes between two datetime objects or ISO strings.
    Args:
        start_time: Start time as datetime object or ISO string
        end_time: End time as datetime object or ISO string
    Returns:
        Duration in minutes as float, or None if calculation fails
    """
    if not start_time or not end_time:
        return None

    try:
        # Convert start_time to datetime if it's a string
        if isinstance(start_time, str):
            start_dt = datetime.fromisoformat(start_time)
        else:
            start_dt = start_time

        # Convert end_time to datetime if it's a string
        if isinstance(end_time, str):
            end_dt = datetime.fromisoformat(end_time)
        else:
            end_dt = end_time

        return (end_dt - start_dt).total_seconds() / 60.0
    except Exception:
        return None

if __name__ == "__main__":
    print(ist_now())
    print(parse_timestamp("2025-11-13 01:40:00"))
