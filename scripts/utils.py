from datetime import datetime
from typing import Optional, Union
from zoneinfo import ZoneInfo

# IST timezone
IST = ZoneInfo("Asia/Kolkata")


def ist_now() -> str:
    """Get current time in IST timezone as ISO format string (no microseconds)."""
    return datetime.now(IST).replace(microsecond=0).isoformat()


def parse_timestamp(timestamp_str: str) -> Optional[str]:
    """Parse timestamp string to IST-aware datetime object.

    Note: Input timestamps are assumed to already be in IST timezone.
    """
    if not timestamp_str:
        return None
    try:
        dt = datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S")
        dt = dt.replace(tzinfo=IST)
        return dt.isoformat()
    except Exception:
        return None


def calculate_duration(start_time: str, end_time: str) -> Union[float, None]:
    """Calculate duration in minutes between two ISO format timestamps.

    Args:
        start_time: Start time in ISO format string
        end_time: End time in ISO format string

    Returns:
        Duration in minutes as float, or None if calculation fails
    """
    if not start_time or not end_time:
        return None

    try:
        start_dt = datetime.fromisoformat(start_time)
        end_dt = datetime.fromisoformat(end_time)
        return (end_dt - start_dt).total_seconds() / 60.0
    except Exception:
        return None


if __name__ == "__main__":
    print(ist_now())
    print(parse_timestamp("2025-11-13 01:40:00"))
