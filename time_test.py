import datetime
from zoneinfo import ZoneInfo

time_stamp = "2025-11-13 01:40:00"

# Parse the string as IST timezone
time_ = datetime.datetime.strptime(time_stamp, "%Y-%m-%d %H:%M:%S")
ist_time = time_.replace(tzinfo=ZoneInfo("Asia/Kolkata"))

# Convert to UTC for database storage
utc_time = ist_time.astimezone(ZoneInfo("UTC"))

print(f"Original IST: {ist_time},hour: {ist_time.hour},minute: {ist_time.minute}")
print(f"UTC for DB: {utc_time},hour: {utc_time.hour},minute: {utc_time.minute}")
print(f"ISO format: {utc_time.isoformat()}")
print(f"IST time in iso format: {ist_time.isoformat()}")
