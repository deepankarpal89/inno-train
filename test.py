from scripts.utils import parse_timestamp
event = {"timestamp": "2025-11-12 21:06:47", "level": "INFO", "phase": "EVAL_MODEL", "event": "end", "message": "Finished evaluation for cv_epoch_1", "data": {"elapsed": 1733579110.71}}
event_timestamp = parse_timestamp(event.get("timestamp"))
print(event_timestamp)
print(type(event_timestamp))
