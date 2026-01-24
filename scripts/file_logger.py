import logging
import json

from scripts.utils import ist_now_isoformat


class JsonFormatter(logging.Formatter):
    def format(self, record):
        log_record = {
            "timestamp": ist_now_isoformat(),
            "logger": record.name,
            "module": record.module,
            "level": record.levelname,
            "message": record.getMessage(),
            "function": record.funcName,
            "line": record.lineno,
            "thread": record.threadName,
        }
        if record.exc_info:
            log_record["exception"] = self.formatException(record.exc_info)
        return json.dumps(log_record)


def get_file_logger(file_name):
    file_logger = logging.getLogger(file_name)
    file_logger.setLevel(logging.DEBUG)

    # Remove any existing handlers to prevent duplicates            
    if file_logger.hasHandlers():
        file_logger.handlers.clear()

    # Add file handler
    file_handler = logging.FileHandler(f"logs/{file_name}.log")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(JsonFormatter())
    file_logger.addHandler(file_handler)

    # Prevent propagation to root logger
    file_logger.propagate = False

    return file_logger
