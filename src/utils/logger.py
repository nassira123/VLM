from pathlib import Path
from typing import Optional

from loguru import logger


def setup_logger(log_path: Optional[str] = None):
    logger.remove()
    logger.add(lambda msg: print(msg, end=""))
    if log_path:
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)
        logger.add(log_path, rotation="10 MB")
    return logger
