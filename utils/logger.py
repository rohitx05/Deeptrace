"""Structured logging with console + file output."""

import logging
import sys
from pathlib import Path
from datetime import datetime


_configured = False


def get_logger(name: str = "deepfake_detector", log_dir: str = "logs/") -> logging.Logger:
    """Get or create a configured logger."""
    global _configured
    logger = logging.getLogger(name)

    if not _configured:
        logger.setLevel(logging.DEBUG)
        formatter = logging.Formatter(
            "[%(asctime)s] [%(levelname)s] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        # Console handler
        console = logging.StreamHandler(sys.stdout)
        console.setLevel(logging.INFO)
        console.setFormatter(formatter)
        logger.addHandler(console)

        # File handler
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_handler = logging.FileHandler(log_path / f"run_{timestamp}.log")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        _configured = True

    return logger
