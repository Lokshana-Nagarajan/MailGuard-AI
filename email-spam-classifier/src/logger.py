"""
logger.py
---------
Centralized logging setup for production deployments. Writes to a
rotating file under logs/ as well as stdout (so it's captured by
`docker logs`, Heroku/Render log streams, etc.) without needing any
external logging service.
"""

from __future__ import annotations

import logging
import os
from logging.handlers import RotatingFileHandler

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOGS_DIR = os.path.join(PROJECT_ROOT, "logs")


def get_logger(name: str = "mailguard") -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:  # already configured (Streamlit re-imports modules on rerun)
        return logger

    logger.setLevel(logging.INFO)
    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s", "%Y-%m-%d %H:%M:%S"
    )

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(fmt)
    logger.addHandler(stream_handler)

    try:
        os.makedirs(LOGS_DIR, exist_ok=True)
        file_handler = RotatingFileHandler(
            os.path.join(LOGS_DIR, "app.log"), maxBytes=2_000_000, backupCount=3
        )
        file_handler.setFormatter(fmt)
        logger.addHandler(file_handler)
    except OSError:
        # Read-only filesystem (some PaaS containers) -- stdout logging still works.
        pass

    return logger
