"""Logging helpers with an optional Loguru dependency."""

from __future__ import annotations

import logging

try:  # pragma: no cover - exercised when loguru is available
    from loguru import logger
except ImportError:  # pragma: no cover - exercised in lean environments
    logger = logging.getLogger("heavy_metal_hpc")
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(levelname)s | %(message)s"))
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)
