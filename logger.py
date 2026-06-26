"""
logger.py
---------
One place to configure logging so every module reports consistently
instead of scattering print() calls. Import get_logger(__name__) at the
top of any module that needs to log.
"""

import logging

_CONFIGURED = False


def _configure_once():
    global _CONFIGURED
    if _CONFIGURED:
        return
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    _configure_once()
    return logging.getLogger(name)