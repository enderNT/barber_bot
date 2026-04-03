from __future__ import annotations

import logging

_logger = logging.getLogger("barbershop.router_input")
_enabled = False


def configure_router_input_logger(enabled: bool) -> None:
    global _enabled
    _enabled = enabled

    if not enabled:
        _logger.handlers.clear()
        _logger.propagate = False
        return

    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(message)s"))
    _logger.handlers.clear()
    _logger.addHandler(handler)
    _logger.setLevel(logging.INFO)
    _logger.propagate = False


def log_router_input(router_input: str) -> None:
    if not _enabled:
        return
    _logger.info("ESTO ES EL CONTEXTO WE:\n%s\n", router_input)
