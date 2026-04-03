from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from app.tracing.types import FieldPolicy


class AllowlistRedactionPolicy(FieldPolicy):
    def __init__(
        self,
        *,
        allowed_keys: set[str] | None = None,
        redact_keys: set[str] | None = None,
        redaction_text: str = "[REDACTED]",
    ) -> None:
        self._allowed_keys = {key.lower() for key in (allowed_keys or set())}
        self._redact_keys = {
            "api_key",
            "authorization",
            "token",
            "access_token",
            "refresh_token",
            "password",
            "secret",
        }
        self._redact_keys.update(key.lower() for key in (redact_keys or set()))
        self._redaction_text = redaction_text

    def sanitize(self, payload: Any) -> Any:
        return self._sanitize(payload)

    def _sanitize(self, payload: Any) -> Any:
        if isinstance(payload, Mapping):
            sanitized: dict[str, Any] = {}
            for key, value in payload.items():
                normalized_key = str(key).lower()
                if self._allowed_keys and normalized_key not in self._allowed_keys and self._looks_like_secret_key(
                    normalized_key
                ):
                    continue
                if normalized_key in self._redact_keys or self._looks_like_secret_key(normalized_key):
                    sanitized[str(key)] = self._redaction_text
                    continue
                sanitized[str(key)] = self._sanitize(value)
            return sanitized

        if isinstance(payload, Sequence) and not isinstance(payload, (str, bytes, bytearray)):
            return [self._sanitize(item) for item in payload]

        return payload

    def _looks_like_secret_key(self, key: str) -> bool:
        return any(marker in key for marker in ("token", "secret", "password", "authorization", "api_key"))


class NoopFieldPolicy(FieldPolicy):
    def sanitize(self, payload: Any) -> Any:
        return payload
