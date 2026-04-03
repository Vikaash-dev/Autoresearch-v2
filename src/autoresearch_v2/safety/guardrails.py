from __future__ import annotations


class SafetyGuardrails:
    def enforce(self, topic: str, domain: str) -> list[str]:
        warnings: list[str] = []
        if domain == "sensitive":
            warnings.append("human review recommended for sensitive domain")
        if not topic.strip():
            warnings.append("empty topic is not allowed")
        return warnings

