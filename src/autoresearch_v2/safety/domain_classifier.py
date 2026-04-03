from __future__ import annotations


class DomainClassifier:
    SENSITIVE_KEYWORDS = ("bio", "medical", "cyber", "weapon")

    def classify(self, topic: str) -> str:
        lowered = topic.lower()
        if any(keyword in lowered for keyword in self.SENSITIVE_KEYWORDS):
            return "sensitive"
        return "general"

