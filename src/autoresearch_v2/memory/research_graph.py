from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class ResearchGraph:
    topic_to_notes: dict[str, list[str]] = field(default_factory=dict)

    def add_note(self, topic: str, note: str) -> None:
        self.topic_to_notes.setdefault(topic, []).append(note)

    def get_notes(self, topic: str) -> list[str]:
        return list(self.topic_to_notes.get(topic, []))

