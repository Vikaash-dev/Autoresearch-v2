from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class SkillRegistry:
    skills: dict[str, str] = field(default_factory=dict)

    def add(self, name: str, description: str) -> None:
        self.skills[name] = description

    def list(self) -> dict[str, str]:
        return dict(self.skills)

