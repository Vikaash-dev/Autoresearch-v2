from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class EngramMemory:
    skills: dict[str, str] = field(default_factory=dict)

    def upsert_skill(self, skill_name: str, description: str) -> None:
        self.skills[skill_name] = description

    def get_skill(self, skill_name: str) -> str | None:
        return self.skills.get(skill_name)

