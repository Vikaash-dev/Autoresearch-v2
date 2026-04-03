from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class ProvenanceEntry:
    claim: str
    literature_note: str
    code_note: str
    formal_note: str


@dataclass(slots=True)
class ClaimProvenanceGraph:
    entries: list[ProvenanceEntry] = field(default_factory=list)

    def add_entry(self, entry: ProvenanceEntry) -> None:
        self.entries.append(entry)

