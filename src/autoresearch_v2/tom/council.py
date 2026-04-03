from __future__ import annotations

from dataclasses import dataclass


DEFAULT_PERSONAS = (
    "skeptic",
    "engineer",
    "visionary",
    "domain_expert",
    "ethicist",
)


@dataclass(slots=True)
class ReviewOutcome:
    accepted: bool
    average_score: float
    scores: dict[str, float]


class ToMReviewerCouncil:
    def __init__(self, personas: tuple[str, ...] = DEFAULT_PERSONAS, acceptance_threshold: float = 4.5) -> None:
        self.personas = personas
        self.acceptance_threshold = acceptance_threshold

    def review(self, manuscript: str) -> ReviewOutcome:
        base = 3.5 + min(len(manuscript), 5000) / 5000.0
        scores = {persona: min(7.0, round(base, 2)) for persona in self.personas}
        average = sum(scores.values()) / len(scores)
        return ReviewOutcome(accepted=average >= self.acceptance_threshold, average_score=average, scores=scores)

