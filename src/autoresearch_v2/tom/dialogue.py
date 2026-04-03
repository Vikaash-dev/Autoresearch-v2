from __future__ import annotations

from dataclasses import dataclass, field

from .council import ReviewOutcome, ToMReviewerCouncil


@dataclass(slots=True)
class DialogueTurn:
    round_id: int
    reviewer_feedback: dict[str, float]
    author_rebuttal: str


@dataclass(slots=True)
class DialogueResult:
    accepted: bool
    rounds: list[DialogueTurn] = field(default_factory=list)
    final_outcome: ReviewOutcome | None = None


class AdversarialDialogueProtocol:
    def __init__(self, council: ToMReviewerCouncil, max_rounds: int = 3) -> None:
        self.council = council
        self.max_rounds = max_rounds

    def run(self, manuscript: str) -> DialogueResult:
        rounds: list[DialogueTurn] = []
        working_manuscript = manuscript
        final_outcome: ReviewOutcome | None = None
        for round_id in range(1, self.max_rounds + 1):
            outcome = self.council.review(working_manuscript)
            final_outcome = outcome
            rounds.append(
                DialogueTurn(
                    round_id=round_id,
                    reviewer_feedback=outcome.scores,
                    author_rebuttal=f"Round {round_id}: addressed reviewer concerns with evidence.",
                )
            )
            if outcome.accepted:
                return DialogueResult(accepted=True, rounds=rounds, final_outcome=outcome)
            working_manuscript += "\n\nRevision: added clarifications and evidence."
        return DialogueResult(accepted=False, rounds=rounds, final_outcome=final_outcome)

