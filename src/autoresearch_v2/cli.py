from __future__ import annotations

import argparse

from autoresearch_v2.core.runtime import run_bootstrap


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="AutoResearch v2 bootstrap CLI")
    parser.add_argument("--topic", required=True, help="Research topic")
    parser.add_argument("--branches", type=int, default=3, help="Number of discovery branches")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    summary = run_bootstrap(topic=args.topic, branches=max(1, args.branches))
    print(
        f"topic={summary.topic}; branches={summary.branch_count}; ready_objectives={summary.ready_objectives}; "
        f"accepted_by_council={summary.accepted_by_council}; verification_passed={summary.verification_passed}; "
        f"domain={summary.domain}; provenance_entries={summary.provenance_entries}; "
        f"safety_warnings={len(summary.safety_warnings)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
