#!/usr/bin/env python
"""
Basic usage: run SERA-X on a research task for 3 rounds.

python examples/basic_research.py
"""
import logging
from autoresearch import ResearchOrchestrator, AutoresearchConfig

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(name)s: %(message)s")
for m in ["autoresearch.pipeline", "autoresearch.memory", "autoresearch.evolution"]:
    logging.getLogger(m).setLevel(logging.WARNING)


def main():
    config = AutoresearchConfig()
    config.verbose = True

    orch = ResearchOrchestrator(
        config=config,
        db_path="research_memory.db",
    )

    result = orch.run(
        task="What methods surpass HyperAgents for autonomous machine learning research?",
        max_rounds=3,
        search_breadth=4,
        search_depth=2,
        success_threshold=0.75,
    )

    orch.print_summary()

    # Print the best report
    best = result.get("best_report", {})
    markdown = best.get("markdown", "")
    if markdown:
        print("\n" + "=" * 60)
        print(markdown[:2000])
        if len(markdown) > 2000:
            print(f"... [{len(markdown) - 2000} more chars — save with --output]")

    # Save to file
    with open("research_report.md", "w", encoding="utf-8") as f:
        f.write(markdown or "No report generated.")
    print("\nReport saved to: research_report.md")


if __name__ == "__main__":
    main()
