from __future__ import annotations

from dataclasses import dataclass, field

from autoresearch_v2.agents.coder_agent import CoderAgent
from autoresearch_v2.agents.critic_agent import CriticAgent
from autoresearch_v2.agents.hypothesis_agent import HypothesisAgent
from autoresearch_v2.agents.literature_agent import LiteratureAgent
from autoresearch_v2.agents.writer_agent import WriterAgent
from autoresearch_v2.compute.planner import ComputePlanner
from autoresearch_v2.compute.simulator import ComputeSimulator
from autoresearch_v2.discovery.branch import Branch
from autoresearch_v2.discovery.grafter import BranchGrafter
from autoresearch_v2.discovery.forest import DiscoveryForest
from autoresearch_v2.discovery.pruner import BranchPruner
from autoresearch_v2.dog.allocator import ComputeBudgetAllocator
from autoresearch_v2.dog.graph import DynamicObjectiveGraph
from autoresearch_v2.dog.objective import Objective
from autoresearch_v2.dog.scheduler import ObjectiveScheduler
from autoresearch_v2.epistemic.pipeline import EpistemicPipeline
from autoresearch_v2.hyper_kernel.kernel import HyperKernel
from autoresearch_v2.hyper_kernel.modifier import BoundedModifier
from autoresearch_v2.hyper_kernel.telemetry import RunTelemetry
from autoresearch_v2.integrations.arxiv_client import ArxivClient
from autoresearch_v2.integrations.openalex_client import OpenAlexClient
from autoresearch_v2.integrations.semantic_scholar import SemanticScholarClient
from autoresearch_v2.memory.engram import EngramMemory
from autoresearch_v2.memory.research_graph import ResearchGraph
from autoresearch_v2.safety.audit_log import AuditLog
from autoresearch_v2.safety.domain_classifier import DomainClassifier
from autoresearch_v2.safety.guardrails import SafetyGuardrails
from autoresearch_v2.tom.dialogue import AdversarialDialogueProtocol
from autoresearch_v2.tom.intent_model import CollaboratorIntentModel
from autoresearch_v2.tom.council import ToMReviewerCouncil


@dataclass(slots=True)
class RunSummary:
    topic: str
    branch_count: int
    ready_objectives: int
    accepted_by_council: bool
    verification_passed: bool
    domain: str = "general"
    safety_warnings: list[str] = field(default_factory=list)
    provenance_entries: int = 0
    telemetry_notes: list[str] = field(default_factory=list)


def run_bootstrap(topic: str, branches: int) -> RunSummary:
    audit = AuditLog()
    classifier = DomainClassifier()
    guardrails = SafetyGuardrails()
    domain = classifier.classify(topic)
    safety_warnings = guardrails.enforce(topic, domain)
    audit.record(f"domain={domain}")

    graph = DynamicObjectiveGraph()
    graph.add_objective(Objective(id="literature", description=f"Ground topic: {topic}", priority=1.0))
    graph.add_objective(
        Objective(
            id="hypothesis",
            description="Generate hypotheses",
            dependencies=["literature"],
            priority=0.9,
        )
    )
    graph.add_objective(
        Objective(
            id="write",
            description="Write manuscript",
            dependencies=["hypothesis"],
            priority=0.8,
        )
    )
    scheduler = ObjectiveScheduler(graph)
    allocator = ComputeBudgetAllocator(total_budget=10.0)
    assignments = scheduler.schedule(["literature_agent", "hypothesis_agent", "writer_agent"], limit=1)
    budgets = allocator.allocate([a.objective_id for a in assignments])
    for a in assignments:
        graph.mark_complete(a.objective_id, outputs={"budget": budgets.get(a.objective_id, 0.0)})
    ready = graph.ready_objectives()

    arxiv = ArxivClient()
    openalex = OpenAlexClient()
    ss = SemanticScholarClient()
    candidates = arxiv.search(topic, 4) + openalex.search(topic, 3) + ss.search(topic, 3)

    literature_agent = LiteratureAgent()
    lit_result = literature_agent.execute(graph.get("literature"), {"topic": topic, "literature_candidates": candidates})
    graph.mark_complete("literature", outputs=lit_result.outputs)

    hypothesis_agent = HypothesisAgent()
    hypo_result = hypothesis_agent.execute(
        graph.get("hypothesis"),
        {"topic": topic, "max_hypotheses": max(1, branches)},
    )
    graph.mark_complete("hypothesis", outputs=hypo_result.outputs)

    coder_agent = CoderAgent()
    code_result = coder_agent.execute(graph.get("hypothesis"), {"hypotheses": hypo_result.outputs.get("hypotheses", [])})

    writer_agent = WriterAgent()
    writer_result = writer_agent.execute(
        graph.get("write"),
        {
            "topic": topic,
            "hypotheses": hypo_result.outputs.get("hypotheses", []),
            "citations": lit_result.outputs.get("citations", []),
        },
    )
    graph.mark_complete("write", outputs=writer_result.outputs)

    forest = DiscoveryForest()
    for i in range(branches):
        forest.add_branch(Branch(id=f"branch-{i+1}", hypothesis=f"H{i+1} for {topic}", budget=1.0))
        forest.score_branch(f"branch-{i+1}", score=0.7 if i == 0 else 0.1)
    pruned = BranchPruner(prune_floor=0.15).prune(forest)
    grafter = BranchGrafter(min_score=0.6)
    if branches >= 2:
        grafter.graft(forest, "branch-1", "branch-2")

    council = ToMReviewerCouncil()
    dialogue = AdversarialDialogueProtocol(council=council, max_rounds=3)
    dialogue_result = dialogue.run(writer_result.outputs.get("manuscript", ""))

    epistemic = EpistemicPipeline()
    verification = epistemic.verify_claim(
        claim=f"Bootstrap run for {topic}",
        citations=lit_result.outputs.get("citations", []),
        execution_log=code_result.outputs.get("execution_log", []),
        formal_required=False,
    )

    compute_plan = ComputePlanner(total_budget=100).plan({"exploration": 1.0, "verification": 1.0, "writing": 0.5})
    cim = CollaboratorIntentModel(risk_appetite=0.6, rigor_vs_speed=0.8, venue_bias=0.5, collaboration_style=0.4)
    adjusted_compute_plan = cim.adjust_budgets(compute_plan)
    estimate = ComputeSimulator().estimate_duration(objective_count=len(graph.objectives))
    audit.record(f"compute_estimate_sec={estimate:.1f}")

    research_graph = ResearchGraph()
    research_graph.add_note(topic, f"candidate_refs={len(candidates)}")
    engram = EngramMemory()
    engram.upsert_skill("bootstrap", "Runs baseline DOG+Discovery+ToM+Epistemic pipeline")

    critic_agent = CriticAgent()
    critic_result = critic_agent.execute(graph.get("write"), writer_result.outputs)
    audit.record(f"critic_score={critic_result.outputs.get('critic_score', 0)}")
    audit.record(f"pruned_branches={len(pruned)}")
    audit.record(f"verification_budget={adjusted_compute_plan.get('verification', 0.0):.2f}")

    telemetry = RunTelemetry(
        agent_latency={
            "literature_agent": literature_agent.get_telemetry().avg_latency_sec,
            "hypothesis_agent": hypothesis_agent.get_telemetry().avg_latency_sec,
            "writer_agent": writer_agent.get_telemetry().avg_latency_sec,
        },
        failure_rates={
            "literature_agent": 0.0,
            "hypothesis_agent": 0.0,
            "writer_agent": 0.0,
        },
        notes=["bootstrap telemetry collected"],
    )
    hk = HyperKernel(modifier=BoundedModifier())
    mod = hk.optimize_agent("writer_agent", telemetry)
    if mod:
        hk.apply_modification(mod)
        audit.record(f"hyper_kernel_mod={mod.change}")

    return RunSummary(
        topic=topic,
        branch_count=branches,
        ready_objectives=len(ready),
        accepted_by_council=dialogue_result.accepted,
        verification_passed=verification.passed,
        domain=domain,
        safety_warnings=safety_warnings,
        provenance_entries=len(epistemic.get_provenance_graph().entries),
        telemetry_notes=audit.events,
    )
