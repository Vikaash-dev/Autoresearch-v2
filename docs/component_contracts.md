# AutoResearch v2 — Component Contracts & Interfaces

## 1. `BaseAgent`

All agents extend `autoresearch.agents.base.BaseAgent`.

```python
class BaseAgent(ABC):
    name: str          # unique agent identifier
    role: str          # human-readable role label
    config: AutoResearchConfig

    def run(
        self,
        objective: Objective,
        state: RunState,
        config: AutoResearchConfig,
    ) -> Any: ...
    # Reads from state, writes enriched data back to state.
    # Must NOT raise unhandled exceptions — let AgentRuntime catch them.
```

**Contract:**
- Must be stateless between calls (state is passed in every time).
- Must write any persistent results back to `state` before returning.
- May call `_llm_prompt(prompt: str) -> str` for LLM interactions.

---

## 2. `ObjectiveGraph` + `Objective`

```python
class Objective(BaseModel):
    id: str
    name: str
    description: str = ""
    dependencies: List[str] = []   # IDs of prerequisite objectives
    status: ObjectiveStatus        # pending | running | completed | failed | skipped
    result: Any = None
    priority: int = 0              # higher = runs first among ready objectives

class ObjectiveGraph:
    def add(self, objective: Objective, handler: Callable) -> None: ...
    def get_ready(self) -> List[Objective]: ...  # deps all COMPLETED
    def mark_running(self, obj_id: str) -> None: ...
    def mark_completed(self, obj_id: str, result: Any) -> None: ...
    def mark_failed(self, obj_id: str, error: str) -> None: ...
    def is_done(self) -> bool: ...
    def summary(self) -> Dict[str, str]: ...
```

---

## 3. `RunState`

Pydantic v2 model, serialised as JSON. All agents read/write to this shared state.

```python
class RunState(BaseModel):
    run_id: str
    topic: str
    status: str           # initialized | running | paused | completed | failed
    iteration: int
    objectives_completed: List[str]
    evidence: List[Dict]         # EvidenceItem dicts
    hypotheses: List[Dict]       # {id, title, description, testability_score, novelty_score, status}
    experiments: List[Dict]      # ExperimentSpec dicts + result
    reviews: List[Dict]          # adversarial review outputs
    reflections: List[Dict]      # SelfReviewModule outputs
    policy_updates: List[Dict]   # applied safe policy updates
    metadata: Dict[str, Any]     # freeform (plan, tom_negotiation, etc.)

    def save(self, directory: Path) -> Path: ...
    @classmethod
    def load(cls, directory: Path) -> "RunState": ...
```

---

## 4. `EvidenceItem` (Literature Registry Schema)

```python
class EvidenceItem(BaseModel):
    id: str             # auto UUID prefix
    source: str         # "arxiv" | "semantic_scholar" | "openalex"
    title: str
    abstract: str
    authors: List[str]
    year: Optional[str]
    url: Optional[str]
    doi: Optional[str]
    arxiv_id: Optional[str]
    tags: List[str]
    claims: List[str]   # extracted claims (populated by CitationGrounding)
    verified: bool      # grounding check passed
    metadata: Dict[str, Any]
```

---

## 5. `ExperimentSpec`

```python
class ExperimentSpec(BaseModel):
    id: str             # "exp_" + uuid prefix
    hypothesis_id: str
    title: str
    description: str
    code: str           # runnable Python code
    language: str       # "python" (extensible)
    dependencies: List[str]
    inputs: Dict[str, Any]
    expected_outputs: List[str]
    timeout_seconds: int
    max_retries: int
    status: str         # pending | running | success | failed
    result: Optional[Dict]
    artifacts: List[str]
    metrics: Dict[str, Any]
    provenance: Dict[str, Any]
```

---

## 6. `ArxivConnector` / `ScholarlyConnector`

```python
class ArxivConnector:
    def search(self, query: str) -> List[Dict[str, Any]]:
        # Returns list of paper dicts with keys:
        # source, arxiv_id, title, abstract, published, url, authors

class ScholarlyConnector:
    def search(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        # Backend-agnostic; supports "semantic_scholar" and "openalex"
        # Returns list with: source, title, abstract, year, authors, doi
```

Both return `[]` on network errors — callers must handle empty lists gracefully.

---

## 7. `ReviewerPersona` + `ReviewerPersonaRegistry`

```python
class ReviewerPersona(BaseModel):
    name: str
    label: str
    description: str
    priorities: List[str]
    rejection_triggers: List[str]
    acceptance_criteria: List[str]
    bias_profile: Dict[str, float]   # e.g. {"novelty": 0.3, "rigor": 0.9}

class ReviewerPersonaRegistry:
    def get(self, name: str) -> Optional[ReviewerPersona]: ...
    def all(self) -> List[ReviewerPersona]: ...
    def register(self, persona: ReviewerPersona) -> None: ...
```

Built-in personas: `methodological_purist`, `empirical_skeptic`, `novelty_maximizer`,
`benchmark_enforcer`.

---

## 8. `SandboxExecutor` + `SelfHealingLoop`

```python
class SandboxExecutor:
    def execute(self, experiment: Dict[str, Any]) -> Dict[str, Any]:
        # Returns: {status, stdout, stderr, returncode, attempts}

class SelfHealingLoop:
    def run(self, code: str, timeout: int) -> Dict[str, Any]:
        # Runs code, detects failures, applies patches, retries up to max_retries
        # Returns: {status, stdout, stderr, attempts, healed}
```

Supported auto-repairs:
- `ModuleNotFoundError` → inject import stub
- `SyntaxError` → normalise indentation
- `NameError` → inject variable stub (`None`)

---

## 9. `SelfReviewModule`

```python
class SelfReviewModule:
    def analyze(self, state: RunState) -> Dict[str, Any]:
        # Returns:
        # {
        #   run_id, iteration,
        #   bottlenecks: [{module, issue, severity, metric, threshold}],
        #   quality_scores: {evidence, hypotheses, experiments, reviews, overall},
        #   proposed_policy_updates: [{module, parameter, current, proposed, rationale, safe}],
        #   summary: str,
        # }
```

---

## 10. `AutoResearchConfig` (key fields)

```yaml
topic: ""
max_iterations: 3
stop_criteria:
  min_confidence: 0.7

llm:
  provider: openai        # extensible
  model: gpt-4o
  temperature: 0.7

literature:
  arxiv_max_results: 20
  scholarly_backend: semantic_scholar

experiment:
  sandbox: subprocess     # or "docker"
  max_retries: 3
  timeout_seconds: 300

tom:
  enabled: true
  reviewer_personas:
    - methodological_purist
    - empirical_skeptic
    - novelty_maximizer
  adversarial_rounds: 2

reflection:
  enabled: true
  safe_modification_only: true
  max_policy_updates_per_run: 5

checkpoint:
  enabled: true
  save_every_n_steps: 1
```
