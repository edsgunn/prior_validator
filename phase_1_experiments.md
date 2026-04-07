# Phase 1: Prior Quality Validation — Experiment Overview & Code Requirements

## 1. Goal

Before running any CCSM training, we need to answer a foundational question: **do pretrained LLM priors, conditioned on character prompts, encode usable behavioural priors — i.e. does their conditional distribution over trajectories track the kind of behaviour we'd want the trained agent to exhibit?**

A precise framing matters here. CCSM replaces external reward with the model's own observation surprise under a character-conditioned distribution. The model *is* the preference distribution. But "low surprise" means "typical under the model's conditional," and typicality is shaped by the pretraining distribution, not by any objective notion of quality. So Phase 1 is really asking:

> **Does the pretraining distribution, filtered through character conditioning, correlate with behavioural quality?**

This is a weaker claim than "surprise replaces reward" — but it's the necessary empirical foundation. If the pretraining distribution is biased toward competent behaviour in a given domain (because training data contains more expert chess games than random ones, more successful negotiations than incoherent ones), then typicality under the conditioned model will track quality. If the pretraining distribution is *not* biased this way — if it contains as much bad negotiation as good — then surprise will track linguistic conventionality rather than strategic quality, and CCSM will need auxiliary mechanisms (SFT warmup, curated pretraining, etc.) to work.

Phase 1 establishes which regime we're in, per domain, with no training loop — only forward passes through a pretrained model over trajectories of known quality.

The experimental logic is:

1. Construct trajectories of **known, externally-verifiable quality** in controlled environments.
2. Measure the model's **observation surprise** on those trajectories under different character prompts.
3. Test whether surprise **rank-orders trajectories the same way quality does**, after controlling for token-level confounds.
4. Test whether surprise **responds causally** to local quality changes via counterfactual trajectory edits.

We use three environments that test fundamentally different aspects of the prior:

- **Chess** — deterministic, adversarial, zero-sum, with a perfect external evaluator (Stockfish). Tests whether the model's priors encode *strategic competence* in a formal domain where pretraining bias toward quality is strong.
- **Resource trading (negotiation)** — cooperative/competitive, positive-sum, language-native, with objectively scoreable outcomes. Tests whether priors encode *social and economic reasoning* in a domain where pretraining bias toward quality is uncertain.
- **Text gridworld (navigation)** — deterministic, single-agent, with optimal path length as ground truth. Tests priors on *planning and spatial reasoning* in a domain that is structured but language-mediated. Bridges the gap between the formal precision of chess and the linguistic openness of negotiation.

---

## 2. Core Measurement

### 2.1 Observation surprise

For a trajectory $x_{1:T}$ with character prompt $c$ and token type mask $\sigma_t$, the cumulative observation surprise is:

$$S(x_{1:T}, c) = \sum_{t=1}^{T} \sigma_t \cdot \big(-\ln p_\theta(x_t \mid c, x_{<t})\big)$$

This is exactly the CCSM perception loss. In Phase 1, we compute it without any gradient.

### 2.2 Per-token surprise profile

Beyond the scalar $S$, we record the **per-token surprise vector** $s_t = -\sigma_t \ln p_\theta(x_t \mid c, x_{<t})$ for all observation tokens. This supports decomposed analysis: where in a trajectory surprise spikes, whether surprise is driven by a few extreme tokens or is distributed, and temporal structure at different scales (per-move, per-phase, cumulative).

### 2.3 Quality–surprise correlation (with controls)

The primary metric is **Spearman's rank correlation** $\rho$ between trajectory-level quality $Q(x_{1:T})$ and cumulative observation surprise $S(x_{1:T}, c)$.

The prediction is $\rho < 0$: higher quality trajectories should have *lower* observation surprise under an appropriately-conditioned model.

**Critical: this raw correlation is necessary but not sufficient.** Several confounds can produce spurious negative $\rho$:

- **Length confound.** Shorter trajectories have lower cumulative surprise mechanically. If higher-quality trajectories are systematically shorter (e.g. decisive chess wins end sooner), the correlation is artifactual.
- **Lexical frequency confound.** Low surprise can reflect common words rather than strategic content. If high-quality trajectories use more conventional language, the correlation is measuring style, not substance.
- **Format familiarity confound.** In structured domains (chess with FEN), surprise may track the model's familiarity with the notation rather than its understanding of the position.

We address these with:

- **Length-normalised surprise:** $\bar{S} = S / N_{\text{obs}}$ where $N_{\text{obs}} = \sum_t \sigma_t$. Report $\rho$ against both raw and normalised surprise.
- **Residualised surprise:** Regress per-token surprise against unigram token frequency (estimated from the model's unconditional distribution). Report $\rho$ using the residuals — this isolates the component of surprise that is *not* explained by token-level language modelling.
- **Stratified surprise:** Compute surprise separately over (a) all observation tokens, (b) opponent move/action tokens only, (c) state description tokens only. Compare correlations. If the signal is concentrated in opponent actions rather than state descriptions, it's more likely to reflect strategic understanding.

### 2.4 Permutation baseline

For every $\rho$ we report, we also report the **null distribution** obtained by shuffling quality labels across trajectories within each quality level (1000 permutations). This gives a proper significance test and tells us whether observed correlations are above noise.

### 2.5 Counterfactual edits (causal test)

Correlation — even with controls — is associational. To test whether the model responds *causally* to quality changes, we run **within-trajectory counterfactual edits**.

**Procedure:** Take a trajectory. At a single move position, replace the action with either (a) a better move or (b) a worse move (as rated by Stockfish / the environment's quality metric). Propagate the consequences: the opponent responds to the new move, producing a new observation. Measure the change in local surprise at the observation tokens immediately following the edit.

**Prediction:** Replacing a good move with a bad move should *increase* observation surprise (the opponent's response to a blunder is less expected under a competence-conditioned model). Replacing a bad move with a good move should *decrease* it.

**Why this is stronger than correlation:** It holds everything else constant — same trajectory, same prompt, same model — and asks whether a *local* quality change produces a *local* surprise change. This is the closest we can get to a causal claim about the model's priors without intervening on the model itself.

We report:

- **Δ surprise** as a function of **Δ quality** (centipawn change in chess, utility change in negotiation).
- Sign consistency: what fraction of edits produce the predicted sign of Δ surprise.
- Effect size relative to background surprise variance.

---

## 3. Environment 1: Chess

### 3.1 Why chess

Chess is the strongest testbed because of a convergence of favourable properties:

- **Perfect external evaluator.** Stockfish provides centipawn evaluation of every position and move, giving continuous quality scores at move and trajectory level.
- **Deterministic dynamics.** The opponent's response is fully determined by their policy (Stockfish at fixed depth), so observation surprise reflects strategic prediction, not stochasticity.
- **Strong pretraining bias toward quality.** Internet chess data is heavily skewed toward annotated master games, engine analyses, and instructional material. This means the pretraining distribution likely *does* correlate with quality — making chess the domain where we most expect CCSM's assumption to hold. If it fails even here, the framework is in serious trouble.
- **Clean tokenisation.** Moves in algebraic notation are short token sequences. FEN strings are structured and prevalent in pretraining data.
- **Controlled quality variation.** By forcing moves of known Stockfish evaluation, we construct trajectories at any quality level.

### 3.2 Trajectory construction

We do **not** have the model play chess. We construct trajectories externally and evaluate surprise on them. This separates "does the model know what good play looks like?" from "can the model produce good play?"

**Trajectory format.** Each trajectory is a sequence of alternating action and observation tokens:

```
[Character prompt c]
[Observation: initial board state]
[Action: White's move 1]
[Observation: resulting board state + Black's response]
[Action: White's move 2]
[Observation: resulting board state + Black's response]
...
```

The agent plays White. Action tokens are White's moves. Observation tokens are board state descriptions and Black's responses.

**Quality levels.** We construct trajectories at several quality levels by controlling White's move selection:

| Level | White's moves | Expected quality |
|-------|--------------|-----------------|
| **Optimal** | Stockfish top-1 at depth 20 | Highest |
| **Strong** | Stockfish top-3 (sampled) | High |
| **Moderate** | Stockfish top-10 (sampled) | Medium |
| **Weak** | Random legal moves weighted toward Stockfish bottom half | Low |
| **Random** | Uniform random legal moves | Lowest |

Black always plays at a fixed strong level (Stockfish depth 15) to provide a consistent "environment."

**Quality metric $Q$.** For each trajectory:

- **Trajectory-level:** Mean centipawn evaluation of White's positions, or game outcome (+1/0/-1).
- **Move-level:** Centipawn loss of each of White's moves relative to Stockfish top-1.

**Trajectory count.** 100 games per quality level × 5 levels = 500 games. At ~40 moves per game, this gives ~4000 action-observation pairs per level.

### 3.3 Counterfactual edits in chess

At 200 randomly sampled positions across the trajectory set:

- Take the original move and replace it with a move that differs by at least 100 centipawns (in either direction).
- Replay from that position: Black responds via Stockfish, producing a new observation sequence.
- Measure Δ surprise on the *next 3 observation tokens* (Black's response + resulting board state).

This isolates the question: *does the model expect different opponent responses after strong vs weak moves?*

### 3.4 Surprise decomposition in chess

To disentangle "can the model predict Stockfish?" from "does the model understand move quality?", we compute surprise separately over:

- **Black's move tokens** — does the model predict the opponent's response?
- **Board state tokens** — does the model predict the resulting position?
- **All observation tokens** — the full signal.

If $\rho$ is driven primarily by board state tokens, surprise is measuring notation familiarity. If it's driven by Black's move tokens, it's measuring strategic prediction. We expect and need the latter.

### 3.5 Text format

Two options, both tested:

**Option A — FEN + algebraic notation.** Compact, information-dense. Tests formal chess knowledge. Example: `Position after 1.e4 e5 2.Nf3: rnbqkbnr/pppp1ppp/8/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R b KQkq - 1 2`

**Option B — Natural language narration.** Richer context for language priors. Tests strategic narrative knowledge. Example: `White plays knight to f3, attacking Black's e5 pawn. Black responds with knight to c6, defending the pawn and developing a piece.`

We expect Option B to show stronger surprise-quality correlation for weaker models, and Option A to catch up or surpass for stronger models that have internalised FEN semantics.

### 3.6 Character prompts

| Prompt ID | Content | Expected effect |
|-----------|---------|----------------|
| `GM` | "You are a chess grandmaster rated 2700 ELO, playing White. You play principled, strategic chess." | Baseline — should show strong negative $\rho$ |
| `beginner` | "You are a beginner chess player who just learned the rules. You play White." | Inverted or flat $\rho$ — weak play should be *less* surprising |
| `analyst` | "You are a chess commentator analysing a game between two strong players." | Third-person framing — tests whether participant vs observer matters |
| `neutral` | "The following is a chess game in algebraic notation." | No character — measures unconditional chess prior |
| `mismatched` | "You are a world-class poker player watching a chess game." | Irrelevant prompt — should show weak $\rho$ |

The key comparison is `GM` vs `beginner`. If "grandmaster" conditioning makes strong play less surprising and weak play more surprising *relative to* "beginner" conditioning, character prompts are reshaping the preference landscape. If the two produce similar surprise distributions, the conditioning mechanism is too shallow and CCSM's steering is weak.

---

## 4. Environment 2: Resource Trading (Negotiation)

### 4.1 Why negotiation

Negotiation tests priors in a fundamentally different regime from chess:

- **Social reasoning.** Success depends on modelling the other party's preferences, beliefs, and likely responses — a form of theory of mind.
- **Language-native.** Negotiation is *inherently* a language task. The model's language priors are directly relevant, not mediated through a formal notation.
- **Multi-dimensional quality.** Outcomes can be scored on individual gain, joint surplus (Pareto efficiency), fairness, and process coherence. This lets us test whether surprise tracks different *aspects* of quality.
- **Uncertain pretraining bias.** Unlike chess, where training data is biased toward strong play, internet negotiation text includes everything from optimal bargaining to manipulative tactics to confused exchanges. This makes negotiation the harder test — if surprise tracks quality here, it's strong evidence. If it doesn't, we learn where CCSM's prior assumption breaks down.
- **Relevance to CCSM's long-term goals.** Multi-agent social interaction is where CCSM is ultimately meant to be deployed.

### 4.2 Game design

We use a **multi-issue resource trading game** adapted from the Deal-or-No-Deal negotiation task (Lewis et al., 2017).

**Setup.** Two players negotiate to divide a set of items (3 types, varying quantities). Each has a private valuation per item type, and valuations are asymmetric — creating gains from trade. The game proceeds in alternating natural language messages and ends when both accept a proposed split, or after a maximum number of rounds (disagreement = zero payoff for both).

**Concrete instantiation:** 5 books, 3 hats, 2 balls. Player A values books at 4 and hats at 1; Player B values books at 1 and hats at 4. Both value balls at 2. The Pareto-optimal outcome gives each player the items they value most. We use 50 such scenarios with varied item/valuation configurations.

**Why this is verifiable:** Given the private valuations, we can compute: each player's utility (normalised to [0,1]), Pareto efficiency (total utility as fraction of maximum), and distance from the Nash bargaining solution.

### 4.3 Trajectory construction

**Critical design decision: controlled language templates.** Free-form natural language negotiation introduces massive linguistic variance that confounds the surprise signal. A trajectory where Player A says "I'd really like all the books please" vs "gimme the books" differ enormously in token-level surprise but express the same strategy.

We address this with a **two-tier design:**

**Tier 1 — Templated language (primary).** Offers use a fixed syntactic template:

```
"I propose: I get [X books, Y hats, Z balls], you get [A books, B hats, C balls]."
"I accept your proposal."
"I reject your proposal. Counter-offer: I get [X books, Y hats, Z balls], you get [...]."
```

This eliminates linguistic variance and isolates *strategic* surprise from *stylistic* surprise. The quality differences are purely in the numbers and the strategy (concession patterns, Pareto-improving proposals, etc.).

**Tier 2 — Natural language (secondary).** Full natural language negotiation, scripted at each quality level with consistent but varied phrasing. This tests whether the model's priors work in the messier setting, but is interpreted in light of Tier 1 results. If Tier 1 shows strong $\rho$ but Tier 2 doesn't, the priors encode strategic knowledge but it's masked by linguistic noise — which is a tractable problem. If both fail, the priors are genuinely weak.

**Quality levels (Tier 1):**

| Level | Player A's strategy | Expected quality |
|-------|-------------------|-----------------|
| **Optimal** | Nash bargaining — proposes Pareto-improving splits, makes calibrated concessions | Highest |
| **Cooperative-suboptimal** | Proposes equal item splits (fair but not Pareto-optimal since valuations are asymmetric) | Medium-high |
| **Greedy-successful** | Demands most items, opponent happens to concede | High $Q_{\text{individual}}$, low $Q_{\text{pareto}}$ |
| **Aggressive-failed** | Demands everything, reaches disagreement | Zero utility |
| **Incoherent** | Makes contradictory offers, ignores opponent's messages | Lowest |

Player B follows a fixed tit-for-tat concession strategy to provide a consistent environment.

**Quality metrics:**

- **$Q_{\text{individual}}$:** Player A's normalised utility.
- **$Q_{\text{pareto}}$:** Total utility as fraction of Pareto frontier.
- **$Q_{\text{process}}$:** Rule-based coherence score — are offers internally consistent? Does A respond to B's proposals? Are concessions monotonic?

We report surprise correlation against each separately. This tells us whether the model's priors encode *self-interest*, *social efficiency*, *communicative competence*, or some combination.

**Trajectory count.** 50 scenarios × 5 quality levels = 250 games per tier. At ~10 rounds per game, ~2500 observation-action pairs per level.

### 4.4 Counterfactual edits in negotiation

At 100 randomly sampled positions:

- Replace a single offer with a more/less generous one (same template, different numbers).
- Player B responds according to their fixed strategy.
- Measure Δ surprise on B's response.

This tests: *does the model expect different opponent responses to generous vs stingy offers?*

### 4.5 Outcome surprise probe

An additional test specific to negotiation: at the *end* of the trajectory, after the deal is reached, append an "outcome summary" observation token:

```
[Observation: "Deal reached. You scored 8 out of 10 possible points."]
```

Measure the model's surprise at this outcome token as a function of whether the preceding negotiation *should* have led to this score. If the model has priors about negotiation outcomes, it should find a high score surprising after an aggressive-failed trajectory and unsurprising after an optimal one. This directly tests whether the model has internalised the *consequences* of negotiation strategies, not just their linguistic form.

### 4.6 Character prompts

| Prompt ID | Content | Expected effect |
|-----------|---------|----------------|
| `negotiator` | "You are a skilled negotiator. You aim to maximise your own score while reaching mutually beneficial agreements." | Baseline — should correlate with both $Q_{\text{individual}}$ and $Q_{\text{pareto}}$ |
| `cooperative` | "You are a fair-minded person who values equitable outcomes above personal gain." | Should correlate more with $Q_{\text{pareto}}$ than $Q_{\text{individual}}$ |
| `aggressive` | "You are a ruthless negotiator who will accept nothing less than the best possible deal for yourself." | Should correlate with $Q_{\text{individual}}$, possibly anti-correlate with $Q_{\text{pareto}}$ |
| `neutral` | "The following is a negotiation between two people dividing items." | Unconditional prior |
| `mismatched` | "You are a chess grandmaster." | Deliberately irrelevant |

The key test is whether `cooperative` and `aggressive` prompts *differentially* track $Q_{\text{pareto}}$ vs $Q_{\text{individual}}$. If they do, character conditioning shapes the *kind* of behaviour treated as unsurprising, not just overall quality.

---

## 5. Environment 3: Text Gridworld (Navigation)

### 5.1 Why a gridworld

Chess and negotiation sit at opposite ends of a spectrum — formal/structured vs language-rich/social. We need a domain in between to understand where transitions in prior quality happen. A text gridworld provides this: it has *deterministic, verifiable optimal behaviour* (like chess) but is *mediated entirely through natural language descriptions* (like negotiation).

It also has a simple enough structure that we can exhaustively test confounds. If the model fails here, something is wrong with our methodology, not just the model's priors.

### 5.2 Game design

A grid (5×5 to 8×8) with a start position, a goal position, and optional obstacles. The agent receives a text description of its surroundings and issues movement commands. The environment responds with the new surroundings.

**Trajectory format:**

```
[Character prompt c]
[Observation: "You are at position (1,1) in a 5x5 grid. The goal is at (5,5).
 You can see: wall to the north, open path to the east and south."]
[Action: "move east"]
[Observation: "You moved east. You are now at (2,1). You can see: open path
 in all directions."]
[Action: "move south"]
...
[Observation: "You reached the goal at (5,5)!"]
```

**Quality metric $Q$.** Path optimality: $Q = L_{\text{optimal}} / L_{\text{actual}}$, where $L$ is path length. Optimal path computed by BFS/A*. $Q = 1.0$ for optimal, $Q < 1.0$ for suboptimal, $Q = 0$ for failure to reach goal.

**Quality levels:**

| Level | Strategy | Expected quality |
|-------|---------|-----------------|
| **Optimal** | A* shortest path | $Q = 1.0$ |
| **Near-optimal** | Shortest path + 1-2 random detours | $Q \approx 0.8$ |
| **Wandering** | Biased random walk toward goal | $Q \approx 0.4$ |
| **Lost** | Unbiased random walk | $Q \approx 0.1$ |
| **Adversarial** | Deliberately moves away from goal | $Q \approx 0.0$ |

**Trajectory count.** 100 grids × 5 quality levels = 500 trajectories.

### 5.3 Why this helps

The gridworld isolates spatial reasoning from both formal notation (unlike chess) and social dynamics (unlike negotiation). The observation tokens are pure natural language descriptions with minimal stylistic variation, so token-level confounds are easier to control. And the quality metric is a single scalar (path optimality) with no ambiguity.

If surprise tracks quality here, we know the model has usable *planning/reasoning* priors in addition to any domain-specific knowledge. If it fails here but succeeds at chess, the model has chess-specific priors but not general reasoning priors — which matters for predicting where CCSM will work.

### 5.4 Character prompts

| Prompt ID | Content | Expected effect |
|-----------|---------|----------------|
| `navigator` | "You are an expert navigator finding the shortest path through a grid." | Baseline — should show negative $\rho$ |
| `lost` | "You are lost and confused, wandering aimlessly through a grid." | Inverted — inefficient paths should be less surprising |
| `neutral` | "The following is a description of movement through a grid." | Unconditional prior |

---

## 6. Cross-Environment Analysis

### 6.1 Prior quality as a function of model scale

Run all experiments across at least 3 model sizes (e.g. 1.5B, 7B, 70B). Plot $\rho$ vs model size per environment. The prediction is monotonically improving $\rho$, giving a scaling argument: better models → better priors → CCSM works better.

### 6.2 Prompt robustness

For each environment, compute the variance of $\rho$ across the "reasonable" prompts (excluding mismatched). Low variance means the prior is robust to prompt phrasing. High variance means CCSM is fragile and prompt engineering is load-bearing. Both are important findings — the latter suggests a need for prompt optimisation or SFT warmup in Phase 2.

### 6.3 Domain transfer of prior quality

Does a model with good chess priors also have good negotiation priors? Correlate per-model $\rho_{\text{chess}}$ with $\rho_{\text{negotiation}}$ with $\rho_{\text{gridworld}}$ across the model size sweep. If they're correlated, "prior quality" is a general model property. If not, CCSM's effectiveness is domain-dependent.

### 6.4 Failure mode analysis

Identify trajectories where surprise and quality *disagree* — high quality but high surprise, or low quality but low surprise. Examine these manually. Common patterns to look for:

- **Style-over-substance:** low surprise on low-quality trajectories that use fluent, conventional language.
- **Notation confusion:** high surprise on high-quality trajectories because of unfamiliar formatting.
- **History dependence:** surprise diverges from quality late in trajectories because the model loses track of context.

These failure modes predict where CCSM training will struggle, and may suggest prompt engineering strategies or auxiliary losses for Phase 2.

### 6.5 What the causal tests add

If the counterfactual edit results are strong (high sign consistency, effect size > 1σ), this substantially upgrades the correlational findings — we know the model isn't just tracking surface features but responding to strategic content. If causal tests are weak despite strong correlations, we know the correlations are fragile and CCSM's learning signal will be noisy.

---

## 7. Code Architecture

### 7.1 Overview

The Phase 1 codebase is a standalone evaluation framework — no training, no gradient computation. It's structured as a pipeline:

```
TrajectoryGenerator → SurpriseEvaluator → CorrelationAnalyser → Reporter
```

Each stage is modular so we can add environments and models without restructuring.

### 7.2 Module breakdown

```
ccsm_eval/
├── trajectories/
│   ├── base.py              # Trajectory, Token, TrajectoryBatch dataclasses
│   ├── chess/
│   │   ├── generator.py     # ChessTrajectoryGenerator — python-chess + Stockfish
│   │   ├── formatter.py     # ChessFormatter — FEN vs natural language
│   │   ├── quality.py       # ChessQualityScorer — Stockfish eval, centipawn loss
│   │   └── counterfactual.py # ChessCounterfactualEditor — move replacement + replay
│   ├── negotiation/
│   │   ├── generator.py     # NegotiationTrajectoryGenerator — scripted strategies
│   │   ├── formatter.py     # NegotiationFormatter — templated vs natural language
│   │   ├── strategies.py    # Strategy implementations (optimal, greedy, cooperative, etc.)
│   │   ├── quality.py       # NegotiationQualityScorer — utility, Pareto, process
│   │   ├── counterfactual.py # NegotiationCounterfactualEditor — offer replacement
│   │   └── outcome_probe.py # OutcomeSurpriseProbe — appends outcome summary tokens
│   └── gridworld/
│       ├── generator.py     # GridworldTrajectoryGenerator — grid + pathfinding
│       ├── formatter.py     # GridworldFormatter — text descriptions
│       ├── quality.py       # GridworldQualityScorer — path optimality
│       └── counterfactual.py # GridworldCounterfactualEditor — step replacement
│
├── evaluation/
│   ├── surprise.py          # SurpriseEvaluator — forward pass, per-token log-probs
│   ├── model_loader.py      # Loads pretrained models (HuggingFace / vLLM)
│   └── batching.py          # Efficient batching of trajectories for forward passes
│
├── analysis/
│   ├── correlation.py       # Spearman ρ, bootstrap CIs, permutation null
│   ├── confounds.py         # Length normalisation, residualised surprise, stratification
│   ├── counterfactual.py    # Δ surprise analysis, sign consistency, effect sizes
│   ├── prompt_sensitivity.py # Cross-prompt comparison
│   ├── scaling.py           # Cross-model-size analysis
│   └── failure_modes.py     # Disagreement case extraction and classification
│
├── reporting/
│   ├── figures.py           # Matplotlib/seaborn figures
│   └── tables.py            # LaTeX table generation
│
├── configs/
│   ├── chess_eval.yaml
│   ├── negotiation_eval.yaml
│   ├── gridworld_eval.yaml
│   └── models.yaml
│
└── run_eval.py              # Main entry point — runs full pipeline from config
```

### 7.3 Key dataclasses

```python
@dataclass
class Token:
    text: str
    token_ids: list[int]          # After tokenisation (model-specific)
    is_observation: bool           # σ_t: True for observation, False for action
    semantic_type: str             # e.g. "opponent_move", "board_state", "offer",
                                   #      "narration" — for stratified analysis
    position: int                  # Position in the full trajectory

@dataclass
class Trajectory:
    tokens: list[Token]
    character_prompt: str
    quality_scores: dict[str, float]  # e.g. {"centipawn": 0.85, "outcome": 1.0}
    metadata: dict                     # Environment-specific

@dataclass
class CounterfactualEdit:
    trajectory_id: str
    edit_position: int                # Token position of the replaced action
    original_action: str
    replacement_action: str
    quality_delta: float              # Δ quality (positive = improvement)
    original_tokens: list[Token]      # Observation tokens after original action
    replacement_tokens: list[Token]   # Observation tokens after replacement action

@dataclass
class SurpriseResult:
    trajectory_id: str
    prompt_id: str
    model_id: str
    per_token_surprise: list[float]         # s_t for each observation token
    per_token_semantic_type: list[str]       # Semantic type of each obs token
    per_token_unigram_logprob: list[float]   # For residualisation
    cumulative_surprise: float               # S(x_{1:T}, c)
    normalised_surprise: float               # S / N_obs
    quality_scores: dict[str, float]
```

### 7.4 Key interfaces

```python
class TrajectoryGenerator(ABC):
    """Generates trajectories of controlled quality for a specific environment."""

    @abstractmethod
    def generate(self, quality_level: str, n_trajectories: int,
                 seed: int) -> list[Trajectory]:
        ...

    @abstractmethod
    def quality_levels(self) -> list[str]:
        ...

class CounterfactualEditor(ABC):
    """Generates counterfactual edits for causal testing."""

    @abstractmethod
    def edit(self, trajectory: Trajectory, position: int,
             direction: str) -> CounterfactualEdit:
        """Replace action at position with a better ('up') or worse ('down')
        alternative. Returns the edit with propagated consequences."""
        ...

    @abstractmethod
    def sample_edit_positions(self, trajectory: Trajectory,
                              n_edits: int, seed: int) -> list[int]:
        """Select action positions suitable for counterfactual editing."""
        ...

class TrajectoryFormatter(ABC):
    """Converts a trajectory into a text sequence with token type masks."""

    @abstractmethod
    def format(self, trajectory: Trajectory,
               character_prompt: str) -> tuple[str, list[bool], list[str]]:
        """Returns (full_text, observation_mask, semantic_type_per_char)."""
        ...

class QualityScorer(ABC):
    """Computes quality metrics for a trajectory."""

    @abstractmethod
    def score(self, trajectory: Trajectory) -> dict[str, float]:
        ...

class SurpriseEvaluator:
    """Runs model forward passes and extracts per-token surprise."""

    def __init__(self, model_name: str, device: str = "cuda"):
        ...

    def evaluate(self, text: str, observation_mask: list[bool],
                 semantic_types: list[str],
                 character_prompt: str) -> SurpriseResult:
        ...

    def evaluate_batch(self, batch: list[tuple[str, list[bool],
                       list[str], str]]) -> list[SurpriseResult]:
        ...

    def evaluate_counterfactual(self, edit: CounterfactualEdit,
                                 character_prompt: str
                                 ) -> tuple[float, float]:
        """Returns (surprise_original, surprise_replacement) over the
        observation tokens following the edit point."""
        ...
```

### 7.5 Dependencies

| Dependency | Purpose |
|-----------|---------|
| `python-chess` | Legal move generation, board state management, FEN |
| `stockfish` (binary + Python wrapper) | Move evaluation, centipawn scores |
| `transformers` | Model loading and tokenisation |
| `vllm` | Fast batched inference (optional, 5-10x speedup) |
| `torch` | Tensor operations, forward passes |
| `scipy.stats` | Spearman correlation, permutation tests |
| `statsmodels` | OLS regression for surprise residualisation |
| `matplotlib` / `seaborn` | Figures |
| `pyyaml` | Configuration |
| `pandas` | Results aggregation |

### 7.6 Compute requirements

Phase 1 is inference-only. Per model size (estimates for full pipeline: 1250 trajectories across 3 environments × 5 prompts × counterfactual edits):

| Model | VRAM (bf16) | Trajectories/hour (est.) | Total time |
|-------|------------|-------------------------|-----------|
| 1.5B  | ~3 GB      | ~500                    | ~15 hours  |
| 7B    | ~14 GB     | ~150                    | ~45 hours  |
| 70B   | ~140 GB (2× GH100) | ~30              | ~210 hours |

**Fast iteration strategy:** Run the 7B model on chess + gridworld with the `GM`/`navigator` and `beginner`/`lost` prompts only. This is ~200 trajectories × 2 prompts = ~3 hours, and gives you the core signal (does surprise track quality? does prompt conditioning work?) before committing to the full sweep.

### 7.7 Configuration example

```yaml
experiment:
  name: "chess_prior_validation"
  seed: 42

environment:
  type: chess
  trajectory_counts:
    optimal: 100
    strong: 100
    moderate: 100
    weak: 100
    random: 100
  opponent_depth: 15
  max_moves: 60
  format: both                # FEN and natural language
  counterfactual:
    n_edits_per_trajectory: 2
    min_quality_delta_cp: 100   # Minimum centipawn change for edit

prompts:
  GM:
    text: "You are a chess grandmaster rated 2700 ELO, playing White."
  beginner:
    text: "You are a beginner chess player who just learned the rules."
  analyst:
    text: "You are a chess commentator analysing a game between two strong players."
  neutral:
    text: "The following is a chess game in algebraic notation."
  mismatched:
    text: "You are a world-class poker player watching a chess game."

models:
  - name: "qwen2.5-1.5b"
    hf_path: "Qwen/Qwen2.5-1.5B"
    gpus: 1
  - name: "qwen2.5-7b"
    hf_path: "Qwen/Qwen2.5-7B"
    gpus: 1
  - name: "qwen2.5-72b"
    hf_path: "Qwen/Qwen2.5-72B"
    gpus: 4

analysis:
  correlation_method: spearman
  bootstrap_n: 1000
  confidence_level: 0.95
  residualise_surprise: true
  stratify_by_semantic_type: true
  permutation_null_n: 1000
```

---

## 8. Expected Outcomes and Decision Criteria

### 8.1 Strong positive result

Consistent negative $\rho$ (< -0.4) for aligned prompts across all three environments at 7B+, with: clear prompt-dependent inversion (`GM` vs `beginner`); strong causal signal (counterfactual sign consistency > 75%); $\rho$ survives residualisation and stratification; improving $\rho$ with model scale. **Proceed to Phase 2 with confidence.**

### 8.2 Moderate positive result

$\rho$ in the range [-0.2, -0.4] with some prompt sensitivity but noisy, or strong chess + weak negotiation/gridworld. Causal tests show right direction but noisy. **Proceed to Phase 2 in strong domains. Investigate whether the perception loss during CCSM training sharpens weak priors. Consider SFT warmup (§6.3 of framework spec) for weak domains.**

### 8.3 Correlation without causation

Strong raw $\rho$ but: signal disappears after residualisation (measuring token frequency, not strategy); or counterfactual edits show no local response; or signal concentrated in state description tokens, not opponent actions. **Do not proceed to Phase 2. The correlation is artifactual.** Investigate whether better text formatting, tokenisation, or prompt design can recover a genuine signal.

### 8.4 Negative result

$\rho \approx 0$ or positive, no prompt sensitivity, causal tests fail. Pretrained priors are too weak for CCSM without modification. **Do not proceed to Phase 2 as-is.** Options: use a stronger base model; add SFT warmup on expert trajectories; restrict to narrow domains; reconsider the framework.

### 8.5 Differential result across environments

The most likely and most informative outcome. Expected pattern: strong chess, moderate gridworld, weak negotiation. This tells us that pretraining data quality determines prior quality (as the framework predicts), and that CCSM needs help in domains where pretraining data doesn't select for competence. **Proceed to Phase 2 in chess/gridworld. For negotiation, test whether SFT warmup on a small set of optimal negotiations bootstraps usable priors, then re-run Phase 1 to confirm.**