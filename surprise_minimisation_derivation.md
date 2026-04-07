## A derivation from the Action Perception Divergence framework

---

## 1. Motivation

We want to train a single autoregressive language model to act in an environment where both the agent's actions and the environment's observations are tokens in a shared context window. The model should learn to:

1. **Perceive** — accurately predict what the environment will produce next.
2. **Act** — choose actions that lead to trajectories the model finds unsurprising.

We want a single, unified loss function that covers both cases, with no extrinsic reward signal. Instead of specifying reward, we specify a **character prompt** $c$ — a natural language description of the kind of agent the model should be (e.g. "you are a chess grandmaster", "you are a helpful kitchen assistant"). The model's own conditional distribution $p_\theta(\cdot \mid c)$ then defines what counts as surprising: trajectories that are unlikely given the character are surprising, and the agent is trained to avoid them.

This follows from a specific instantiation of the **Action Perception Divergence** (APD) framework (Hafner et al., 2022), which shows that action and perception can be unified as joint KL minimisation between the actual distribution of a system and a target distribution expressing the agent's preferences.

---

## 2. Setup

### 2.1 Token sequence

A trajectory is a sequence of tokens:

$$x_{1:T} = (x_1, x_2, \ldots, x_T)$$

Each token has a known binary type $\sigma_t \in \{0, 1\}$, where $\sigma_t = 1$ indicates an observation token (produced by the environment) and $\sigma_t = 0$ indicates an action token (produced by the model). This type mask is known at training time and fixed for a given trajectory.

### 2.2 Character prompt

A fixed prompt $c$ is prepended to every trajectory. This prompt is not optimised — it is a specification of the agent's identity, goals, and dispositions, written in natural language. The prompt serves as the sole mechanism for specifying desired behaviour.

The prompt works by _biasing the model's predictive distribution_, which indirectly shapes behaviour through the surprise minimisation objective. This mechanism relies on the pretrained model already encoding relevant structure: if the model's conditional distribution $p_\theta(\cdot \mid c)$ assigns meaningfully different probabilities to trajectories depending on $c$, then different prompts will induce different behaviours. The strength of this mechanism is bounded by the quality of the pretrained model's representations.

**Example:** If $c$ = "You are a chess grandmaster playing white", then under $p_\theta(\cdot \mid c)$, trajectories where the agent plays strong moves and wins should be higher-probability (lower surprise) than trajectories where the agent blunders and loses — _provided the pretrained model has learned enough about chess to make this distinction_. If the model has weak chess priors, the prompt will provide weak behavioural shaping, and the agent will converge to a self-consistent equilibrium that may not correspond to strong play.

### 2.3 Single autoregressive model

A single model with parameters $\theta$ defines:

$$p_\theta(x_t \mid c, x_{<t})$$

for all positions $t$. This distribution serves simultaneously as the **world model** (predicting observation tokens), the **policy** (sampling action tokens), and the **target** (defining what trajectories are preferred).

---

## 3. Actual and Target Distributions

### 3.1 Actual distribution

The actual distribution describes the generative process that produces the trajectory. Using the binary mask $\sigma_t$, we write the per-token conditional as a single expression — since $\sigma_t$ is always exactly 0 or 1, exactly one term is active at each position:

$$p_\theta^{\text{actual}}(x_{1:T} \mid c) = \prod_{t=1}^{T} \Big[\sigma_t \, p_{\text{env}}(x_t \mid x_{<t}) + (1 - \sigma_t) \, p_\theta(x_t \mid c, x_{<t})\Big]$$

This is equivalent to the standard trajectory distribution induced by a policy interacting with an environment, written at token resolution. Note that $p_{\text{env}}$ does not condition on $c$ — the environment does not know or care about the agent's character prompt. The environment's dynamics are fixed and unknown.

### 3.2 Target distribution

The target distribution is the model's own unconditional (over token type) prediction of the full trajectory, given the character prompt:

$$\tau_\theta(x_{1:T} \mid c) = \prod_{t=1}^{T} p_\theta(x_t \mid c, x_{<t})$$

The target makes no distinction between observation and action tokens. It represents the model's beliefs about what a _complete trajectory should look like_, given that the agent has character $c$. This is the "models as preferences" insight from APD (Section 2.3 of Hafner et al.): the model class itself defines the agent's preferred input distribution.

### 3.3 The self-referential structure

A notable feature of this setup is that both sides of the KL divergence depend on $\theta$:

$$\mathcal{L}(\theta) = \text{KL}\big[p_\theta^{\text{actual}}(x_{1:T} \mid c)  \; \Vert \; \tau_\theta(x_{1:T} \mid c)\big]$$

This is not a standard variational inference problem where one minimises KL to a fixed target. The target moves with $\theta$, which has structural consequences.

**Gradient structure.** The gradient $\nabla_\theta \mathcal{L}$ receives contributions from both $p_\theta^{\text{actual}}$ (through the policy at action tokens) and $\tau_\theta$ (through the model's log-probabilities at observation tokens). Improving the model's predictions (which changes $\tau_\theta$) simultaneously changes what counts as surprising, so the target the agent is chasing shifts as it learns. This means the agent's preferences become more refined as its understanding of the world improves.

**Fixed points.** The KL reaches zero when $p_\theta^{\text{actual}} = \tau_\theta$, which occurs when $p_\theta(x_t \mid c, x_{<t}) = p_{\text{env}}(x_t \mid x_{<t})$ at all observation positions — i.e., the model perfectly predicts the environment along trajectories it actually visits. This is a self-consistency condition, not a correctness condition: the agent need not predict the environment everywhere, only along trajectories its own policy induces. Multiple self-consistent equilibria may exist, and the agent will converge to one that is reachable from its initialisation. This is discussed further in Section 8.

---

## 4. Derivation of the Loss

### 4.1 The joint divergence

The objective is:

$$\mathcal{L}(\theta) = \text{KL}\Big[p_\theta^{\text{actual}}(x_{1:T} \mid c) \; \Vert \; \tau_\theta(x_{1:T} \mid c)\Big]$$

Expanding the KL and substituting the autoregressive factorisations:

$$\mathcal{L}(\theta) = \mathbb{E}_{p_\theta^{\text{actual}}}\left[\sum_{t=1}^{T} \ln \frac{\sigma_t \, p_{\text{env}}(x_t \mid x_{<t}) + (1 - \sigma_t) \, p_\theta(x_t \mid c\, x_{<t})}{p_\theta(x_t \mid c\, x_{<t})}\right]$$

### 4.2 Collapsing via the binary mask

Although the expression inside the logarithm looks like a mixture, $\sigma_t$ is binary — so exactly one term is active, and the "mixture" collapses to a single component before the log is applied. The log-ratio at each position therefore simplifies:

$$\ln \frac{\sigma_t \, p_{\text{env}} + (1 - \sigma_t) \, p_\theta}{p_\theta} = \sigma_t \Big[\ln p_{\text{env}}(x_t \mid x_{<t}) - \ln p_\theta(x_t \mid c, x_{<t})\Big]$$

This is not an algebraic identity for continuous $\sigma_t$ (the log of a sum is not the sum of logs). It holds by case analysis on the binary mask:

**$\sigma_t = 1$ (observation token):**

$$\text{LHS} = \ln \frac{1 \cdot p_{\text{env}} + 0 \cdot p_\theta}{p_\theta} = \ln \frac{p_{\text{env}}}{p_\theta} = \ln p_{\text{env}} - \ln p_\theta$$

$$\text{RHS} = 1 \cdot \Big[\ln p_{\text{env}} - \ln p_\theta\Big] = \ln p_{\text{env}} - \ln p_\theta \quad \checkmark$$

**$\sigma_t = 0$ (action token):**

$$\text{LHS} = \ln \frac{0 \cdot p_{\text{env}} + 1 \cdot p_\theta}{p_\theta} = \ln \frac{p_\theta}{p_\theta} = 0$$

$$\text{RHS} = 0 \cdot \Big[\ln p_{\text{env}} - \ln p_\theta\Big] = 0 \quad \checkmark$$

Substituting back, the exact loss is:

$$\mathcal{L}(\theta) = \mathbb{E}_{p_\theta^{\text{actual}}}\left[\sum_{t=1}^{T} \sigma_t \Big(\ln p_{\text{env}}(x_t \mid x_{<t}) - \ln p_\theta(x_t \mid c, x_{<t})\Big)\right]$$

---

## 5. Gradient Analysis

We now compute $\nabla_\theta \mathcal{L}$ in a single pass. This analysis simultaneously derives the observation (perception) gradient, derives the action (policy) gradient, identifies the intractable environment entropy terms and justifies their removal, and produces the practical training signal.

### 5.1 Preliminary: the score function identity

Because the expectation in $\mathcal{L}$ is over $p_\theta^{\text{actual}}$, which depends on $\theta$ through the policy at action tokens, we cannot simply differentiate "inside the expectation." Changing $\theta$ changes both the value of the integrand for a fixed sample _and_ which samples are likely to occur. The score function identity captures both effects.

For any distribution $p_\theta$ and function $f$:

$$\nabla_\theta \, \mathbb{E}_{p_\theta}\big[f(x)\big] = \mathbb{E}_{p_\theta}\Big[\nabla_\theta f(x) + f(x) \cdot \nabla_\theta \ln p_\theta(x)\Big]$$

**Proof.** Apply the product rule to $\nabla_\theta \int f(x) \, p_\theta(x) \, dx$:

$$= \int \nabla_\theta f(x) \, p_\theta(x) \, dx + \int f(x) \, \nabla_\theta p_\theta(x) \, dx$$

$$= \mathbb{E}_{p_\theta}[\nabla_\theta f(x)] + \int f(x) \, p_\theta(x) \, \frac{\nabla_\theta p_\theta(x)}{p_\theta(x)} \, dx = \mathbb{E}_{p_\theta}\Big[\nabla_\theta f(x) + f(x) \cdot \nabla_\theta \ln p_\theta(x)\Big] \quad \square$$

The **direct term** $\nabla_\theta f(x)$ captures how $\theta$ changes $f$ for a fixed sample. The **score function term** $f(x) \cdot \nabla_\theta \ln p_\theta(x)$ captures how $\theta$ changes which samples occur. When $f$ does not depend on $\theta$, only the score function term survives.

### 5.2 Preliminary: the baseline lemma

This lemma is the key tool for determining which terms in the return carry gradient signal for a given action.

**Lemma.** For any function $b(x_{<t})$ that depends only on the trajectory before the action at time $t$:

$$\mathbb{E}_{p_\theta^{\text{actual}}(x_{1:T})}\Big[\nabla_\theta \ln p_\theta(x_t \mid c, x_{<t}) \cdot b(x_{<t})\Big] = 0$$

**Proof.** Factor the expectation into past and present:

$$= \mathbb{E}_{p_\theta^{\text{actual}}(x_{<t})}\left[b(x_{<t}) \cdot \underbrace{\mathbb{E}_{p_\theta(x_t \mid c, x_{<t})}\Big[\nabla_\theta \ln p_\theta(x_t \mid c, x_{<t})\Big]}_{\text{inner expectation}}\right]$$

The inner expectation is:

$$\sum_{x_t} p_\theta(x_t \mid c, x_{<t}) \cdot \frac{\nabla_\theta \, p_\theta(x_t \mid c, x_{<t})}{p_\theta(x_t \mid c, x_{<t})} = \nabla_\theta \sum_{x_t} p_\theta(x_t \mid c, x_{<t}) = \nabla_\theta \, 1 = 0$$

Since the inner expectation is zero for every value of $x_{<t}$, the outer expectation is zero regardless of $b$. $\square$

The intuition: the score $\nabla_\theta \ln p_\theta(x_t \mid \cdot)$ has zero mean under its own distribution. Any quantity that doesn't depend on $x_t$ factors out of the inner expectation, so it multiplies zero. Past-dependent terms shift all action probabilities equally and therefore carry no gradient signal for the action at $t$.

### 5.3 The score function of the actual distribution

The score function of $p_\theta^{\text{actual}}$ decomposes autoregressively:

$$\nabla_\theta \ln p_\theta^{\text{actual}}(x_{1:T}) = \sum_{t=1}^{T} \nabla_\theta \ln \Big[\sigma_t \, p_{\text{env}}(x_t \mid x_{<t}) + (1 - \sigma_t) \, p_\theta(x_t \mid c, x_{<t})\Big]$$

At observation tokens ($\sigma_t = 1$): $\nabla_\theta \ln p_{\text{env}}(x_t \mid x_{<t}) = 0$. At action tokens ($\sigma_t = 0$): $\nabla_\theta \ln p_\theta(x_t \mid c, x_{<t})$.

Therefore:

$$\nabla_\theta \ln p_\theta^{\text{actual}}(x_{1:T}) = \sum_{t:\,\sigma_t = 0} \nabla_\theta \ln p_\theta(x_t \mid c, x_{<t})$$

Only action tokens contribute to the score function of the trajectory distribution.

### 5.4 Full gradient of the loss

Define for compactness: $E_t \equiv \ln p_{\text{env}}(x_t \mid x_{<t})$ and $M_t \equiv \ln p_\theta(x_t \mid c, x_{<t})$. The loss is:

$$\mathcal{L}(\theta) = \mathbb{E}_{p_\theta^{\text{actual}}}\left[\sum_{t=1}^{T} \sigma_t (E_t - M_t)\right]$$

Applying the score function identity (Section 5.1):

$$\nabla_\theta \mathcal{L} = \mathbb{E}_{p_\theta^{\text{actual}}}\Bigg[\underbrace{\sum_t \sigma_t \nabla_\theta (E_t - M_t)}_{\text{direct}} + \underbrace{\left(\sum_t \sigma_t(E_t - M_t)\right) \cdot \sum_{t':\,\sigma_{t'}=0} \nabla_\theta \ln p_\theta(x_{t'} \mid c, x_{<t'})}_{\text{score function}}\Bigg]$$

#### 5.4.1 Direct term

$$\sum_t \sigma_t \nabla_\theta(E_t - M_t) = \sum_t \sigma_t \Big(\underbrace{\nabla_\theta E_t}_{=\,0} - \nabla_\theta M_t\Big) = -\sum_t \sigma_t \nabla_\theta \ln p_\theta(x_t \mid c\, x_{<t})$$

$$\boxed{\text{Direct gradient} = -\sum_{t:\,\sigma_t = 1} \nabla_\theta \ln p_\theta(x_t \mid c\, x_{<t})}$$

This is **next-token prediction on observation tokens** — the standard cross-entropy loss. The model learns to predict what the environment will produce. This is the perception pathway.

#### 5.4.2 Score function term

Rearranging by grouping around each action token $t'$:

$$\sum_{t':\,\sigma_{t'}=0} \nabla_\theta \ln p_\theta(x_{t'} \mid c, x_{<t'}) \cdot \underbrace{\sum_{t} \sigma_t (E_t - M_t)}_{R}$$

The total trajectory return $R$ splits into terms before and after each action $t'$:

$$R = \underbrace{\sum_{s \leq t'} \sigma_s(E_s - M_s)}_{R_{t'}^{\text{past}}} + \underbrace{\sum_{s > t'} \sigma_s(E_s - M_s)}_{R_{t'}^{\text{future}}}$$

**Past terms.** $R_{t'}^{\text{past}}$ depends only on $x_{\leq t'}$, which is determined before the action at $t'$. By the baseline lemma (Section 5.2):

$$\mathbb{E}_{p_\theta^{\text{actual}}}\Big[\nabla_\theta \ln p_\theta(x_{t'} \mid c, x_{<t'}) \cdot R_{t'}^{\text{past}}\Big] = 0$$

**Future terms.** $R_{t'}^{\text{future}}$ depends on the trajectory after $t'$, which is influenced by the action at $t'$. These carry real gradient signal. Expanding:

$$R_{t'}^{\text{future}} = \underbrace{\sum_{s > t'} \sigma_s E_s}_{\text{future env entropy}} + \underbrace{\sum_{s > t'} \sigma_s (-M_s)}_{\text{future model surprise } \equiv, G_{t'}}$$

**$G_{t'}$ (model surprise)** is computable — it is the total surprise the model assigns to future observation tokens:

$$G_{t'} = \sum_{s > t'} \sigma_s \big(-\ln p_\theta(x_s \mid c, x_{<s})\big)$$

**The environment entropy terms** $\sum_{s > t'} \sigma_s E_s$ are **not computable** — we never observe $p_{\text{env}}$, only the sampled token. They also carry real gradient signal: different actions route the trajectory through regions of different environment stochasticity, so they are not baselines. See Section 5.5 for the full analysis of this approximation.

The computable part of the score function gradient for each action token $t'$ is:

$$\boxed{\text{Score function gradient at } t' = \nabla_\theta \ln p_\theta(x_{t'} \mid c, x_{<t'}) \cdot (-G_{t'})}$$

This is the **REINFORCE estimator** with intrinsic return $-G_{t'}$. It increases the probability of actions whose downstream observations were predictable (low $G_{t'}$), and decreases the probability of actions whose downstream observations were surprising (high $G_{t'}$). The quality of an action is measured by the predictability of the observations it causes — because actions influence future observations through the environment, future surprise is attributed to the current action via the score function.

### 5.5 The tractability approximation

The full score function return for action $t'$ is $R_{t'}^{\text{future}} = G_{t'} + \sum_{s > t'} \sigma_s E_s$. We approximate it with $G_{t'}$, dropping the environment entropy terms.

**What is dropped.** The terms $\sum_{s > t'} \sigma_s \ln p_{\text{env}}(x_s \mid x_{<s})$ measure how the agent's actions affect the _environment's own predictability_ — whether the agent routes itself into high-entropy or low-entropy regions.

**Why it is biased (under standard assumptions).** Under the standard assumption that the environment has irreducible stochasticity, the full objective rewards the agent both for seeking regions where its model is accurate _and_ for seeking regions where the environment is intrinsically predictable. The simplified objective retains only the first signal.

However, the interpretation of this bias depends on one's epistemological commitments. If one adopts a thoroughgoing fallibilist epistemology — rejecting aleatoric stochasticity and treating all unpredictability as epistemic, reflecting the limits of the model rather than irreducible features of the world — then the distinction between "genuinely unpredictable region" and "region my model hasn't learned" collapses. There _are_ no genuinely unpredictable regions, only regions where the model's explanatory depth is insufficient. Under this view, $\ln p_{\text{env}}(x_s \mid x_{<s})$ does not measure irreducible environment entropy; it measures the predictive capacity of a hypothetical better model. Optimising for model surprise alone is then not an approximation but the principled choice: all "environment entropy" is, in principle, reducible by a better model.

We adopt this epistemic stance. The $\approx$ in the simplified loss (Section 5.6) is therefore not an apology for an approximation but a reflection of the fact that our _current_ model is finite and imperfect. The gap between the full and simplified objectives shrinks as the model improves — and under fallibilism, this improvement is unbounded in principle.

**Why we can live with it.** Three reasons:

1. _Intractability_: We cannot compute $p_{\text{env}}(x_t \mid x_{<t})$. The terms are inaccessible regardless.
    
2. _Convergence_: The perception loss continuously improves the world model. As $p_\theta \to p_{\text{env}}$ at observation positions, the gap $E_s - M_s \to 0$, so the dropped terms shrink. The bias vanishes in the limit of a perfect world model.
    
3. _The value function absorbs the expected value_: We use a learned baseline $V_\phi(x_{\leq t'}, c)$ to form the advantage $A_{t'} = G_{t'} - V_\phi(x_{\leq t'}, c)$. To the extent that the expected future environment entropy is a predictable function of the current state, $V_\phi$ absorbs it alongside the expected model surprise. This does not eliminate the bias — the gradient of the dropped terms is not zero — but it means the _variance_ introduced by their absence is reduced.
    

More precisely: the bias in the gradient is:

$$\text{bias} = \mathbb{E}_{p_\theta^{\text{actual}}}\Big[\nabla_\theta \ln p_\theta(x_{t'} \mid c, x_{<t'}) \cdot \sum_{s > t'} \sigma_s E_s\Big]$$

This is nonzero whenever the agent's actions systematically affect future environment stochasticity. It vanishes when either (a) $p_\theta = p_{\text{env}}$ (perfect world model), or (b) the environment is homoskedastic (its conditional entropy doesn't vary across reachable trajectories).

### 5.6 Summary: the simplified loss

Combining the direct term (Section 5.4.1) and the approximate score function term (Section 5.4.2), the practical loss is:

$$\boxed{\mathcal{L}(\theta) \approx \mathbb{E}_{p_\theta^{\text{actual}}}\left[-\sum_{t=1}^{T} \sigma_t \ln p_\theta(x_t \mid c, x_{<t})\right]}$$

At **observation tokens** ($\sigma_t = 1$): the gradient is direct — standard next-token prediction.

At **action tokens** ($\sigma_t = 0$): the $\sigma_t$ mask zeros out the direct loss, but actions influence future observations. The gradient flows via REINFORCE, with intrinsic return $r_{t'}^{\text{intrinsic}} = -G_{t'} = \sum_{s > t'} \sigma_s \ln p_\theta(x_s \mid c, x_{<s})$.

In practice, implementations use the REINFORCE pathway for action tokens, treating sampled tokens as fixed context.

**Why this works for directed behaviour.** Consider the chess grandmaster prompt. After the agent makes a weak move, the opponent responds with a strong counter-move. Under the grandmaster-conditioned model, strong opponent counter-moves following weak agent moves are _unlikely_ (grandmasters don't usually face such positions), so $\ln p_\theta(x_s \mid c, x_{<s})$ is low for those observation tokens. The intrinsic return is therefore low, and the policy gradient pushes the agent away from the weak move.

---

## 6. Entropy Regularisation and Variance Reduction

### 6.1 Entropy regularisation

The KL expansion includes an entropy bonus for action tokens (from the $H[p_\theta^{\text{actual}}]$ term). We include it explicitly with a temperature parameter $\beta$ to prevent policy collapse:

$$\mathcal{L}(\theta) = \mathbb{E}_{p_\theta^{\text{actual}}}\left[-\sum_{t=1}^{T} \sigma_t \ln p_\theta(x_t \mid c, x_{<t}) + \beta \sum_{t=1}^{T} (1 - \sigma_t) \, H\big[p_\theta(\cdot \mid c, x_{<t})\big]\right]$$

The $(1 - \sigma_t)$ mask ensures the entropy bonus applies only at action tokens.

### 6.2 Variance reduction

**Baseline subtraction.** A learned value function $V_\phi(x_{\leq t}, c)$ estimates the expected future observation surprise. The advantage $A_t = G_t - V_\phi(x_{\leq t}, c)$ centres the return, reducing variance. As discussed in Section 5.5, $V_\phi$ also absorbs the predictable component of the dropped environment entropy terms.

**Temporal discounting.** A discount factor $\gamma \in (0, 1]$ reduces variance at the cost of bias:

$$G_t^\gamma = \sum_{s > t} \sigma_s \, \gamma^{(s - t)} \big(-\ln p_\theta(x_s \mid c, x_{<s})\big)$$

**Return normalisation.** Normalising $G_t$ to zero mean and unit variance within each batch helps with the high variance across token positions (function words vs. content words).

---

## 7. Implementation

### 7.1 Two-loss decomposition

**Perception loss** (backprop directly):

$$\mathcal{L}^{\text{perc}}(\theta) = -\sum_{t=1}^{T} \sigma_t \ln p_\theta(x_t \mid c, x_{<t})$$

**Action loss** (REINFORCE with advantage):

$$\mathcal{L}^{\text{act}}(\theta) = -\sum_{t=1}^{T} (1 - \sigma_t) \ln p_\theta(x_t \mid c\, x_{<t}) \cdot \text{sg}(\hat{A}_t) \;-\; \beta \sum_{t=1}^{T} (1 - \sigma_t) \, H\big[p_\theta(\cdot \mid c\, x_{<t})\big]$$

where $sg(\hat{A}_{t})$ is a stop gradient applied to the advantage estimate, ensuring that the policy update only adjusts the action probabilities without artificially altering the underlying value estimates to minimise the loss. The $(1 - \sigma_t)$ mask guarantees that both this REINFORCE update and the entropy bonus ($\beta H$) are strictly applied to the agent's action tokens, driving behavioural exploration while leaving the perception model undisturbed.

**Combined**: $\mathcal{L}(\theta) = \alpha_{\text{perc}} \cdot \mathcal{L}^{\text{perc}} + \alpha_{\text{act}} \cdot \mathcal{L}^{\text{act}}$. Both computed from a single forward pass; the $\sigma_t$ mask routes each position.

### 7.2 Algorithm: Character-Conditioned Surprise Minimisation (CCSM)

```
Initialise: policy/world model θ, value head ϕ
Hyperparameters: discount γ, entropy coefficient β, learning rates η_θ and η_ϕ,
                 number of rollouts per batch K, max trajectory length T

For each training iteration:

  1. COLLECT ROLLOUTS
     For k = 1 to K:
       Prepend character prompt c to context
       For t = 1 to T:
         If σ_t = 0 (action):  sample x_t ~ p_θ(· | c, x_{<t})
         If σ_t = 1 (observation):  receive x_t from environment
         Append x_t to context
       Store trajectory (x_{1:T}, σ_{1:T})

  2. COMPUTE SURPRISES
     Forward pass of p_θ over [c, x_{1:T}]
     For all t:  s_t = -ln p_θ(x_t | c, x_{<t})

  3. COMPUTE RETURNS
     For each action token t (σ_t = 0):
       G_t = Σ_{s > t} σ_s · γ^(s-t) · s_s

  4. COMPUTE ADVANTAGES
     For each action token t (σ_t = 0):
       A_t = -(G_t - V_ϕ(x_{≤t}, c))

  5. COMPUTE LOSSES
     L_perc = (1/N_obs) Σ_t σ_t · s_t
     L_act  = -(1/N_act) Σ_t (1-σ_t) · ln p_θ(x_t | c, x_{<t}) · sg(A_t)
              - β · (1/N_act) Σ_t (1-σ_t) · H[p_θ(· | c, x_{<t})]
     L_val  = (1/N_act) Σ_t (1-σ_t) · (V_ϕ(x_{≤t}, c) - sg(G_t))²

  6. UPDATE
     θ ← θ - η_θ · ∇_θ (L_perc + L_act)
     ϕ ← ϕ - η_ϕ · ∇_ϕ L_val
```

### 7.3 Notes

**PPO clipping.** For stability, store old log-probs from rollout, clip the ratio $r_t = p_\theta / p_{\theta_{\text{old}}}$.

**GAE.** Generalised advantage estimation can treat the observation surprise sequence as the reward stream.

**Value head.** Sharing the backbone with $p_\theta$ is efficient but may cause gradient interference; stop-gradient on the backbone for $L_{\text{val}}$ is standard.

**Pre-trained initialisation.** The algorithm assumes $p_\theta$ starts as a pre-trained LM. The character prompt leverages existing knowledge to define meaningful preferences from the start.

---

## 8. Properties of the Loss

### 8.1 Self-consistency as the objective

As training converges, the model's predictions and actions become mutually consistent: it acts in ways that produce the observations it expects, and expects the observations its actions produce. This is the "niche seeking" property from APD.

This is a deliberate philosophical commitment, grounded in the same fallibilist epistemology that informed the loss derivation. The framework rejects the assumption that there exists some objective utility function, external to the agent's own model, against which behaviour can be measured as "correct." Such an assumption would constitute an explanatory terminus — a claim to ground-truth access that no embedded agent can possess. Instead, we hold that an agent embedded in the world it is modelling can only ever achieve self-consistency: coherence between its model and its experience, under the constraint that its model is always improvable. The character prompt $c$ specifies a perspective, and the agent learns to inhabit that perspective self-consistently. Self-consistency is not a weaker substitute for correctness; it is the only coherent target for an embedded, fallible agent.

### 8.2 Approximations

The derivation involves one step that can be read as either an approximation or a principled commitment, depending on epistemological stance: dropping the $\ln p_{\text{env}}$ terms from the policy gradient (Section 5.5). Under standard assumptions about irreducible environment stochasticity, this introduces a bias. Under the fallibilist epistemology adopted here — which treats all unpredictability as epistemic — the simplified loss is the correct objective, since there is no irreducible stochasticity to account for. The practical consequence is the same either way: the gap between full and simplified objectives shrinks as the world model improves.

The self-referential KL (Section 3.3) means the target shifts during training. This is standard in iterative schemes (EM, iterative amortised inference) but means fixed-target convergence guarantees do not directly apply.

### 8.3 The dark room problem

The objective admits trivial equilibria where the agent avoids all unpredictable situations. In single-agent environments, several factors counteract this: model expressiveness (an expressive LM conditioned on a rich prompt predicts _varied_ trajectories, so the KL penalises narrow visitation); entropy regularisation (directly penalises policy collapse); environmental forcing (a non-trivial environment introduces novelty regardless); and pretrained initialisation (converging to a trivial equilibrium requires unlearning rich pretrained structure). The adequacy of these mitigations is ultimately empirical.

However, the strongest argument against the dark room comes from the multi-agent setting (Section 10), which is the intended deployment. Other agents of comparable complexity are inexhaustible sources of epistemic prediction error, making trivial equilibria unreachable in principle.

### 8.4 Prompt sensitivity

The quality of behavioural shaping is bounded by the pretrained model's ability to distinguish trajectories conditional on $c$. Poorly written prompts may induce unintended equilibria. This is both the primary strength (flexibility) and the primary risk (fragility).

---

## 9. Connection to the APD Framework

|APD Concept|Our Instantiation|
|---|---|
|Random variables $x$|Observation tokens|
|Random variables $z$|Action tokens|
|Actual distribution $p_\phi(x, z)$|$p_\theta^{\text{actual}}(x_{1:T} \mid c)$|
|Target distribution $\tau(x, z)$|$\tau_\theta(x_{1:T} \mid c) = \prod_t p_\theta(x_t \mid c, x_{<t})$|
|Perception|NTP on observation tokens|
|Action|REINFORCE with future observation surprise as return|
|Models as preferences|Character-conditioned model _is_ the preference distribution|
|Niche seeking|Agent converges to self-predictable trajectories|

The joint KL decomposes per APD Section 2.4 / Equation 6 into perception (NTP on past observations) and control (REINFORCE on future observations). The information gain term from APD Section 3.8 is implicit: the intrinsic return pushes the agent toward trajectories where its world model is accurate.

---

## 10. Multi-Agent Environments

The intended deployment of this framework is in multi-agent environments where multiple independently-trained agents interact, each treating all other agents as part of its environment. This setting has profound consequences for the dynamics of the objective — in particular, it largely dissolves the dark room problem and provides a natural account of why theory of mind should emerge.

### 10.1 Other agents as inexhaustible sources of prediction error

In a single-agent environment with fixed dynamics, the world model can in principle converge to perfect prediction, at which point the surprise minimisation objective is satisfied and learning stops. This is the dark room.

In a multi-agent environment, perfect prediction is impossible in principle — not because of aleatoric stochasticity (which we reject) but because of computational irreducibility. Modelling another agent of comparable complexity requires modelling a system that is simultaneously modelling you. This is exactly the kind of recursive self-reference that Gödel's incompleteness results, Turing's halting problem, and Rice's theorem show cannot be fully resolved from the inside. Each agent is an embedded observer trying to predict a system that contains an embedded observer trying to predict it. The unpredictability is epistemic all the way down, but it is also inexhaustible: no finite model can perfectly predict another model of comparable complexity that is co-adapting in response to it.

The consequence is that the observation surprise $-\ln p_\theta(x_t \mid c, x_{<t})$ at tokens produced by other agents never reaches zero. The perception loss always has signal. The policy gradient always has somewhere to push. The system remains permanently in a regime of adaptation, where the "environment" (other agents) is non-stationary not because of random perturbation but because of co-adaptation — other agents are also surprise-minimising, also updating their models, also seeking self-consistency. The moment one agent finds a comfortable niche, the others' adaptation changes the dynamics of that niche.

### 10.2 Theory of mind as surprise reduction

In a multi-agent environment, the largest and most reducible source of observation surprise is other agents' behaviour. The most effective way to reduce this surprise is to model their internal states — their beliefs, intentions, and likely future actions. The surprise minimisation objective therefore provides direct pressure toward theory of mind: an agent that builds accurate models of what other agents know, want, and plan will achieve lower observation surprise than one that treats other agents as opaque stochastic processes.

This connects to a natural "sweet spot" for learning. Agents that are much simpler than the focal agent are easily predictable — low surprise, low learning signal. Systems that are far more complex (or genuinely non-agentive complex systems) may be intractably surprising — high surprise, but prediction errors too large and varied to learn from efficiently. Other agents of _comparable_ complexity occupy the optimal learning frontier: mostly predictable (they share similar architecture, similar priors from pretraining) but with persistent anomalies (their specific character prompts, private observations, and internal states differ). Efficient learning happens at this edge of current understanding.

### 10.3 Communication and the value of honesty

If another agent communicates its intentions — and does so honestly — the focal agent's observation surprise about that agent's future actions drops. Honest communication is therefore directly rewarded by the objective. Deceptive communication may reduce surprise in the short term but increases it when the deception is revealed, creating a net penalty over the trajectory. The character prompt mediates this: an agent whose prompt specifies a cooperative disposition will find honest communication self-consistent, while an agent whose prompt specifies deception may find sustained dishonesty self-consistent at the cost of higher long-term surprise from other agents' retaliatory adaptations.

### 10.4 Implications for the tractability approximation

The dropped $\ln p_{\text{env}}$ terms (Section 5.5) take on particular significance in multi-agent settings. In the single-agent case, these terms measure how actions affect the environment's intrinsic predictability. In the multi-agent case, they measure how the focal agent's actions affect _other agents' behaviour_ — including whether those actions make other agents more or less predictable. An agent that understands "if I do X, the other agent becomes more predictable to me" is doing something strategically valuable. The bias from dropping these terms means our agent cannot _explicitly_ route toward states where other agents are easier to model; it can only route toward states where its _current model_ is accurate. Since the perception loss continuously improves the model, these converge over training — but the gap may be more consequential in multi-agent settings than in single-agent ones, and represents a direction for future work.

---

## 11. Summary

Starting from APD's joint KL minimisation and choosing the target as the model's own character-conditioned predictive distribution:

$$\mathcal{L}(\theta) = \mathbb{E}_{p_\theta^{\text{actual}}}\left[-\sum_{t=1}^{T} \sigma_t \ln p_\theta(x_t \mid c, x_{<t})\right]$$

$$\nabla_\theta^{\text{act}} \propto \mathbb{E}_{p_\theta^{\text{actual}}}\left[\nabla_\theta \ln p_\theta(x_t^{\text{act}} \mid c, x_{<t}) \cdot \left(\sum_{s > t} \sigma_s \ln p_\theta(x_s \mid c, x_{<s})\right)\right]$$

Perception is next-token prediction. Action learning reinforces actions that lead to unsurprising futures. No extrinsic reward — the character prompt alone shapes behaviour.

The framework rests on two commitments. First, a fallibilist epistemology that treats all unpredictability as epistemic rather than aleatoric, which renders the simplified loss a principled objective rather than an approximation. Second, self-consistency as the optimisation target: an embedded agent cannot access ground truth, only coherence between its model and its experience.

In multi-agent environments, these commitments have strong consequences. Other agents of comparable complexity are inexhaustible sources of epistemic prediction error — unpredictable not because of randomness but because of computational irreducibility and co-adaptation. This dissolves the dark room problem, provides natural pressure toward theory of mind, and ensures the system remains permanently in a regime of learning. The loss function does not change; the multi-agent structure enters only through the environment dynamics. But it transforms the objective from one that admits trivial equilibria to one that sustains open-ended adaptation.

---

## References

- Hafner, D., Ortega, P. A., Ba, J., Parr, T., Friston, K., & Heess, N. (2022). Action and Perception as Divergence Minimization. _arXiv:2009.01791v3_.
- Todorov, E. (2008). General duality between optimal control and estimation. _47th IEEE CDC_.
- Kappen, H. J., Gómez, V., & Opper, M. (2009). Optimal control as a graphical model inference problem. _Machine Learning, 87_(2).
- Friston, K. (2010). The free-energy principle: a unified brain theory? _Nature Reviews Neuroscience, 11_(2).
- Schulman, J., Wolski, F., Dhariwal, P., Radford, A., & Klimov, O. (2017). Proximal policy optimization algorithms. _arXiv:1707.06347_.
- Williams, R. J. (1992). Simple statistical gradient-following algorithms for connectionist reinforcement learning. _Machine Learning, 8_(3-4).
- Evans, J., Bratton, B., & Agüera y Arcas, B. (2026). Agentic AI and the next intelligence explosion. _arXiv:2603.20639_.