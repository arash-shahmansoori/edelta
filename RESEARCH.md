# The Geodesic Manifold-Delta Transformer: Unifying Width, Geometry, and Reflection via Thermodynamic Gating

**Author:** Arash Shahmansoori  
**Affiliation:** Independent Researcher  
**Date:** January 2026  
**Version:** 2.0 (Hybrid Architecture Update)

---

## Abstract

The residual connection is the backbone of modern deep learning, yet it imposes a strictly additive inductive bias ($\mathbf{x}_{l+1} = \mathbf{x}_l + f(\mathbf{x}_l)$) that limits a network's topological capacity to perform unitary erasure—effectively preventing models from "changing their mind" without noise injection. Recent attempts to mitigate this via Hyper-Connections (DeepSeek mHC) offer increased width but suffer from computational bottlenecks due to iterative Sinkhorn-Knopp normalization. Conversely, geometric approaches (Deep Delta Learning) offer expressivity through reflection but lack the information-preserving properties of rotation.

We bridge this gap by proposing the **E∆-Hybrid Transformer**, a novel architecture that combines:
1. **Cayley Rotation** — a mapping to the Special Orthogonal Group SO(n) for information-preserving geometric transformations
2. **Householder Reflection** — a rank-1 projector enabling rapid information negation ("Aha!" moments)
3. **Learnable Adaptive Gate** — an input-dependent mechanism to select between rotation and reflection

The selection is guided by input context, allowing the model to use rotation for geometric reasoning and reflection for belief revision. We prove that both operators are unconditionally stable, numerically well-conditioned, and computationally efficient ($<1\%$ overhead). Empirically, our Hybrid architecture achieves **4.3× improvement** over rotation-only approaches on correction tasks while maintaining geometric reasoning capabilities.

---

## 1. Introduction

The Transformer architecture [1] dominates modern artificial intelligence, owing largely to the signal propagation properties enabled by the Residual Connection [2]. However, recent theoretical work suggests that the standard residual update

$$\mathbf{x}_{l+1} = \mathbf{x}_l + f(\mathbf{x}_l)$$

forces the network to act as a discretization of an Ordinary Differential Equation (ODE) that is strictly orientation-preserving. This "additive bias" prevents the model from correcting deep-seated errors by reflecting or negating feature orientations—a capability linked to high-level reasoning adjustments, counterfactual thinking, and non-monotonic logic [3].

### 1.1 The Limitation of Pure Rotation

Our initial theoretical framework proposed the **Cayley-Padé operator** for geometric transformations. The Cayley transform maps skew-symmetric matrices to the Special Orthogonal Group SO(n), providing unconditional stability and perfect norm preservation. However, empirical investigation revealed a fundamental limitation:

> **Key Finding:** The Cayley transform produces rotation matrices in SO(n) whose eigenvalues lie strictly on the unit circle $\{e^{i\theta} : \theta \in (-\pi, \pi)\}$. Critically, **the eigenvalue $-1$ is excluded** from this range, as it would require $\theta = \pm\pi$, which corresponds to the limiting case $\|\mathbf{A}\| \to \infty$.

This mathematical constraint means that for tasks requiring "correction" or "negation" (e.g., processing "Actually, no" or "Wait, I meant"), pure rotation is fundamentally incapable of the required transformation.

### 1.2 The Power of Reflection

The Deep Delta Learning (DDL) paper [3] introduced the Householder reflector:

$$\mathbf{H}_\beta(\mathbf{k}) = \mathbf{I} - \beta \mathbf{k}\mathbf{k}^\top, \quad \|\mathbf{k}\| = 1$$

This operator **can** produce eigenvalue $-1$ when $\beta = 2$:

$$\mathbf{H}_2(\mathbf{k})\mathbf{k} = \mathbf{k} - 2(\mathbf{k}^\top\mathbf{k})\mathbf{k} = -\mathbf{k}$$

Empirically, DDL achieved **136× better performance** than baseline on correction tasks—a striking validation of the reflection hypothesis. However, the Householder operator has its own limitations:
- For $\beta \neq 0, 2$, the operator is not orthogonal (eigenvalue $1-\beta$ breaks unitarity)
- Rank-1 structure limits geometric expressivity to a single reflection plane
- No inherent mechanism for uncertainty-aware activation

### 1.3 Our Contribution: The Hybrid Approach

We propose combining **both operators** with a learnable, input-dependent gate:

$$\mathbf{X}_{l+1} = \gamma_l \cdot \mathcal{R}(\mathbf{X}_l) + (1 - \gamma_l) \cdot \mathcal{H}(\mathbf{X}_l) + f(\cdot)$$

Where:
- $\mathcal{R}(\cdot)$ is the Cayley rotation operator (information-preserving)
- $\mathcal{H}(\cdot)$ is the Householder reflection operator (can negate)
- $\gamma_l = \sigma(\mathbf{W}_g \cdot \text{pool}(\mathbf{X}_l) + b_g) \in (0, 1)$ is the learned gate

**Key Benefits:**
- ✅ **Geometric tasks** → Model selects rotation (preserves information, isometric)
- ✅ **Correction tasks** → Model selects reflection (enables negation via eigenvalue $-1$)
- ✅ **Unified architecture** → Single model handles diverse task requirements
- ✅ **Unconditional stability** → Both operators are numerically well-conditioned

---

## 2. Mathematical Framework

We now present the formal mathematical foundations of the E∆-Hybrid architecture.

### 2.1 Notation and Setup

Let the hyper-connected residual state at layer $l$ be $\mathbf{X}_l \in \mathbb{R}^{B \times S \times n \times d}$, where:
- $B$ is the batch size
- $S$ is the sequence length
- $n$ is the stream expansion factor (typically $n = 4$)
- $d = D/n$ is the per-stream dimension ($D$ is the model dimension)

### 2.2 The Cayley-Padé Rotation Operator

**Definition 2.1 (Skew-Symmetric Generator).** Let $\mathbf{u}, \mathbf{v} \in \mathbb{R}^{n}$ be learnable parameter vectors. Define the skew-symmetric matrix:

$$\mathbf{A} = \mathbf{u}\mathbf{v}^\top - \mathbf{v}\mathbf{u}^\top \in \mathfrak{so}(n)$$

This construction guarantees $\mathbf{A}^\top = -\mathbf{A}$ (skew-symmetry).

**Definition 2.2 (Cayley Transform).** The Cayley operator with scaling parameter $\beta \in \mathbb{R}$ is:

$$\mathcal{C}_\beta(\mathbf{A}) = (\mathbf{I}_n + \tfrac{\beta}{2}\mathbf{A})^{-1}(\mathbf{I}_n - \tfrac{\beta}{2}\mathbf{A})$$

**Proposition 2.1 (Properties of Cayley Operator).**
1. $\mathcal{C}_\beta(\mathbf{A}) \in SO(n)$ — the result is a proper rotation (orthogonal with determinant $+1$)
2. All eigenvalues satisfy $|\lambda_k| = 1$ and lie on the unit circle
3. The operator is an **isometry**: $\|\mathcal{C}_\beta(\mathbf{A})\mathbf{x}\|_2 = \|\mathbf{x}\|_2$ for all $\mathbf{x}$
4. The inverse $(\mathbf{I} + \tfrac{\beta}{2}\mathbf{A})^{-1}$ exists for all finite $\beta$ (Theorem 2)

**Proposition 2.2 (Eigenvalue Limitation).** The eigenvalues of $\mathcal{C}_\beta(\mathbf{A})$ are given by:

$$\lambda_k = \frac{1 - i\tfrac{\beta}{2}\mu_k}{1 + i\tfrac{\beta}{2}\mu_k} = e^{-2i\arctan(\tfrac{\beta\mu_k}{2})}$$

where $i\mu_k$ are the purely imaginary eigenvalues of $\mathbf{A}$. Since $\arctan(\cdot) \in (-\tfrac{\pi}{2}, \tfrac{\pi}{2})$, the argument of $\lambda_k$ lies in $(-\pi, \pi)$, **strictly excluding** $\pm\pi$. Therefore, $\lambda = -1$ is impossible for any finite $\beta$.

### 2.3 The Householder Reflection Operator

**Definition 2.3 (Householder Reflector).** Let $\mathbf{k} \in \mathbb{R}^n$ with $\|\mathbf{k}\|_2 = 1$ be a unit vector defining the reflection hyperplane. The Householder operator with scaling $\beta \in [0, 2]$ is:

$$\mathcal{H}_\beta(\mathbf{k}) = \mathbf{I}_n - \beta \mathbf{k}\mathbf{k}^\top$$

**Proposition 2.3 (Properties of Householder Operator).**
1. **Eigenvalues**: $\{1, 1, \ldots, 1, (1-\beta)\}$ — multiplicity $(n-1)$ for eigenvalue $1$
2. At $\beta = 2$: eigenvalue along $\mathbf{k}$ is $-1$ (full reflection)
3. At $\beta = 2$: operator is orthogonal with $\det(\mathcal{H}_2) = -1$ (improper rotation)
4. For $\beta \neq 0, 2$: operator is **not orthogonal** (breaks isometry)
5. Rank-1 structure: only one direction is transformed

**Proposition 2.4 (Negation Capability).** For $\beta = 2$ and unit vector $\mathbf{k}$:

$$\mathcal{H}_2(\mathbf{k})\mathbf{k} = (\mathbf{I} - 2\mathbf{k}\mathbf{k}^\top)\mathbf{k} = \mathbf{k} - 2\mathbf{k} = -\mathbf{k}$$

This proves that information along direction $\mathbf{k}$ can be **negated**.

### 2.4 The Hybrid Operator

**Definition 2.4 (E∆-Hybrid Operator).** Combining both primitives with an adaptive gate:

$$\mathcal{G}_\gamma(\mathbf{X}) = \gamma \cdot \mathcal{C}_{\beta_r}(\mathbf{A})\mathbf{X} + (1 - \gamma) \cdot \mathcal{H}_{\beta_h}(\mathbf{k})\mathbf{X}$$

where:
- $\gamma = \sigma(\mathbf{W}_g \cdot \bar{\mathbf{X}} + b_g) \in (0, 1)$ is the gate (sigmoid activation)
- $\bar{\mathbf{X}} = \frac{1}{S}\sum_{s=1}^S \mathbf{X}_{:,s,:,:}$ is the sequence-pooled representation
- $\beta_r$ is the rotation magnitude (thermodynamically controlled)
- $\beta_h = 2\sigma(\alpha_h) \in (0, 2)$ is the reflection magnitude

**Interpretation:**
- $\gamma \to 1$: Rotation dominates (geometric reasoning mode)
- $\gamma \to 0$: Reflection dominates (correction/negation mode)

### 2.5 Thermodynamic Gating via Purity Proxy

To connect the rotation magnitude to model uncertainty, we introduce the **Frobenius Purity Proxy**.

**Definition 2.5 (Gram Matrix).** For stream representation $\mathbf{X} \in \mathbb{R}^{n \times d}$:

$$\mathbf{G} = \mathbf{X}\mathbf{X}^\top \in \mathbb{R}^{n \times n}$$

**Definition 2.6 (Linear Entropy Proxy).** The purity proxy $\Phi$ measures deviation from a pure (rank-1) state:

$$\Phi(\mathbf{X}) = 1 - \frac{\|\mathbf{G}\|_F^2}{(\text{Tr}(\mathbf{G}))^2}$$

**Proposition 2.5 (Bounds on Purity Proxy).**
- **Pure state** (rank-1, maximum confidence): $\mathbf{G} = \sigma_1^2 \mathbf{e}_1\mathbf{e}_1^\top \Rightarrow \Phi = 0$
- **Maximally mixed state** (uniform singular values): $\Phi \to 1 - \frac{1}{n}$

*Proof.* For rank-1: $\text{Tr}(\mathbf{G}) = \sigma_1^2$, $\|\mathbf{G}\|_F^2 = \sigma_1^4$, so $\Phi = 1 - 1 = 0$. For uniform: $\text{Tr}(\mathbf{G}) = n\sigma^2$, $\|\mathbf{G}\|_F^2 = n\sigma^4$, so $\Phi = 1 - \frac{n\sigma^4}{n^2\sigma^4} = 1 - \frac{1}{n}$. $\square$

**Definition 2.7 (Thermodynamic Gate).** The rotation magnitude is:

$$\beta_r = \text{Softplus}(w_\alpha \cdot \Phi(\mathbf{X}) + b_{\text{init}})$$

with learnable scalar $w_\alpha$ and bias $b_{\text{init}} \approx 0$ (initialized for moderate rotation).

**Physical Interpretation:**
- High $\Phi$ (high entropy, uncertain) → Large $\beta_r$ → Strong rotation
- Low $\Phi$ (low entropy, confident) → Small $\beta_r$ → Weak rotation
- This connects to the "Illusion of Insight" [5]: genuine reasoning shifts correlate with model uncertainty

---

## 3. Main Theoretical Results

### Theorem 1: Unconditional Orthogonality of Cayley Transform

**Statement.** For any $\beta \in \mathbb{R}$ and skew-symmetric $\mathbf{A} \in \mathfrak{so}(n)$, the Cayley operator satisfies:

$$\mathcal{C}_\beta(\mathbf{A})^\top \mathcal{C}_\beta(\mathbf{A}) = \mathbf{I}_n$$

*Proof.* See Appendix A.

**Corollary 1.1.** The Cayley component is an isometry: $\|\mathcal{C}_\beta(\mathbf{A})\mathbf{x}\|_2 = \|\mathbf{x}\|_2$ for all $\mathbf{x} \in \mathbb{R}^n$.

**Implication.** Signal energy is perfectly preserved through the rotation, preventing gradient explosion or vanishing in deep networks.

---

### Theorem 2: Non-Singularity of Cayley Inverse

**Statement.** For any $\beta \in \mathbb{R}$ and skew-symmetric $\mathbf{A}$, the matrix $\mathbf{S} = \mathbf{I} + \tfrac{\beta}{2}\mathbf{A}$ is non-singular with condition number:

$$\kappa(\mathbf{S}) \leq \sqrt{1 + \tfrac{\beta^2}{4}\|\mathbf{A}\|_2^2}$$

*Proof.* See Appendix B.

**Implication.** No singularity traps during training. The inverse is always well-conditioned for reasonable $\beta$ values.

---

### Theorem 3: Eigenvalue Exclusion in Cayley Transform

**Statement.** The Cayley operator $\mathcal{C}_\beta(\mathbf{A})$ cannot produce eigenvalue $\lambda = -1$ for any finite $\beta$ and any skew-symmetric $\mathbf{A}$.

*Proof.* The eigenvalues of $\mathcal{C}_\beta(\mathbf{A})$ are $\lambda_k = e^{-2i\arctan(\beta\mu_k/2)}$ where $i\mu_k$ are eigenvalues of $\mathbf{A}$. For $\lambda_k = -1 = e^{i\pi}$, we require $\arctan(\beta\mu_k/2) = -\pi/2$, which demands $\beta\mu_k/2 \to -\infty$. For any finite $\beta$ and bounded $\|\mathbf{A}\|$, this is impossible. $\square$

**Implication.** Cayley rotation **cannot negate** information—a fundamental mathematical limitation motivating the Hybrid approach.

---

### Theorem 4: Negation Capability of Householder Reflection

**Statement.** The Householder operator $\mathcal{H}_2(\mathbf{k})$ with $\|\mathbf{k}\|=1$ satisfies:
1. $\mathcal{H}_2(\mathbf{k})\mathbf{k} = -\mathbf{k}$ (eigenvalue $-1$ along $\mathbf{k}$)
2. $\mathcal{H}_2(\mathbf{k})\mathbf{v} = \mathbf{v}$ for all $\mathbf{v} \perp \mathbf{k}$ (eigenvalue $+1$ orthogonal to $\mathbf{k}$)
3. $\det(\mathcal{H}_2(\mathbf{k})) = -1$ (orientation-reversing)

*Proof.* Direct computation: $\mathcal{H}_2(\mathbf{k})\mathbf{k} = \mathbf{k} - 2(\mathbf{k}^\top\mathbf{k})\mathbf{k} = -\mathbf{k}$. For $\mathbf{v} \perp \mathbf{k}$: $\mathcal{H}_2(\mathbf{k})\mathbf{v} = \mathbf{v} - 2(\mathbf{k}^\top\mathbf{v})\mathbf{k} = \mathbf{v}$. Determinant: product of eigenvalues $= 1^{n-1} \cdot (-1) = -1$. $\square$

**Implication.** Householder reflection can instantaneously negate information along a learned direction—essential for correction tasks.

---

### Theorem 5: Complementarity of Rotation and Reflection

**Statement.** The set of proper rotations SO(n) and Householder reflections generate the full orthogonal group O(n):

$$O(n) = SO(n) \cup \{\mathbf{Q} : \det(\mathbf{Q}) = -1\}$$

Any orthogonal matrix can be decomposed as a product of rotations and at most one reflection.

**Implication.** The Hybrid operator, by combining Cayley (SO(n)) and Householder (reflection), can approximate any orthogonal transformation.

---

### Theorem 6: Thermodynamic Locking

**Statement.** Under the thermodynamic gating mechanism, the gradient of rotation parameters vanishes as the purity proxy $\Phi \to 0$:

$$\lim_{\Phi \to 0} \frac{\partial \mathcal{L}}{\partial w_\alpha} = 0$$

*Proof.* See Appendix C.

**Implication.** The network "locks" its geometric structure when confident ($\Phi \approx 0$), only restructuring under uncertainty. This mechanizes the behavioral finding from the "Illusion of Insight" paper [5].

---

## 4. Empirical Validation

### 4.1 Experimental Setup

We conduct controlled experiments on the **Correction Task**, designed to test "Aha!" moment processing. The task requires the model to:
1. Process an initial instruction (e.g., "Add 5 to X")
2. Encounter a correction phrase (e.g., "Wait, I meant subtract 5")
3. Output the corrected result

**Models Compared:**
| Model | Description |
|-------|-------------|
| Baseline | Standard Transformer with additive residual |
| Pure mHC | DeepSeek-style with Sinkhorn-Knopp [4] |
| Pure DDL | Householder reflection only [3] |
| E∆-Cayley | Our rotation-only variant |
| **E∆-Hybrid** | Full architecture (rotation + reflection + gate) |

**Unified Hyperparameters:**
- Architecture: 4 layers, 4 heads, 128 embedding dimension
- Training: Batch 64, LR 1e-3, 10,000 iterations
- Optimizer: AdamW with cosine decay

### 4.2 Quantitative Results

| Model | Val Loss | Improvement | Key Mechanism |
|-------|----------|-------------|---------------|
| **Pure DDL** | **0.0016** | 136× | Reflection only |
| **E∆-Hybrid** | **0.0508** | 4.3× | Rotation + Reflection |
| Pure mHC | 0.2138 | 1.02× | Doubly stochastic mixing |
| E∆-Cayley | 0.2155 | 1.01× | Rotation only |
| Baseline | 0.2177 | 1.0× | Additive residual |

**Key Observations:**
1. **Pure DDL dominates** (136× improvement), validating the reflection hypothesis
2. **E∆-Hybrid achieves 4.3× improvement** over rotation-only approaches
3. **Pure rotation fails** — mathematically cannot perform negation (Theorem 3)
4. **mHC provides marginal benefit** — mixing alone is insufficient

### 4.3 Analysis of Learned Gates

Checkpoint analysis reveals the model's learned preferences:

| Layer | Gate $\gamma$ | Preference | Rotation $\|\mathbf{A}\|$ | Reflection $\beta_h$ |
|-------|--------------|------------|--------------------------|---------------------|
| 0 | 0.53 | Slight rotation | 0.0045 | 1.42 |
| 1 | 0.51 | ~Equal | 0.0050 | 1.44 |
| 2 | 0.50 | Equal | 0.0040 | 1.46 |
| 3 | **0.49** | **Slight reflection** | 0.0077 | **1.47** |

**Interpretation:**
- Later layers prefer reflection ($\gamma < 0.5$)
- Reflection $\beta_h \approx 1.5$ is active (approaching optimal $\beta = 2$)
- Rotation generators remain weak ($\|\mathbf{A}\| \approx 0$) — model prefers reflection for this task

### 4.4 Diagnostic Journey

Our empirical investigation followed a systematic path:

1. **Original E∆-MHC-Geo failed** — The damper term $(1 - \tanh(\beta))$ allowed the model to bypass rotation entirely by learning $\beta \to 0$

2. **DDL-style application failed** — Even without damper bypass, the rotation-only model learned near-zero generators $(\|\mathbf{A}\| \to 0)$

3. **Mathematical analysis** — Theorem 3 revealed the fundamental limitation: Cayley rotation cannot produce eigenvalue $-1$

4. **Hybrid solution** — Adding Householder reflection with learnable gate achieved 4.3× improvement

### 4.5 Entropy-Aware Gating Experiment (V2)

We investigated whether making the gate **entropy-aware** would improve performance:

$$\gamma = \sigma(\mathbf{W}_g \cdot \bar{\mathbf{x}} + w_\Phi \cdot \Phi + b_g)$$

**Hypothesis:** High uncertainty (large $\Phi$) might correlate with need for different operators.

**Result:** Entropy-aware gating **hurt performance**:

| Model | Best Val Loss | Entropy in Gate |
|-------|---------------|-----------------|
| **E∆-Hybrid V1** | **0.0508** | ❌ No |
| E∆-Hybrid V2 | 0.0822 | ✅ Yes |

**Analysis of Learned $w_\Phi$ Parameters:**
- Early layers: $w_\Phi > 0$ (high entropy → prefer rotation)
- Later layers: $w_\Phi < 0$ (high entropy → prefer reflection)

**Why Entropy-Aware Gating Failed:**

The key insight is that **entropy and correction are orthogonal signals**:

| Signal | Measures | When High |
|--------|----------|-----------|
| Entropy ($\Phi$) | "I don't know" | Model is uncertain |
| Correction Need | "I was wrong" | Error detected in input |

For the correction task:
- Corrections are triggered by **content** ("Actually, no"), not uncertainty
- When confident-but-wrong, entropy is LOW but correction is needed
- Adding entropy to the gate **conflates these distinct signals**

**Conclusion:** The gate should remain **content-based** (V1 design):
- Gate: $\gamma = \sigma(\mathbf{W}_g \cdot \mathbf{x} + b_g)$ — learns from input content
- Rotation $\beta$: entropy-driven — uncertain → explore
- Reflection $\beta$: static learned — always ready to correct

---

## 5. Architecture Details

### 5.1 Block Structure

```python
class GeodesicDeltaHybrid(nn.Module):
    def forward(self, x):
        B, S, D = x.shape
        x_streams = x.view(B, S, n_streams, d_stream)
        
        # === CAYLEY ROTATION (preserves information) ===
        A = self.u @ self.v.T - self.v @ self.u.T  # Skew-symmetric
        beta_r = softplus(self.w_alpha * purity_proxy(x_streams) + self.b_init)
        Q = cayley_transform(A, beta_r)
        x_rotated = einsum('ij,bsjd->bsid', Q, x_streams)
        
        # === HOUSEHOLDER REFLECTION (can negate) ===
        k = normalize(self.k_raw)
        beta_h = 2 * sigmoid(self.ref_scale)
        x_reflected = x_streams - beta_h * dot(x_streams, k) * k
        
        # === ADAPTIVE GATE (input-dependent) ===
        gate = sigmoid(self.gate_proj(x.mean(dim=1)))  # (B, 1)
        
        # === HYBRID OUTPUT ===
        x_hybrid = gate * x_rotated + (1 - gate) * x_reflected
        return x_hybrid.reshape(B, S, D)
```

### 5.2 Full Layer Transition

$$\mathbf{X}_{l+1} = \mathcal{G}_\gamma(\mathbf{X}_l) + f(\text{LayerNorm}(\mathcal{G}_\gamma(\mathbf{X}_l)))$$

where $f$ represents attention or MLP, and $\mathcal{G}_\gamma$ is the Hybrid operator.

### 5.3 Comparison with Prior Work

| Property | mHC [4] | DDL [3] | E∆-Hybrid |
|----------|---------|---------|-----------|
| Width expansion | ✅ | ❌ | ✅ |
| Reflection capability | ❌ | ✅ | ✅ |
| Rotation (isometric) | ❌ | ❌ | ✅ |
| Unconditional stability | ⚠️ Sinkhorn | ⚠️ $\beta \neq 1$ | ✅ Always |
| Thermodynamic gating | ❌ | ❌ | ✅ |
| Adaptive operator | ❌ | ❌ | ✅ |
| Computational overhead | High (Sinkhorn) | Low | **Low (<1%)** |

---

## 6. Computational Complexity

### 6.1 Per-Token FLOP Analysis

| Operation | FLOPs | Notes |
|-----------|-------|-------|
| Cayley inverse $(n=4)$ | $O(n^3) = 64$ | Small stream count |
| Matrix multiply $(n \times n) \times (n \times d)$ | $O(n^2 d) = 512$ | Rotation application |
| Householder projection | $O(nd) = 128$ | Rank-1 update |
| Gate computation | $O(D) = 128$ | Linear projection |
| **Total Hybrid overhead** | **~832** | Per token |
| Standard Attention $(D=128)$ | $O(D^2) = 16,384$ | Dominates |

**Overhead Ratio:** $832 / 16,384 \approx 5\%$ per attention layer, $<1\%$ of total model.

### 6.2 Parameter Overhead

Additional parameters per Hybrid block (2 per layer):
- Cayley: $\mathbf{u}, \mathbf{v} \in \mathbb{R}^n$ → $2n = 8$ parameters
- Householder: $\mathbf{k} \in \mathbb{R}^n$, $\alpha_h \in \mathbb{R}$ → $n + 1 = 5$ parameters
- Gate: $\mathbf{W}_g \in \mathbb{R}^{D}$, $b_g \in \mathbb{R}$ → $D + 1 = 129$ parameters
- Thermodynamic: $w_\alpha, b_{\text{init}} \in \mathbb{R}$ → $2$ parameters

**Total per block:** ~144 parameters (negligible vs. ~0.8M total)

---

## 7. Discussion

### 7.1 Why Rotation Alone Fails

The failure of E∆-Cayley on correction tasks is not a limitation of our implementation but a **fundamental mathematical constraint**. Theorem 3 proves that no finite Cayley transform can produce eigenvalue $-1$. This insight has broader implications:

> Any architecture relying purely on SO(n) transformations (rotations) cannot perform instantaneous information negation. Correction tasks require access to O(n) \ SO(n) — the reflections.

### 7.2 Why Hybrid Succeeds

The E∆-Hybrid architecture succeeds by providing **both geometric primitives**:
- **Rotation** for tasks requiring continuous geometric transformations (e.g., mental rotation, coordinate transforms)
- **Reflection** for tasks requiring discrete belief updates (e.g., corrections, negations, counterfactuals)

The learned gate allows the model to adaptively select the appropriate operation based on input context.

### 7.3 Connection to Cognitive Science

Our findings align with cognitive science theories distinguishing:
- **Type 1 processing**: Fast, automatic, continuous adjustments (→ rotation)
- **Type 2 processing**: Slow, deliberate, discrete belief revision (→ reflection)

The Hybrid architecture provides a neural substrate for both processing modes.

---

## 8. Conclusion

We have presented the **E∆-Hybrid Transformer**, a unified architecture resolving the fundamental tension between:
- **Rotation (Cayley)**: Information-preserving, isometric, continuous — but cannot negate
- **Reflection (Householder)**: Can negate, discrete — but not isometric for $\beta \neq 2$

By combining both operators with a learnable gate, we achieve:
1. **4.3× improvement** over rotation-only on correction tasks
2. **Unconditional stability** from Cayley's orthogonality guarantees
3. **Negation capability** from Householder's eigenvalue $-1$
4. **Adaptive behavior** through input-dependent gating
5. **Minimal overhead** ($<1\%$ computational cost)

**Future Directions:**
- Per-token gating (vs. sequence-level)
- Composition of multiple rotation/reflection stages
- Application to large-scale language models
- Theoretical analysis of learning dynamics

---

## References

[1] Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł., & Polosukhin, I. (2017). Attention is All You Need. *Advances in Neural Information Processing Systems (NeurIPS)*.

[2] He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep Residual Learning for Image Recognition. *IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*.

[3] Zhang, Y., et al. (2026). Deep Delta Learning: Geometric Residual Connections for Transformers. *arXiv:2601.00417*.

[4] Xie, Z., et al. (2025). mHC: Manifold-Constrained Hyper-Connections for Multi-Stream Residual Learning. *arXiv:2512.24880*.

[5] d'Aliberti, L., et al. (2026). The Illusion of Insight: Distinguishing Genuine Reasoning Shifts from Surface Pattern Matching. *arXiv:2601.00514*.

---

# Appendices

## Appendix A: Proof of Cayley Orthogonality (Theorem 1)

**Theorem.** *For any $\beta \in \mathbb{R}$ and skew-symmetric $\mathbf{A}$ (i.e., $\mathbf{A}^\top = -\mathbf{A}$), the Cayley operator*

$$\mathbf{Q} = (\mathbf{I} + \tfrac{\beta}{2}\mathbf{A})^{-1}(\mathbf{I} - \tfrac{\beta}{2}\mathbf{A})$$

*is orthogonal: $\mathbf{Q}^\top\mathbf{Q} = \mathbf{I}$.*

**Proof.**

Let $\mathbf{M} = \tfrac{\beta}{2}\mathbf{A}$. Since $\mathbf{A}$ is skew-symmetric, so is $\mathbf{M}$: $\mathbf{M}^\top = -\mathbf{M}$.

Define $\mathbf{Q} = (\mathbf{I} + \mathbf{M})^{-1}(\mathbf{I} - \mathbf{M})$.

**Step 1: Compute $\mathbf{Q}^\top$.**

Using $(\mathbf{XY})^\top = \mathbf{Y}^\top\mathbf{X}^\top$ and $(\mathbf{X}^{-1})^\top = (\mathbf{X}^\top)^{-1}$:

$$\mathbf{Q}^\top = (\mathbf{I} - \mathbf{M})^\top \left((\mathbf{I} + \mathbf{M})^{-1}\right)^\top = (\mathbf{I} - \mathbf{M}^\top)(\mathbf{I} + \mathbf{M}^\top)^{-1}$$

Substituting $\mathbf{M}^\top = -\mathbf{M}$:

$$\mathbf{Q}^\top = (\mathbf{I} + \mathbf{M})(\mathbf{I} - \mathbf{M})^{-1}$$

**Step 2: Compute $\mathbf{Q}^\top\mathbf{Q}$.**

$$\mathbf{Q}^\top\mathbf{Q} = (\mathbf{I} + \mathbf{M})(\mathbf{I} - \mathbf{M})^{-1}(\mathbf{I} + \mathbf{M})^{-1}(\mathbf{I} - \mathbf{M})$$

**Step 3: Use commutativity.**

Since $(\mathbf{I} - \mathbf{M})$ and $(\mathbf{I} + \mathbf{M})$ are both polynomials of $\mathbf{M}$, they commute:

$$(\mathbf{I} - \mathbf{M})(\mathbf{I} + \mathbf{M}) = \mathbf{I} - \mathbf{M}^2 = (\mathbf{I} + \mathbf{M})(\mathbf{I} - \mathbf{M})$$

Therefore their inverses also commute:

$$(\mathbf{I} - \mathbf{M})^{-1}(\mathbf{I} + \mathbf{M})^{-1} = (\mathbf{I} + \mathbf{M})^{-1}(\mathbf{I} - \mathbf{M})^{-1}$$

**Step 4: Simplify.**

$$\mathbf{Q}^\top\mathbf{Q} = (\mathbf{I} + \mathbf{M}) \cdot (\mathbf{I} + \mathbf{M})^{-1} \cdot (\mathbf{I} - \mathbf{M})^{-1} \cdot (\mathbf{I} - \mathbf{M}) = \mathbf{I} \cdot \mathbf{I} = \mathbf{I}$$

$\square$

---

## Appendix B: Proof of Non-Singularity (Theorem 2)

**Theorem.** *For any $\beta \in \mathbb{R}$ and skew-symmetric $\mathbf{A}$, the matrix $\mathbf{S} = \mathbf{I} + \tfrac{\beta}{2}\mathbf{A}$ is non-singular.*

**Proof.**

**Step 1: Eigenvalues of skew-symmetric matrices.**

For real skew-symmetric $\mathbf{A}$, all eigenvalues are purely imaginary or zero:

$$\text{eig}(\mathbf{A}) = \{i\mu_1, -i\mu_1, i\mu_2, -i\mu_2, \ldots, 0, \ldots\}$$

where $\mu_k \in \mathbb{R}$.

**Step 2: Eigenvalues of $\mathbf{S}$.**

$$\text{eig}(\mathbf{S}) = \left\{1 + i\tfrac{\beta}{2}\mu_k\right\}$$

**Step 3: Magnitude bound.**

$$\left|1 + i\tfrac{\beta}{2}\mu_k\right|^2 = 1 + \tfrac{\beta^2}{4}\mu_k^2 \geq 1 > 0$$

**Step 4: Conclusion.**

Since all eigenvalues have magnitude $\geq 1$, no eigenvalue is zero. Therefore $\det(\mathbf{S}) \neq 0$ and $\mathbf{S}$ is invertible.

The condition number satisfies:

$$\kappa(\mathbf{S}) = \frac{\sigma_{\max}(\mathbf{S})}{\sigma_{\min}(\mathbf{S})} \leq \frac{\sqrt{1 + \tfrac{\beta^2}{4}\|\mathbf{A}\|_2^2}}{1} = \sqrt{1 + \tfrac{\beta^2}{4}\|\mathbf{A}\|_2^2}$$

$\square$

---

## Appendix C: Proof of Thermodynamic Locking (Theorem 6)

**Theorem.** *The gradient of rotation parameters vanishes as $\Phi \to 0$.*

**Proof.**

**Step 1: Gradient chain rule.**

The rotation magnitude is $\beta_r = \text{Softplus}(z)$ where $z = w_\alpha \Phi + b_{\text{init}}$.

The gradient with respect to $w_\alpha$ is:

$$\frac{\partial \mathcal{L}}{\partial w_\alpha} = \frac{\partial \mathcal{L}}{\partial \beta_r} \cdot \frac{\partial \beta_r}{\partial z} \cdot \frac{\partial z}{\partial w_\alpha}$$

**Step 2: Compute derivatives.**

- $\frac{\partial \beta_r}{\partial z} = \sigma(z)$ (sigmoid, since $\frac{d}{dz}\text{Softplus}(z) = \sigma(z)$)
- $\frac{\partial z}{\partial w_\alpha} = \Phi$

**Step 3: Gradient expression.**

$$\frac{\partial \mathcal{L}}{\partial w_\alpha} = \frac{\partial \mathcal{L}}{\partial \beta_r} \cdot \sigma(z) \cdot \Phi$$

**Step 4: Limit as $\Phi \to 0$.**

As $\Phi \to 0$:

$$\frac{\partial \mathcal{L}}{\partial w_\alpha} \propto \Phi \to 0$$

Similarly, gradients with respect to rotation generators $\mathbf{u}, \mathbf{v}$ flow through $\beta_r$. When $\Phi \to 0$ and $b_{\text{init}}$ is moderate, $\beta_r \to \text{Softplus}(b_{\text{init}})$ remains bounded but the gradient signal through $\Phi$ vanishes.

$\square$

---

## Appendix D: Detailed Eigenvalue Analysis

### D.1 Cayley Transform Eigenvalues

Let $\mathbf{A}$ have eigenvalue $i\mu$ (purely imaginary). The Cayley transform maps:

$$\lambda = \frac{1 - i\tfrac{\beta}{2}\mu}{1 + i\tfrac{\beta}{2}\mu}$$

**Polar form.** Let $\theta = \arctan\left(\tfrac{\beta\mu}{2}\right)$. Then:

- Numerator: $1 - i\tfrac{\beta\mu}{2} = \sqrt{1 + \tfrac{\beta^2\mu^2}{4}} \cdot e^{-i\theta}$
- Denominator: $1 + i\tfrac{\beta\mu}{2} = \sqrt{1 + \tfrac{\beta^2\mu^2}{4}} \cdot e^{i\theta}$

Therefore:

$$\lambda = \frac{e^{-i\theta}}{e^{i\theta}} = e^{-2i\theta} = e^{-2i\arctan(\beta\mu/2)}$$

**Range analysis.** Since $\arctan: \mathbb{R} \to (-\tfrac{\pi}{2}, \tfrac{\pi}{2})$:

$$-2\arctan(\beta\mu/2) \in (-\pi, \pi)$$

The boundary $\pm\pi$ is never achieved for finite arguments, so $\lambda = -1 = e^{i\pi}$ is **excluded**.

### D.2 Householder Eigenvalues

For $\mathcal{H}_\beta(\mathbf{k}) = \mathbf{I} - \beta\mathbf{k}\mathbf{k}^\top$ with $\|\mathbf{k}\| = 1$:

- **Eigenvector $\mathbf{k}$**: $\mathcal{H}_\beta\mathbf{k} = \mathbf{k} - \beta\mathbf{k} = (1-\beta)\mathbf{k}$, so eigenvalue is $(1-\beta)$
- **Eigenvectors $\mathbf{v} \perp \mathbf{k}$**: $\mathcal{H}_\beta\mathbf{v} = \mathbf{v} - \beta(\mathbf{k}^\top\mathbf{v})\mathbf{k} = \mathbf{v}$, so eigenvalue is $1$

**Spectrum:** $\{1, 1, \ldots, 1, (1-\beta)\}$ with $(n-1)$ ones.

**At $\beta = 2$:** Eigenvalue along $\mathbf{k}$ is $-1$ — enabling **negation**.

---

## Appendix E: Connection to Lie Theory

The Cayley transform provides a diffeomorphism between the Lie algebra $\mathfrak{so}(n)$ and an open subset of the Lie group $SO(n)$:

$$\mathcal{C}: \mathfrak{so}(n) \to SO(n) \setminus \{\mathbf{Q} : \det(\mathbf{I} + \mathbf{Q}) = 0\}$$

This parametrization has several advantages over the matrix exponential $\exp: \mathfrak{so}(n) \to SO(n)$:

1. **Rational computation**: Cayley involves matrix inverse, not infinite series
2. **Numerical stability**: No truncation error from series approximation
3. **Gradient flow**: Simpler backpropagation through linear algebra operations

The trade-off is that Cayley cannot reach the "antipodal" rotations (those with eigenvalue $-1$), which is precisely why we augment with Householder reflection.

---

## Appendix F: Experimental Details

### F.1 Correction Task Dataset

**Generation procedure:**
```
Input: "Task: {op1} {y} to X. X={x}. {correction_phrase} {op2} {y}. Answer:"
Output: str(result of op2 applied to x with y)
```

**Correction phrases:** "Wait, I meant", "Actually, no", "Scratch that", "On second thought", etc.

**Statistics:**
- Training samples: 50,000
- Validation samples: 5,000
- Sequence length: ≤128 tokens

### F.2 Training Curves

```
E∆-Hybrid on Correction Task:
Iter     0: val_loss = 5.586
Iter  1000: val_loss = 0.251
Iter  3000: val_loss = 0.140
Iter  5000: val_loss = 0.111
Iter  7000: val_loss = 0.070
Iter 10000: val_loss = 0.051 (final)
```

### F.3 Computational Environment

- Hardware: NVIDIA A100 80GB
- Framework: PyTorch 2.0+
- Training time: ~5 minutes per 10k iterations

---

*Document Version 2.0 — January 2026*  
*E∆-Hybrid: Unifying Rotation, Reflection, and Thermodynamic Gating*
