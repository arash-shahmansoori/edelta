# Comparative Analysis: E∆-MHC-Geo vs. JPmHC

**Date:** March 2026
**Context:** This document provides a precise, cross-referenced comparison between our proposed **E∆-MHC-Geo** architecture and the concurrent work **JPmHC** (Biswa Sengupta, Jinhua Wang & Leo Brunswic, arXiv:2602.18308v2, updated March 4, 2026). Every claim below is intended to be checked against the March v2 arXiv version and our `main.tex`. This document will serve as the basis for incorporating JPmHC into our Related Work and experimental evaluation.

**Note on theorem/equation references:** Our `main.tex` uses section-based numbering for theorems (e.g., Theorem 4.1) and sequential numbering for equations. To avoid ambiguity, this document references theorems by **name and label** (e.g., "the Eigenvalue Exclusion theorem, `thm:exclusion`") rather than compiled numbers.

---

## 0. Convergent Insights (What Both Papers Agree On)

Before detailing the differences, it is important to acknowledge the substantial agreement between the two works:

1. **Diagnosis:** Both papers independently identify that DeepSeek's mHC (Sinkhorn-based doubly stochastic projection onto the Birkhoff polytope) causes gradient instability. JPmHC proves this via operator-valued free probability (eigenvalue contraction and eigenspace misalignment); we demonstrate it empirically (mHC fails catastrophically on stability and near-$\pi$ tasks, as reported in our `main.tex`).
2. **Prescription:** Both papers propose replacing the Birkhoff constraint with an orthogonal constraint via the Cayley transform.
3. **Input-Adaptivity:** Both compute the Cayley parameters dynamically from the input, preserving the expressiveness of Hyper-Connections.

---

## 1. Architectural Topology: Series Pre-Transformation vs. Parallel Routing

The most fundamental structural difference lies in *where* the geometric operator is applied within the Transformer block relative to the compute sub-layer (Attention/MLP).

### JPmHC (Parallel Routing)

JPmHC follows the standard residual paradigm. From their Equation (14):

$$x_{out} = H_{res} \cdot x_{streams} + H_{post} \cdot (y \otimes \mathbf{1}_n)$$

where $y = F(\bar{x}_{in})$ and $\bar{x}_{in}$ is the stream-averaged output of $H_{pre} \cdot x_{streams}$. The residual path ($H_{res}$) and compute path ($H_{pre} \to F \to H_{post}$) operate **in parallel** on the same original input $x_{streams}$, applying *different* transformations ($H_{res}$ vs. $H_{pre}$) to it.

*   **Consequence:** The Attention mechanism sees the input through the lens of $H_{pre}$ (a row-stochastic softmax projection), while the residual shortcut transforms the input through $H_{res}$ (a Cayley-constrained orthogonal matrix). These are geometrically unrelated transforms applied to the same raw input. At the addition point, the model must reconcile two outputs that were computed in incompatible coordinate systems, placing a burden on $H_{post}$ to bridge this representational gap.

### E∆-MHC-Geo (Series Pre-Transformation)

Our architecture applies the geometric operator $\mathcal{G}_\gamma(X)$ **first**, in series, before any stream splitting. From our full-layer transition equation (`eq:full_layer` in `main.tex`):

$$X_{l+1} = \mathcal{G}_\gamma(X_l) + H_{post}^\top F(H_{pre} \cdot \text{LN}(\mathcal{G}_\gamma(X_l)))$$

The geometrically transformed state $X_{geo} = \mathcal{G}_\gamma(X_l)$ is used as **both** the shortcut and the input to the compute path (after LayerNorm).

*   **Advantage ("Alignment Before Compute"):** The Attention and Feed-Forward sub-layers operate on a representation that has already been rotated/reflected into the optimal geometric basis. Both the shortcut and compute paths share the same aligned coordinate system, eliminating the representational misalignment present in JPmHC. This frees attention capacity for feature extraction rather than geometric reconciliation.

---

## 2. Reflection Branch: Boundary Access vs. Finite Cayley Only

### JPmHC (Restricted to a Subset of SO(n))

JPmHC relies exclusively on the Cayley transform for its residual mixer $H_{res}$. As stated in their Appendix A.5: *"This mapping is a diffeomorphism from the space of skew-symmetric matrices to the connected component of O(n) containing the identity (i.e., det = +1), minus the set where I − W/2 is singular."*

Critically, while $SO(n)$ itself can contain matrices with eigenvalue $-1$ (e.g., a rotation by $\pi$ in 3D has eigenvalues $\{1, -1, -1\}$ with $\det = +1$), the **image of the Cayley parametrization** specifically excludes all such matrices. This is precisely our Eigenvalue Exclusion theorem (`thm:exclusion` in `main.tex`): the eigenvalues of the Cayley transform are $\lambda_k = e^{-2i\arctan(\beta\mu_k/2)}$, and since $\arctan: \mathbb{R} \to (-\pi/2, \pi/2)$, the argument lies in $(-\pi, \pi)$, strictly excluding $\pm\pi$. Therefore $\lambda = -1 = e^{i\pi}$ is impossible for any finite parameters.

*   **Consequence:** JPmHC is geometrically incapable of performing exact negation, mirroring, or belief reversal in the latent space via its residual mixer. Their paper does not discuss or acknowledge this limitation.

### E∆-MHC-Geo (Cayley + Householder via Hybrid Gate)

Our architecture introduces a learned operator-selection gate $\gamma$ (the Hybrid Operator definition, `def:hybrid` in `main.tex`) that interpolates between a Cayley rotation ($\det = +1$, excludes $\lambda = -1$) and a Householder reflection fixed at $\beta = 2$ ($\det = -1$, provides $\lambda = -1$). This gives the model access to both connected components of the orthogonal group $O(n)$:

$$\mathcal{G}_\gamma(X) = \gamma(X) \cdot Q(X)X + (1 - \gamma(X)) \cdot H_2(k(X))X$$

Our Asymptotic $O(n)$ Coverage theorem (`thm:coverage` in `main.tex`) proves that as $\gamma \to 1$ the operator lies in $SO(n)$, and as $\gamma \to 0$ it lies in $O(n) \setminus SO(n)$.

*   **Empirical Validation:** On our exact reflection benchmark ($y = -x$), the gate converges to $\gamma \to 0.029$, automatically selecting the Householder branch (reflection results table, `tab:reflection` in `main.tex`). JPmHC has no mechanism to achieve this.

---

## 3. Cayley Transform Mechanics: Exact Rank-2 vs. Iterative Full-Rank Approximation

### JPmHC (Iterative Full-Rank Fixed-Point Approximation)

JPmHC predicts a full $n \times n$ unconstrained matrix $\tilde{H}$, skew-symmetrizes it ($W = \tilde{H} - \tilde{H}^\top$), and then uses a **fixed-point iteration** (their Algorithm 3 in Appendix B.2, described in Section 3.1) to approximate the Cayley retraction:

$$Y_{i+1} = I_n + \frac{\alpha}{2} W(I_n + Y_i), \quad i = 0, \ldots, s-1$$

with step-size $\alpha = 0.1$ and typically $s = 2$ iterations. Crucially, this is an **approximation**. JPmHC's own Proposition I.1 states: *"For Y produced by Algorithm 8, $\|Yx\| \approx \|x\|$ for all $x \in \mathbb{R}^n$, with the approximation improving with more iterations s."* And their Proposition I.2: *"The determinant satisfies $|\det(Y)| \approx 1$, approaching exactness as $s \to \infty$."* They report orthonormality deviation $\|Y^\top Y - I\|_{\max} < 10^{-3}$ at $s = 2$.

*   **Consequence:** The residual mixer is only approximately orthogonal and only approximately norm-preserving, with the quality depending on hyperparameters ($\alpha$, $s$) that must be tuned.

### E∆-MHC-Geo (Exact Rank-2 Analytical Solution)

Our architecture predicts two vectors $\mathbf{u}(\mathbf{x})$ and $\mathbf{v}(\mathbf{x})$ and constructs a **Rank-2** skew-symmetric generator:

$$\mathbf{A} = \mathbf{u}\mathbf{v}^\top - \mathbf{v}\mathbf{u}^\top$$

The Cayley transform $Q = (I + \frac{\beta}{2}A)^{-1}(I - \frac{\beta}{2}A)$ is then computed **exactly** via `torch.linalg.solve` (a direct linear system solver). At $n = 4$ streams, this is a 4×4 system that is trivially fast.

*   **Advantage:** No approximation error, no hyperparameters to tune ($\alpha$, $s$), and deterministic output. Our Unconditional Orthogonality theorem (`thm:orthogonality` in `main.tex`) guarantees $Q^\top Q = I$ exactly (up to floating-point precision), not approximately.
*   **Trade-off (acknowledged):** The Rank-2 generator restricts each Cayley operator to rotate in a single 2D plane per application (with the remaining $n - 2$ dimensions fixed). JPmHC's full-rank $W$ can rotate simultaneously in multiple planes. However, since we apply the geometric operator **twice per block** (once before attention, once before MLP, as shown in Figure 4 of `main.tex`), and across $L = 6$ layers, the effective rotation coverage is 12 potentially distinct 2D plane rotations. For $n = 4$ streams, the maximum number of orthogonal 2D rotation planes is $n(n-1)/2 = 6$, so 12 applications provide more than sufficient coverage.

---

## 4. Stream Aggregation: Learned Weight Matrices vs. Per-Token Dynamic Softmax

### JPmHC (Per-Token Dynamic Generation)

JPmHC generates all three mixing matrices ($H_{pre}$, $H_{post}$, $H_{res}$) dynamically for every token via a single fused linear projection from the flattened stream representation (their Equation 12):

$$[\tilde{H}_{pre} | \tilde{H}_{post} | \tilde{H}_{res}] = W_{fused} \cdot \text{LayerNorm}(x_{flat})$$

$H_{pre}$ is then projected to row-stochastic via `softmax(dim=-1)`, $H_{post}$ to column-stochastic via `softmax(dim=-2)`, and $H_{res}$ to orthogonal via the iterative Cayley transform. All three $n \times n$ matrices change with every token.

### E∆-MHC-Geo (Learned Weight Matrices + Dynamic Geometric Operator)

Our pre/post mappings ($H_{pre}$, $H_{post}$) are standard `nn.Linear` layers with learned weight matrices that are shared across all tokens (initialized to identity, as shown in our code in Section 8 of `main.tex`). The weight matrices are updated during training but are **not** dynamically generated per token.

Instead, all per-token dynamic routing is concentrated in the geometric operator $\mathcal{G}_\gamma$, which computes input-dependent $\mathbf{u}(\mathbf{x})$, $\mathbf{v}(\mathbf{x})$, $\beta(\mathbf{x})$, $\mathbf{k}(\mathbf{x})$, and $\gamma(\mathbf{x})$ via dedicated neural networks.

*   **Design Philosophy:** Our approach cleanly separates concerns: the pre/post mappings handle stable structural stream aggregation/broadcasting, while the geometric operator handles all input-adaptive geometric transformations. JPmHC entangles both roles into a single fused projection, which may increase flexibility but also increases the risk of instability and the parameter overhead per token.

---

## 5. Midpoint Collapse Regularization (Absent in JPmHC)

A contribution in our paper that has no counterpart in JPmHC is the **midpoint collapse regularization** and its theoretical analysis.

Because our Hybrid interpolates between two disconnected components of $O(n)$, the blended operator at intermediate $\gamma \in (0, 1)$ is not itself orthogonal (our Non-Orthogonality at Midpoint theorem, `thm:midpoint`). We introduce the regularization $\mathcal{L}_{gate} = 4\gamma(1 - \gamma)$ to force binary gate decisions, and prove the **Universal Zero-Gradient theorem** (`thm:universal_zero`): any smooth, symmetric regularizer has zero gradient at $\gamma = 0.5$, making this a fundamental mathematical limitation, not a design flaw. We further identify the escape mechanisms (task-loss gradient, input variation, biased initialization) that allow the model to break out of this critical point.

JPmHC has no analogous mechanism because it does not attempt to combine operators from different connected components of $O(n)$. This means JPmHC also has no theoretical framework for understanding when and why a model should switch between geometric strategies.

---

## 6. JPmHC's Unique Theoretical Strengths (Fair Acknowledgment)

While our architecture is more complete, JPmHC makes genuine theoretical contributions that our paper does not:

1. **Operator-Valued Free Probability Analysis:** JPmHC develops a rigorous spectral diagnosis of why doubly stochastic skip connections fail. Their operator-valued Dyson equation (Proposition 2.1) proves that eigenvalue contraction and eigenspace misalignment in the Birkhoff polytope cause partial spectral collapse of the end-to-end Jacobian. This is a novel and powerful analytical tool that provides complementary theoretical evidence for the same conclusion we demonstrate empirically.

2. **Implicit Differentiation for Sinkhorn:** Their custom backward pass (Appendix H) reduces Sinkhorn backpropagation memory from $O(T)$ to $O(1)$ via implicit differentiation at the fixed point, eliminating DDP synchronization stalls. This is a practical engineering contribution for anyone still using Sinkhorn-based projections.

3. **Grassmannian and Permutation Variants:** JPmHC proposes additional mixer variants (Grassmannian subspace projection, spectral-gap-optimal permutations) that we do not explore. These remain untrained at scale in their paper (Section 7.4) but represent interesting future directions.

Our paper should cite these contributions fairly while emphasizing that JPmHC's Cayley implementation is architecturally incomplete (no reflection branch, no regularization theory, approximate orthogonality, parallel rather than series integration).

### Important Caveats About JPmHC's Reported Results

Two caveats from JPmHC's own paper bear noting:

*   **Training is ongoing:** JPmHC Section 5 explicitly states *"Training is ongoing."* The Cayley variant has been trained for ~419K steps and the Sinkhorn variant for ~349K steps. Their reported numbers (31.4% exact-match, 40.5% pass@1) may improve with further training.
*   **Pre/post architecture confound:** JPmHC Section 7.4 acknowledges that their Cayley and Sinkhorn variants *"differ in pre/post normalization and mapping architecture, making it impossible to attribute the entire performance gap to the manifold choice alone."* Specifically, their Cayley variant uses `LayerNorm` + `softmax` for pre/post mappings while Sinkhorn uses `RMSNorm` + `sigmoid` (their Table 1).

---

## 7. Summary Comparison Table

| Dimension | **JPmHC** | **E∆-MHC-Geo** |
|:----------|:----------|:----------------|
| **Block topology** | Parallel routing ($H_{res}$ shortcut ‖ $H_{pre} \to F \to H_{post}$) | Series pre-transformation ($\mathcal{G}_\gamma$ applied first, then split) |
| **Cayley generator** | Full-rank $W = \tilde{H} - \tilde{H}^\top$ | Rank-2 $A = uv^\top - vu^\top$ |
| **Cayley computation** | Iterative fixed-point approximation ($s = 2$, $\alpha = 0.1$) | Exact analytical solve (`torch.linalg.solve`) |
| **Orthogonality** | Approximate ($\|Y^\top Y - I\|_{\max} < 10^{-3}$) | Exact ($Q^\top Q = I$, `thm:orthogonality`) |
| **Eigenvalue $-1$** | Excluded (not discussed in their paper) | Excluded by Cayley, recovered via Householder gate |
| **O(n) coverage** | Cayley image within $SO(n)$ only ($\det = +1$) | Full $O(n)$ ($\det \in \{-1, +1\}$, `thm:coverage`) |
| **Regularization theory** | None | Midpoint collapse + Universal Zero-Gradient theorem |
| **Pre/post mappings** | Per-token dynamic (fused linear + softmax) | Learned weight matrices (shared across tokens) |
| **Spectral theory** | Operator-valued Dyson equation (strong contribution) | Empirical validation (470× norm preservation) |
| **Evaluation domain** | ARC-AGI (discrete logic, 7M params) | Synthetic geometric benchmarks (~1.79M params) |
| **Parameter scale** | ~7M (TRM backbone) | ~1.79M (custom Transformer) |
| **Training status** | Ongoing (~419K steps) | Complete (2000 iterations) |

---

## 8. Optional Future Evaluation: ARC-AGI

### Motivation

While our continuous synthetic benchmarks (Gyroscope, Stability, Reflection, Near-$\pi$) precisely validate individual mathematical theorems, ARC-AGI could be considered later as a downstream stress test on complex reasoning tasks. It should not be part of the main paper comparison unless a full, matched, multi-seed experiment is completed.

### Why ARC-AGI is Ideal for Demonstrating Our Advantages

ARC-AGI (Chollet, 2019) measures fluid intelligence through visual logic puzzles. Each task presents demonstration input-output grid pairs, and the model must infer the latent transformation rule and produce the exact output grid. Crucially:

1. **Many tasks require exact geometric transforms:** Spatial mirroring, flipping, transposition, and pattern inversion are common ARC-AGI operations. These are precisely the operations where JPmHC's inability to produce eigenvalue $-1$ in its residual mixer becomes a concrete limitation, forcing the Attention/MLP path to compensate for what the residual path cannot express.

2. **Exact-match evaluation amplifies geometric precision:** ARC-AGI uses all-or-nothing exact-match scoring—even a single wrong cell means failure. This directly rewards the superior norm preservation and manifold precision of E∆-MHC-Geo.

3. **Recursive weight tying amplifies approximation error:** JPmHC's TRM backbone applies the same 2 unique blocks recursively 12 times. At each recursive pass, JPmHC's iterative Cayley approximation introduces a small orthonormality deviation ($\|Y^\top Y - I\|_{\max} < 10^{-3}$). Over 12 recursive applications of the same mixer, these deviations **compound**, gradually degrading the spectral properties that orthogonality was supposed to guarantee. Our exact analytical Cayley solve ($Q^\top Q = I$ exactly) does not compound any such error, providing a strictly better foundation for deep recursion.

### Text-Based Representation (No Vision Pipeline Required)

ARC-AGI grids are rectangular matrices of integers 0-9 (representing colors), with dimensions up to 30×30 (JPmHC Section 4.1). They do **not** require CNNs or Vision Transformers. For a standard text-based Transformer, the grids are serialized into 1D token sequences.

JPmHC uses the TRM backbone (Jolicoeur-Martineau, 2025), which already handles this serialization. While JPmHC does not describe the exact tokenization format, the standard approach treats grid cells as discrete tokens from a vocabulary of 10 color values plus special tokens for row/grid delimiters and task structure. The task is formulated as autoregressive sequence-to-sequence prediction:

*   **Input prompt:** Serialized demonstration pairs (Input Grid → Output Grid), followed by the test Input Grid.
*   **Target:** The serialized test Output Grid.
*   **Loss:** Standard cross-entropy (next-token prediction) on the output grid tokens. JPmHC reports this as "stablemax cross-entropy" (their Section 4.4).
*   **Evaluation:** Greedy decode of the output grid, then exact-match comparison against ground truth (cell-by-cell, including grid dimensions).

This pure sequence formulation allows us to drop our E∆-MHC-Geo Hybrid block directly into the TRM backbone, replacing JPmHC's parallel-routing Cayley module with our series pre-transformation Hybrid module.

### Experimental Plan

1. **Backbone:** Adopt the TRM architecture (Jolicoeur-Martineau, 2025) with ~7M parameters, $n = 4$ streams, hidden dim $d = 512$ (effective dim $nd = 2048$), 8 attention heads, 2 unique weight-tied blocks applied 6 times each (12 total recursive passes with Adaptive Computation Time halting). Source: JPmHC Table 10.
2. **Mixer Replacement:** Replace JPmHC's parallel Cayley module with our E∆-MHC-Geo Hybrid block (series pre-transformation, exact Rank-2 Cayley + Householder gate + midpoint collapse regularization).
3. **Training:** Match JPmHC's training configuration: AdamAtan2 optimizer, lr $10^{-4}$, global batch size 768, weight decay 0.1, gradient clipping 1.0, bfloat16 mixed precision, ~400K steps on ARC-AGI-1 (400 training tasks, 400 evaluation tasks, 200 ablation/validation). Source: JPmHC Tables 10--11 and Section 4.1.
4. **Hardware:** JPmHC uses 8× NVIDIA B200 192GB GPUs with PyTorch DDP + `torch.compile` (their Table 11). Comparable GPU resources will be required.
5. **Metrics:** Report exact-match accuracy (greedy), pass@$k$ ($k \in \{1, 2, 5, 10, 100, 1000\}$), and eval LM loss. Compare against JPmHC's reported best values: 31.4% exact-match (step 418,854) and 40.5% pass@1 (step 380,755). Note: these best values occur at different checkpoints.
6. **Gate Analysis:** Log per-layer gate values $\gamma$ during inference on individual ARC-AGI tasks. If tasks involving spatial mirroring or inversion show $\gamma \to 0$ (Householder selection) while rotation/translation tasks show $\gamma \to 1$ (Cayley selection), this will be evidence of automatic operator selection on a real-world reasoning benchmark.

### Expected Outcomes

We hypothesize that E∆-MHC-Geo will outperform JPmHC on ARC-AGI for three reasons:
1. **Exact orthogonality** does not compound approximation error under 12-fold recursive weight tying, unlike JPmHC's iterative approximation.
2. **Householder reflection** enables the model to represent mirror/negation operations directly in the residual path, which pure Cayley cannot express.
3. **Series pre-transformation** provides geometrically aligned representations to the Attention mechanism, improving feature extraction on spatial reasoning tasks.

These hypotheses remain to be validated experimentally. The synthetic benchmark results provide theoretical grounding for considering this direction, but ARC-AGI performance depends on many additional factors (tokenization quality, training dynamics, optimizer interaction) that require empirical testing. For the current paper, ARC-AGI should be referenced only as a possible future benchmark, not as evidence for the proposed architecture.