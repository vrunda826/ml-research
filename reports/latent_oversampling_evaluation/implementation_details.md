# Implementation Details: Controlled Latent-Space Oversampling Suite

**Author:** Antigravity Research Engineering  
**Date:** October 15, 2026  
**Document Purpose:** Detailed technical specification and mathematical formalization of experimental conditions E0–E4, representations, neighborhood mechanisms, leakage prevention guarantees, and threshold policies.

---

## 1. Experimental Conditions Overview

| Experiment ID | Condition Name | Representation / Latent Space | Oversampling Mechanism | Downstream Model | Purpose |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **E0** | Primary Baseline (B3) | Original standardized input space ($D=462$) | None | Logistic Regression ($C=1.0$) & Ridge ($\alpha=100.0$) | Benchmark anchor |
| **E1** | Ordinary Random Oversampling | Original standardized input space ($D=462$) | Random replication with replacement of minority training samples | Logistic Regression ($C=1.0$) & Ridge ($\alpha=100.0$) | Measure simple class balancing |
| **E2a** | Linear Latent-Space Oversampling | Truncated SVD / PCA latent manifold ($d=32$) | $k$-NN interpolation between positive training latent vectors | Logistic Regression & Ridge on latent representations | Test well-regularized linear manifold oversampling |
| **E2b** | Neural Latent-Space Oversampling | Regularized Autoencoder bottleneck ($d=32$) | $k$-NN interpolation between positive training latent vectors | Logistic Regression & Ridge on latent representations | Test non-linear neural manifold oversampling |
| **E3** | Oversampling Sensitivity Grid | Best representation from E2 (Linear vs Neural) | Grid across target ratios $\alpha \in \{0.35, 0.50, 0.65\}$ and $k \in \{1, 3, 5\}$ | Downstream linear models on augmented latent space | Evaluate hyperparameter stability via training-only CV |
| **E4** | Constrained Latent-Space Variant | Latent space with centroid-distance constraint | Manifold-constrained interpolation (rejecting points exceeding $2\sigma$ from positive centroid) | Downstream linear models on augmented latent space | Test whether bounding synthetic drift improves precision |

---

## 2. Mathematical Formulation of Latent-Space Oversampling

Let the training set be $\mathcal{D}_{\text{train}} = \{(x_i, y_{b,i}, y_{c,i})\}_{i=1}^{N_{\text{tr}}}$, where $x_i \in \mathbb{R}^{D}$ ($D=462$), $y_{b,i} \in \{0, 1\}$, and $y_{c,i} \in [0, 24]$.  
Let $\mathcal{P} = \{i : y_{b,i} = 1\}$ denote positive participants ($|\mathcal{P}| = 50$) and $\mathcal{N} = \{i : y_{b,i} = 0\}$ denote negative participants ($|\mathcal{N}| = 113$).

### 2.1 Latent Space Mapping
An encoder $f_\theta: \mathbb{R}^D \to \mathbb{R}^d$ ($d \ll D$, here $d=32$) maps standardized input features into latent vectors:
$$z_i = f_\theta(x_i)$$
- In **E2a (Linear)**: $f_\theta$ is computed via Truncated Singular Value Decomposition (PCA) fit strictly on the training partition: $X_{\text{tr}} = U \Sigma V^T$, $z_i = x_i V_d$.
- In **E2b (Neural)**: $f_\theta$ is a regularized autoencoder trained with MSE reconstruction loss and weight decay on training participants only:
  $$\min_{\theta, \phi} \frac{1}{N_{\text{tr}}} \sum_{i=1}^{N_{\text{tr}}} \|x_i - g_\phi(f_\theta(x_i))\|_2^2 + \lambda_{\text{reg}} (\|\theta\|_2^2 + \|\phi\|_2^2)$$

### 2.2 Synthetic Sample Generation in Latent Space
For a target positive ratio $\alpha \in (0, 1)$ (default $\alpha = 0.50$), the number of synthetic samples required is:
$$N_{\text{synth}} = \max\left(0, \left\lceil \frac{\alpha \cdot |\mathcal{N}|}{1 - \alpha} \right\rceil - |\mathcal{P}|\right)$$
For $\alpha = 0.50$, $N_{\text{synth}} = 113 - 50 = 63$.

For each synthetic sample $m \in \{1, \dots, N_{\text{synth}}\}$:
1. **Anchor Selection:** An anchor participant $i \in \mathcal{P}$ is sampled uniformly at random from positive training participants: $i \sim \text{Uniform}(\mathcal{P})$.
2. **Neighbor Selection:** Among the remaining positive training participants $\mathcal{P} \setminus \{i\}$, compute Euclidean distances in latent space:
   $$d_{ij} = \|z_i - z_j\|_2, \quad \forall j \in \mathcal{P} \setminus \{i\}$$
   Identify the $k$ nearest positive neighbors $\mathcal{K}_k(i)$. Choose one neighbor $j \sim \text{Uniform}(\mathcal{K}_k(i))$.
3. **Interpolation Coefficient:** Sample an interpolation weight $\lambda_m \sim \text{Uniform}(0.15, 0.85)$ to avoid degenerate boundary duplication.
4. **Synthetic Point Construction:**
   - Synthetic latent vector:
     $$z_{\text{synth}, m} = z_i + \lambda_m (z_j - z_i)$$
   - Continuous PHQ-8 target:
     $$y_{c, \text{synth}, m} = y_{c, i} + \lambda_m (y_{c, j} - y_{c, i})$$
   - Binary label:
     $$y_{b, \text{synth}, m} = 1$$
5. **Provenance Tracking:** Record anchor ID, neighbor ID, $\lambda_m$, and distance $\|z_i - z_j\|_2$ in `synthetic_sample_audit.csv`.

---

## 3. Strict Anti-Leakage Safeguards

Data leakage is strictly prohibited through the following verified controls:

1. **Fold-Internal Representation Fitting:** Inside every cross-validation fold $k \in \{1, \dots, 5\}$:
   - The scaler (`StandardScaler`) is fitted strictly on fold training data $X_{\text{tr}, k}$.
   - The encoder $f_\theta$ (PCA or Autoencoder) is fitted strictly on fold training data $X_{\text{tr}, k}$.
   - Validation fold data $X_{\text{val}, k}$ is transformed using the fold-specific fitted scaler and encoder.
2. **Untouched Validation Participants:**
   - Neither features nor labels of validation participants are ever present in the interpolation candidate pool $\mathcal{P}$.
   - Synthetic samples are generated strictly inside fold $k$'s training space.
3. **Development Isolation:**
   - The official development set ($N=56$) is transformed strictly using the global training-fitted encoder ($N=163$).
   - Zero development participants are ever interpolated, decoded, or used for parameter selection.

---

## 4. Threshold Policy

To ensure fairness between uncalibrated probabilities and oversampled models:
1. **Fixed Standard Threshold:** Primary evaluation reports the conventional cutoff:
   $$\hat{y}_b = \mathbb{I}(P(y=1|x) \ge 0.50)$$
2. **Training-OOF Tuned Threshold:** To account for probability shift induced by oversampling (which naturally inflates predicted probabilities), we compute 5-fold training out-of-fold (OOF) probabilities for each model and select the threshold $\tau^* \in [0.10, 0.90]$ that maximizes training OOF Macro-F1:
   $$\tau^* = \arg\max_{\tau} \text{Macro-F1}(\mathbf{y}_{\text{tr}}, \mathbb{I}(\mathbf{p}_{\text{tr, OOF}} \ge \tau))$$
   This frozen threshold $\tau^*$ is then applied to the development set predictions. Both fixed ($0.50$) and training-tuned ($\tau^*$) metrics are reported side-by-side.

---

## 5. Statistical Bootstrap Protocol

All primary comparisons on the official development set ($N=56$) are evaluated using paired participant-level bootstrap resampling:
- **Bootstrap Replicates:** $B = 1,000$ iterations.
- **Random Seed:** Fixed seed $42$.
- **Sampling:** Resamples participants with replacement ($N=56$).
- **Metric Difference:** For each replicate $b$, $\Delta M^{(b)} = M_{\text{model}}^{(b)} - M_{\text{baseline}}^{(b)}$.
- **Confidence Intervals:** Empirical 95% interval $[q_{0.025}(\Delta M), q_{0.975}(\Delta M)]$.
- **Significance Criterion:** Two-sided empirical $p$-value $p = 2 \cdot \min(\mathbb{P}(\Delta M \ge 0), \mathbb{P}(\Delta M \le 0))$. Significance is claimed only if $p < 0.05$ and the 95% CI strictly excludes zero.
