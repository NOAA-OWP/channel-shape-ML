---
title: Model Architecture
---

# Model Architecture

The machine learning architecture for **n-out** represents the compound floodplain component of Stage 4 in our parameterization cascade.

---

## Target Derivation via Chow's Compound Separation

The overbank target variable $n_{\text{out}}$ is derived during preprocessing using Chow's compound channel method. By segmenting observed stage-discharge curves at the bankfull depth threshold ($d_{bf}$), in-channel conveyance is analytically subtracted:

$$
n_{\text{out}} = \frac{A_{\text{out}} R_{\text{out}}^{2/3} S^{1/2}}{Q_{\text{total}} - Q_{\text{in}}}
$$

This isolates pure floodplain resistance, preventing the overbank model from being biased by channel bed friction.

---

## Architecture Reuse
Because the fundamental geomorphic drivers (slope, drainage area, topology, hydraulic radius) govern both bulk and in-channel resistance, we retain the hybrid ensemble:
(a) Attention-Augmented Deep Residual Tabular Network
(b) Gradient-Boosted Decision Forest

The models learn a distinct mapping tailored to main-channel bedforms and grain friction without being confounded by floodplain vegetation.
