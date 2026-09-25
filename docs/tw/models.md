---
title: Model Architecture
---

# Model Architecture: TopWidth

Bankfull top width ($W_{bf}$) forms **Stage 1** of our cascaded machine learning parameterization framework. As the primary horizontal scaling dimension of river systems, accurate width estimation anchors all subsequent downstream predictions (Depth, Shape, and Manning's Roughness).

---

## Machine Learning Architecture

For Stage 1, bankfull width is parameterized using tuned **Gradient-Boosted Decision Trees (XGBoost)**:

1. **Non-Linear Geomorphic Partitioning**:
   * Tree-based partitioning naturally captures sharp geomorphic regime transitions (e.g., bedrock-confined canyons vs. unconfined alluvial floodplains).
   * Regularized gradient boosting minimizes structural variance while capturing complex feature interactions between drainage scale, valley slope, and local flow velocity proxies.

2. **Log-Space Optimization**:
   * Because river widths span four orders of magnitude (from meter-scale headwaters to kilometer-scale lowland mainstems) and are strictly positive, models are optimized in log-transformed space:
     $$y_{\log} = \ln(W_{bf})$$
   * To prevent re-transformation bias when converting predictions back to physical units, an analytical variance correction is applied:
     $$\hat{W}_{bf} = \exp\left(\hat{w}_{\log} + \frac{\sigma^2_{\text{val}}}{2}\right)$$

---

## Input Feature Conditioning (8 Base Features)

The Stage 1 model conditions strictly on the **8 base hydro-geomorphic features**:

1. **Stabilized Sinuosity ($S_{\text{stabilized}}$)**: Meander intensity with short-reach topological smoothing.
2. **Stream Power Index Proxy ($\text{SPI}_{\text{actual}}$)**: Kinetic energy potential driving channel widening ($A \cdot S$).
3. **Bankfull Velocity Proxy ($V_{bf,\text{proxy}}$)**: Reach-scale conveyance speed and boundary energy.
4. **Fluvial Concavity Deviation ($\text{FCD}$)**: Departure from Flint's Law equilibrium slope ($S = k_s A^{-0.45}$).
5. **Floodplain Storage Proxy ($\text{FS}_{\text{proxy}}$)**: Lateral room for valley storage and energy dissipation.
6. **Hack's Law Deviation ($\text{HD}$)**: Basin elongation and hydrograph concentration timing.
7. **Relative Position ($\text{RP}$)**: Dimensionless position along the network from headwater to outlet.
8. **Upstream Convergence Ratio ($\text{UCR}$)**: Network confluence identification for sediment and discharge surges.

---

!!! info "Upcoming Multi-Model Ensemble"
    While Stage 1 currently utilizes regularized XGBoost, the upcoming release will integrate a multi-seed hybrid ensemble combining XGBoost with attention-augmented deep tabular residual networks (`PhysicalResTabNet`).
