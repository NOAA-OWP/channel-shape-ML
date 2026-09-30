---
title: Model Architecture
---

# Model Architecture: Depth

Bankfull depth ($d_{bf}$) constitutes **Stage 2** of our cascaded machine learning parameterization pipeline. In river hydraulics, depth governs cross-sectional conveyance capacity, boundary shear stress ($\tau_b = \gamma R S$), and sediment transport competence.

---

## Cascaded Feature Conditioning (11 Features)

To enforce physical coupling across dimensions, Stage 2 does not predict depth in isolation. Instead, it conditions on the **8 base hydro-geomorphic features** augmented by Stage 1 predicted width and derived geomorphic ratios:

1. **8 Base Hydro-Geomorphic Features**:
   * Sinuosity, Stream Power Index Proxy, Velocity Proxy, Fluvial Concavity Deviation, Floodplain Storage Proxy, Hack's Law Deviation, Relative Position, Upstream Convergence Ratio.
2. **Stage 1 Predicted Bankfull Width ($\hat{W}_{bf}$)**:
   * Direct horizontal scale constraint ensuring cross-sectional continuity.
3. **Width-to-Drainage Ratio ($\text{WDR}$)**:
   * Defined as $\text{WDR} = \frac{\hat{W}_{bf}}{\sqrt{A}}$, capturing channel planform confinement and braiding tendencies.
4. **Width-Slope Product ($\text{WSP}$)**:
   * Defined as $\text{WSP} = \hat{W}_{bf} \cdot S$, acting as a proxy for total cross-sectional stream power per unit downstream reach.

---

## Machine Learning Architecture & Optimization

Stage 2 is parameterized using regularized **Gradient-Boosted Decision Trees (XGBoost)**:

* **Log-Space Optimization**:
  $$y_{\log} = \ln(d_{bf})$$
* **Analytical Variance Bias Correction**:
  $$\hat{d}_{bf} = \exp\left(\hat{y}_{\log} + \frac{\sigma^2_{\text{val}}}{2}\right)$$
* **Non-Linear Threshold Capture**:
  Captures steep mountain step-pool transitions into low-gradient pool-riffle networks without requiring piecewise empirical regimes.

---

!!! info "Upcoming Multi-Model Ensemble"
    The standalone Stage 2 XGBoost model will be integrated with deep residual tabular networks (`PhysicalResTabNet`) in the forthcoming ensemble update.
