---
title: Model Architecture
---

# Model Architecture: Shape Exponent

The cross-sectional shape exponent ($r$) constitutes **Stage 3** of our cascaded parameterization framework. In non-linear hydraulic routing and stage-discharge modeling, $r$ defines how conveyance area ($A$) and hydraulic radius ($R$) grow as water surface elevation rises from low baseflow to bankfull stage.

---

## Dingman Power-Law Geometry Formulation

Following Dingman's generalized cross-sectional power-law representation, channel depth $y(z)$ at lateral offset $z \in [-W_{bf}/2, W_{bf}/2]$ is parameterized by:

$$
y(z) = d_{bf} \left(1 - \left|\frac{2z}{W_{bf}}\right|^r\right)
$$

The dimensionless exponent $r \ge 1.0$ governs channel profile curvature:

* **$r = 1.0$**: Exact triangular channel (steep V-shaped bedrock headwaters).
* **$r = 2.0$**: Classical parabolic profile (equilibrium alluvial channels).
* **$r > 3.0$**: Flat-bottomed rectangular channel (braided or wide cohesive-bed rivers).

---

## Cascaded Feature Conditioning (14 Features)

Stage 3 conditions on the complete upstream geometric cascade:

1. **8 Base Hydro-Geomorphic Features**: Sinuosity, Stream Power Index, Velocity Proxy, Fluvial Concavity Deviation, Floodplain Storage Proxy, Hack's Law Deviation, Relative Position, Upstream Convergence Ratio.
2. **Stage 1 & Stage 2 Cascaded Predictions**: Bankfull Top Width ($\hat{W}_{bf}$) and Bankfull Depth ($\hat{d}_{bf}$).
3. **Derived Cross-Sectional Geometric Ratios**:
   * **Aspect Ratio ($\text{AR}$)**: $\text{AR} = \frac{\hat{W}_{bf}}{\hat{d}_{bf}}$, governing lateral versus vertical shear boundaries.
   * **Hydraulic Scale Index ($\text{HSI}$)**: Composite scaling index relating cross-sectional conveyance to valley slope.
   * **Shear-to-Width Ratio ($\text{SWR}$)**: Ratio quantifying perimeter drag relative to surface width.
   * **Width-to-Drainage Ratio ($\text{WDR}$)**: Planform confinement metric.

---

## Machine Learning Architecture & Optimization

Stage 3 is trained using tuned **Gradient-Boosted Decision Trees (XGBoost)**:

* **Log-Space Transformation**:
  $$y_{\log} = \ln(r - 0.99)$$
  ensuring predicted exponents strictly respect the physical lower bound $r \ge 1.0$.
* **Analytical Bias Correction**:
  $$\hat{r} = 0.99 + \exp\left(\hat{y}_{\log} + \frac{\sigma^2_{\text{val}}}{2}\right)$$

---

!!! info "Upcoming Multi-Model Ensemble"
    Integration of deep tabular residual networks (`PhysicalResTabNet`) with multi-seed epistemic ensembling will be deployed in the upcoming release.
