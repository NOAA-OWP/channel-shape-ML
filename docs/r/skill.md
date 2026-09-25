---
title: Model Skill
---

# Model Skill: Shape Exponent (Stage 3)

---

## Prediction Performance

![Observed vs. Predicted Dingman Exponent r](../assets/images/r/v2.0.0/skill/fit.png){ loading=lazy }

The 3-panel scatter evaluation illustrates model performance across Training, Validation, and Testing partitions. Predictions show strong agreement along the 1:1 line across the continuous shape spectrum, effectively capturing transitions from narrow steep channels ($r \approx 1.2$) to broad parabolic profiles ($r \approx 2.5$).

---

## Cumulative Distribution Alignment (CDF)

![CDF Distribution: Observed vs. Predicted Shape Exponent r](../assets/images/r/v2.0.0/skill/bankfull_shaper_cdf_distribution.png){ loading=lazy }

The cumulative distribution function (CDF) comparison demonstrates precise quantile matching between field-measured cross-sectional surveys and machine learning predictions, preserving the empirical distribution of continental river channel geometries.

---

## Regional Diagnostics

![HLR Performance Breakdown](../assets/images/r/v2.0.0/skill/bankfull_shaper_hlr_performance.png){ loading=lazy }

Performance stratified across **Hydrologic Landscape Regions (HLRs)** demonstrates consistent accuracy:

- **Montane & High-Gradient Reaches**: Strong skill in identifying low-exponent ($r < 1.5$) V-shaped bedrock profiles where steep valley confinement restricts lateral widening.
- **Lowland Alluvial Basins**: Accurate capture of equilibrium parabolic profiles ($r \approx 1.8\text{--}2.2$) with smooth transition into wide depositional floodplains.

![HLR Choropleth Map](../assets/images/r/v2.0.0/skill/bankfull_shaper_hlr_choropleth.png){ loading=lazy }

The continental choropleth map confirms spatial consistency across all 20 HLR units without regional boundary artifacts.

---

## Stream Order Bias Analysis

![Residual Bias by Stream Order](../assets/images/r/v2.0.0/training_bias/bankfull_shaper_residual_bias_by_streamorder.png){ loading=lazy }

Evaluating prediction residuals against Strahler stream orders (1 through 9) reveals a zero-centered residual profile across network scales, confirming that the Stage 3 model maintains consistent geometric fidelity across both headwater tributaries and large continental river channels.

---

!!! info "Upcoming Neural Loss Curves"
    Training loss and validation convergence curves for the deep tabular neural network (`PhysicalResTabNet`) will be added with the deployment of the hybrid multi-model ensemble.
