---
title: Depth (Y) — Model Skill & Continental Evaluation
---

# Model Skill & Continental Evaluation

> **Publication Reference**: Modaresi Rad, A., et al. (2024). *Enhancing River Channel Dimension and Bathymetry Estimates Across Continental Scale Using Machine Learning and Functional Hydraulic Geometry*. **Journal of Geophysical Research: Machine Learning and Computation**, 1(3), e2024JH000173.

---

## Evaluation Framework & Goodness-of-Fit Metrics

Model performance for river depth ($Y$) and Functional Hydraulic Geometry parameters ($f$ and $c$) is evaluated on out-of-sample testing sets comprising thousands of quality-screened USGS streamgages and HYDRoSWOT Acoustic Doppler Current Profiler (ADCP) survey cross-sections.

To evaluate non-linear hydraulic behavior robustly across diverse river scales, we employ multiple complementary goodness-of-fit criteria:

### 1. Normalized Nash-Sutcliffe Efficiency (NNSE)

Standard Nash-Sutcliffe Efficiency ($NSE \in (-\infty, 1]$) is sensitive to extreme outliers and bounded asymmetrically. We utilize the Normalized Nash-Sutcliffe Efficiency ($NNSE \in [0, 1]$) proposed by Nossent & Bauwens (2012):

$$
NNSE = \frac{1}{2 - NSE}, \quad \text{where } NSE = 1 - \frac{\sum_{i=1}^N (Y_{\text{obs}, i} - Y_{\text{sim}, i})^2}{\sum_{i=1}^N (Y_{\text{obs}, i} - \bar{Y}_{\text{obs}})^2}
$$

* $NNSE = 1.0$: Perfect agreement between simulated and observed depths.
* $NNSE = 0.5$: Simulation performance equal to the observed mean ($NSE = 0.0$).
* $NNSE > 0.65$: High predictive skill suitable for hydraulic routing and hydrodynamic inundation modeling.

### 2. Kling-Gupta Efficiency (KGE)

Decomposes prediction error into correlation ($\rho$), variability ratio ($\alpha = \sigma_{\text{sim}}/\sigma_{\text{obs}}$), and bias ratio ($\beta = \mu_{\text{sim}}/\mu_{\text{obs}}$):

$$
KGE = 1 - \sqrt{(\rho - 1)^2 + (\alpha - 1)^2 + (\beta - 1)^2}
$$

---

## Continental Station Evaluation

The map below illustrates Normalized Nash-Sutcliffe Efficiency ($NNSE$) scores for predicting channel depth using the FHG exponent $f$ model across all out-of-sample test stations with $>50$ paired discharge-depth measurements:

![Continental Station NNSE for Channel Depth Parameter f](../../assets/images/v1.0.0/y/Best_Model_Sub_f_cMAP_f_Y_Testing_NNSE.png){ loading=lazy }

---

## Multi-Tier Ensemble Evaluation: CDF Analysis

Cumulative Distribution Functions (CDFs) of Normalized Nash-Sutcliffe Efficiency ($NNSE$) scores were evaluated across all independent testing stations:

![CDF Comparison of NNSE Scores for Depth Parameter f across Ensemble Tiers](../../assets/images/v1.0.0/y/Best_Model_norm_f_CDF_f_Y_Testing_NNSE.png){ loading=lazy }

### CDF Performance Highlights

* **Continental Median Skill**: The median station performance reaches **$\text{NNSE} \approx 0.94$** (50th percentile red dashed marker).
* **Reliability Threshold ($\text{NNSE} \ge 0.66$)**: Over **$88\%$ of all testing stations** exceed the $\text{NNSE} = 0.66$ benchmark (blue dashed marker) for high-accuracy hydrodynamic modeling.
* **Top-Tier Precision**: Over **$50\%$ of stations** achieve $\text{NNSE} \ge 0.94$, reflecting near-perfect depth rating curve representation.

---

## Maximum Flow ($Q_{\max}$) Diagnostic Evaluation

A critical test of empirical and machine-learned hydraulic geometry is performance during extreme peak discharges. Depth predictions were evaluated specifically at the **maximum observed historical discharge ($Q_{\max}$)** for each testing station:

![Actual vs Predicted Depth at Maximum Observed Flow (Qmax)](../../assets/images/v1.0.0/y/Best_Model_Sub_f_scatter_type1_Y_act_max_f_Y_Testing.png){ loading=lazy }

### Diagnostics at Extreme Flows

1. **Exceptional Correlation ($R^2 = 0.93$)**: As shown in the diagnostic scatter, the model achieves **$R^2 = 0.93$** along the 1:1 line at peak historical flow stages ($Q_{\max}$).
2. **Stream Scale Consistency**: High linearity is maintained across small upland reaches and large alluvial mainstems without systematic over- or under-prediction.
3. **No Saturation Plateauing**: The power-law formulation ($Y = c \cdot Q^f$) allows continuous, physically unbounded depth expansion during extreme flood stages.

---

## Performance Stratified by Environmental Quantiles

To verify model stability across continental environmental gradients, goodness-of-fit was evaluated across quartiles/quantiles of major physical drivers:

![Model Performance Across Quantiles of Influential Environmental Variables](../../assets/images/v1.0.0/y/Fig5_y.png){ loading=lazy }
*Figure 2: Performance metrics ($R^2$ and depth distributions) evaluated across quantiles of channel slope, bankfull discharge, elevation, and base flow index ([Modaresi Rad et al., 2024](https://doi.org/10.1029/2024JH000173)).*

### Quantile Performance Details

* **Channel Slope**: $R^2$ ranges from $0.65$ in moderate gradient streams to $0.86$ in low-slope valleys ($0.0- 0.001$).
* **Bankfull Discharge ($Q_{\text{bf}}$)**: $R^2$ increases steadily with stream scale, from $0.32$ in headwater trickles ($0 - 53\text{ m}^3/\text{s}$) to $0.76$ in large channels ($303 - 15,466\text{ m}^3/\text{s}$).
* **Elevation**: $R^2$ ranges from $0.67$ to $0.86$, peaking ($R^2 \approx 0.86$) across interior lowland plains ($121 - 216\text{ m}$).
* **Base Flow Index (BFI)**: Strong predictive skill ($R^2 = 0.70 - 0.86$) across all baseflow contribution regimes.

---

## Literature Benchmarking & Comparative Evaluation

We benchmarked the v1.0.0 Depth model against published continental regional curves and recent machine learning frameworks:

### 1. Regional Regression Baseline: Blackburn-Lynch et al. (2017) Across 20 HLRs

[Blackburn-Lynch et al. (2017)](https://doi.org/10.1111/1752-1688.12567) developed regional power-law curves stratified across the 20 **Hydrologic Landscape Regions (HLRs)** of the United States.

![Performance Comparison with Blackburn-Lynch et al. (2017) Across 20 HLRs](../../assets/images/v1.0.0/y/Fig7.png){ loading=lazy }
*Figure 3: Goodness-of-Fit comparison between the proposed ML model and Blackburn-Lynch et al. (2017) regional regressions across all 20 Hydrologic Landscape Regions ([Modaresi Rad et al., 2024](https://doi.org/10.1029/2024JH000173)).*

#### Regional Performance Across HLRs (Figure 7e)

* **Proposed ML Model (Grey Bars)**: $R^2$ maintains high stability between **$0.59$** (HLR 19) and **$0.78$** (HLR 15), with continental $\text{median } R^2 \approx 0.68$.
* **Blackburn-Lynch (Black Dots)**: Shows substantial regional instability with negative and near-zero $R^2$ in multiple regions ($\text{median } R^2 \approx 0.35$).
* The proposed ML model **statistically outperforms Blackburn-Lynch regional equations across 19 of the 20 HLRs**.

---

### 2. Global Equations & Modern ML Benchmarks: Andreadis (2013) & Doyle (2023)

The proposed ML model was benchmarked against global discharge-based depth equations ([Andreadis et al., 2013](https://doi.org/10.1002/wrcr.20440)) and modern machine learning models ([Doyle et al., 2023](https://doi.org/10.1029/2022WR033621)):

![Benchmarking Against Global Equations and Doyle et al. (2023) ML Models](../../assets/images/v1.0.0/y/Fig10.png){ loading=lazy }
*Figure 4: Scatter distributions and Goodness-of-Fit metrics ($R^2, \text{KGE}, \text{NRMSE}$) comparing the proposed ML model against Andreadis et al. (2013) and Doyle et al. (2023) ([Modaresi Rad et al., 2024](https://doi.org/10.1029/2024JH000173)).*

---

### 3. Physiographic Division & Province Evaluation (Figure S13)

Model generalizability was evaluated across US Physiographic Divisions (8 divisions) and Physiographic Provinces (23 provinces):

![Model Performance Partitioned by Physiographic Provinces and Divisions](../../assets/images/v1.0.0/y/Fig_S13.png){ loading=lazy }
*Figure 5: Performance comparison of the proposed ML model against Blackburn-Lynch et al. (2017) and Bieger et al. (2015) across Physiographic Provinces and Divisions ([Modaresi Rad et al., 2024](https://doi.org/10.1029/2024JH000173)).*

* **Physiographic Provinces (Figure S13c)**: Proposed ML model $R^2$ consistently ranges from $0.60 - 0.85$ across provinces, outperforming Blackburn-Lynch et al. (2017) in nearly every province.
* **Physiographic Divisions (Figure S13d)**: Proposed ML model $R^2$ remains consistently high ($R^2 \approx 0.65 - 0.85$) across all 8 major divisions, significantly outperforming Bieger et al. (2015) regional estimates.

---

## Section Navigation

- [Depth v1.0.0 Overview](index.md): Problem statement, FHG continuity formulation, and summary.
- [Model Architecture](models.md): Candidate ML models, feature engineering, and the 3-tier Stacking Meta-Learner.
- [Explainable AI (XAI)](xai.md): Global SHAP feature importances, 2D interaction dependencies, and physical interpretations.
