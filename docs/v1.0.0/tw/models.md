---
title: Model Architectures & Optimization (v1.0.0)
description: Machine learning algorithms, hyperparameter tuning, feature selection, and stacked meta-learner design for TopWidth v1.0.
---

# TopWidth: Model Architectures & Optimization

The v1.0.0 TopWidth modeling framework uses a robust multi-model machine learning architecture to predict At Station Hydraulic Geometry (AHG) parameters across CONUS river networks ([Modaresi Rad et al., 2024](https://doi.org/10.1029/2024JH000173)). To capture non-linear interactions across diverse physiographic settings while preventing overfitting, the pipeline integrates feature selection, dimensionality reduction, hyperparameter optimization, and a two-tier **stacked meta-learner**.

---

## Target Formulation & Physical Constraining

The v1.0.0 TopWidth modeling framework predicts both dimensional channel widths and dimensionless hydraulic geometry scaling parameters:

1. **In-Channel & Bankfull Width Targets**:
   > </br> **In-Channel Width ($TW_{\text{in}}$)**: Evaluated at the **100% AEP discharge** ($Q_{\text{100\% AEP}}$, 1-year mean annual flow).
   > </br> **Bankfull Width ($TW_{\text{bf}}$)**: Evaluated at the **50% AEP discharge** ($Q_{\text{50\% AEP}}$, 2-year channel-forming flood).
   > </br> Target transformation: Log-transformed ($\ln(TW)$) to normalize right-skewed width distributions across stream orders:
     $y_{TW} = \ln(TW) \implies \widehat{TW} = \exp(\hat{y}_{TW})$

2. **AHG Power-Law Scaling Exponent ($b$)**:
   > </br> Fitted across multi-flow ADCP surveys under mass continuity ($b + f + m = 1.0$).
   > </br> Bounded in $(0, 1)$ via logit transformation and used specifically with depth exponent $f$ to derive the continuous Dingman shape parameter ($r = f/b$):
     $y_b = \ln\left(\frac{b}{1 - b}\right) \iff b = \frac{1}{1 + \exp(-y_b)}$

```mermaid
flowchart TD
    subgraph OBS ["1. USGS HYDRoSWOT ADCP Field Measurements"]
        ADCP["<b>3,543 Field Stations</b><br/>Observed multi-flow cross-sectional surveys & stage-discharge rating curves"]
    end

    subgraph FLOW ["2. NWIS Flow Threshold Extraction"]
        direction LR
        Q100["<b>100% AEP Flow (Q<sub>100%</sub>)</b><br/>In-Channel Mean Annual Flow<br/><i>Derives TW<sub>in</sub> Label</i>"]
        Q50["<b>50% AEP Flow (Q<sub>50%</sub>)</b><br/>Bankfull 2-Yr Flow<br/><i>Derives TW<sub>bf</sub> Label</i>"]
        AHG_FIT["<b>AHG Continuity Fitting</b><br/>W = a &middot; Q<sup>b</sup><br/><i>Derives Exponent b Label</i>"]
    end

    subgraph ML_TARGETS ["3. Multi-Tier ML Engine"]
        direction LR
        ML_IN["<b>Predict In-Channel TW</b><br/><code>TW<sub>in</sub> = exp(ŷ<sub>in</sub>)</code>"]
        ML_BF["<b>Predict Bankfull TW</b><br/><code>TW<sub>bf</sub> = exp(ŷ<sub>bf</sub>)</code>"]
        ML_B["<b>Predict Exponent b</b><br/><code>b = logit<sup>-1</sup>(ŷ<sub>b</sub>)</code><br/><i>(For Dingman r = f / b)</i>"]
    end

    OBS ==> FLOW ==> ML_TARGETS

    class OBS highlight-blue;
    class FLOW highlight-blue;
    class ML_IN highlight-teal;
    class ML_BF highlight-orange;
    class ML_B highlight-blue;
```

---

## Machine Learning Algorithms

To explore structural model diversity, five distinct non-linear machine learning algorithms were trained and benchmarked alongside a stacked meta-learner:

```mermaid
flowchart TD
    subgraph TIER1 ["Tier 1: Diverse Base Learners"]
        direction TB
        XGB["<b>XGBoost</b><br/>Exact greedy tree boosting with L1/L2 regularization"]
        ET["<b>ExtraTrees</b><br/>Extremely randomized trees with aggressive variance reduction"]
        LGBM["<b>LightGBM</b><br/>Fast leaf-wise tree growth with histogram binning"]
        CAT["<b>CatBoost</b><br/>Oblivious decision trees with ordered boosting"]
        RF["<b>Random Forest</b><br/>Bagged decorrelated decision tree ensemble"]
    end

    subgraph CV ["Out-of-Fold Cross-Validation"]
        OOF["OOF Meta-Feature Generator<br/><i>Prevents Target Snooping & Data Leakage</i>"]
    end

    subgraph TIER2 ["Tier 2: Meta-Learning Layer"]
        META["<b>Stacked Meta-Learner</b><br/>Regularized Meta-Regressor (Ridge / ElasticNet)<br/><i>Dynamically weights models based on hydro-climate regimes</i>"]
    end

    subgraph FINAL ["Final Predictions"]
        PRED["<b>Optimized TopWidth Prediction</b><br/><code>W(Q) = &acirc; &middot; Q<sup>b&#770;</sup></code> &bull; <code>W<sub>bf</sub> = &acirc; &middot; Q<sub>bf</sub><sup>b&#770;</sup></code>"]
    end

    XGB --> OOF
    ET --> OOF
    LGBM --> OOF
    CAT --> OOF
    RF --> OOF

    OOF ==> META
    META ==> PRED

    classDef default fill:#1e293b,stroke:#475569,stroke-width:1.5px,color:#f8fafc;
    classDef highlight fill:#c2410c,stroke:#fb923c,stroke-width:2px,color:#ffffff;
    class META highlight;
```

---

## Two-Tier Stacked Meta-Learner Architecture

Rather than relying on any single algorithm, the v1.0.0 pipeline employs a **Stacked Generalization (Stacking)** framework:

$$\hat{y}_{\text{meta}} = g\left(\sum_{k=1}^K w_k \cdot \hat{y}_k + \mathbf{\theta}^T \mathbf{z}\right)$$

where:

* $\hat{y}_k$ is the out-of-fold prediction from the $k$-th base learner ($k \in \{\text{XGBoost}, \text{ExtraTrees}, \text{LightGBM}, \text{CatBoost}, \text{RF}\}$).
* $w_k$ is the optimal blending weight assigned by the meta-regressor.
* $\mathbf{z}$ represents high-level environmental conditioning covariates.
* $g(\cdot)$ is the meta-learner link function.

### Advantages of the Meta-Learner

1. **Complementary Strengths**: Tree boosting models (XGBoost/LightGBM) excel at sharp physical transitions (e.g., bedrock knickpoints, urbanized valleys), while randomized bagging models (ExtraTrees/RF) maintain smooth interpolations in data-sparse regions.
2. **Leakage Prevention**: Base learners generate meta-features strictly using $K$-fold out-of-fold (OOF) cross-validation, ensuring the meta-learner never observes base predictions generated on training folds.
3. **Variance Reduction**: Meta-learning consistently reduces prediction variance across all stream orders compared to individual standalone models.

---

## Feature Selection & Dimensionality Reduction

The raw environmental feature catalog contained 116 candidate variables spanning hydro-climatology (PRISM/Daymet), catchment topography (DEM), stream network topology (Reference Fabric), soil physics (POLARIS / SSURGO), and flow statistics (NWM 2.1).

To eliminate redundant collinearity and build a robust model, a four-stage feature pruning pipeline was applied:

```mermaid
flowchart TD
    RAW["<b>Raw Candidate Feature Catalog</b><br/>116 environmental & hydro-climatic variables"]
    STEP1["<b>Stage 1: Expert Knowledge Screening</b><br/>Pre-screen variables based on hydrological & geomorphic principles"]
    STEP2["<b>Stage 2: Recursive Feature Elimination & Elbow Method</b><br/>Iterative backward pruning by importance &bull; Identify optimal complexity via elbow curve"]
    STEP3["<b>Stage 3: Principal Component Analysis (PCA)</b><br/>Compress collinear thematic sub-domains (flood frequency, soil profiles) into orthogonal components"]
    FINAL["<b>Final Selected Feature Subset (30 Features)</b><br/>Optimal parsimonious input space for continental generalization"]

    RAW --> STEP1 --> STEP2 --> STEP3 --> FINAL

    class RAW highlight-blue;
    class STEP1 highlight-blue;
    class STEP2 highlight-orange;
    class STEP3 highlight-teal;
    class FINAL highlight-orange;
```

### Final 30-Feature Thematic Breakdown

The selected 30 predictors span 7 critical hydro-geomorphic domains:

1. **Hydro-Climatology**: Mean annual precipitation, potential evapotranspiration (PET), seasonality index, aridity index.
2. **Network Topology & Scale**: Upstream drainage area (`totdasqkm`), Strahler stream order, arbolate sum (`arb_sum`), downstream path length.
3. **Terrain & Geomorphometry**: Channel bed slope, elevation, Topographic Wetness Index (TWI), Fluvial Concavity Deviation (FCD), relief ratio.
4. **Soil Physical Properties**: Saturated hydraulic conductivity ($K_{sat}$), saturated soil water content ($\theta_s$), percent clay, silt, and sand fractions.
5. **Hydrologic Flow Regimes**: Bankfull discharge proxy ($Q_{bf}$), NWM 2.1 flood frequency principal components (PC0, PC1), Base Flow Index (BFI).
6. **Land Cover & Vegetation**: Leaf Area Index (LAI), Normalized Difference Vegetation Index (NDVI), riparian tree canopy fraction.
7. **Geological Context**: US NED Physiographic Diversity and regional lithologic resistance indices.

---

## Hyperparameter Optimization Strategy

Hyperparameters were systematically tuned using Bayesian Optimization across 10-fold cross-validation with out-of-fold (OOF) evaluation.

```mermaid
flowchart TD
    subgraph P1 ["1. 10-Fold Cross-Validation Partitioning"]
        direction LR
        S1["<b>3,543 Field Stations</b><br/>Partitioned across 10 cross-validation folds"]
        S2["<b>Out-of-Fold Evaluation</b><br/>Ensures all stations are evaluated out-of-bag"]
    end

    subgraph P2 ["2. Bayesian Hyperparameter Optimization"]
        direction LR
        B1["<b>TPE Bayesian Search</b><br/>100 iterations over learning rate, tree depth, and subsample ratios"]
        B2["<b>Early Stopping Regularization</b><br/>Halt after 50 rounds without validation skill gain"]
    end

    subgraph P3 ["3. Multi-Tier Meta-Learner Stacking"]
        direction LR
        M1["<b>Out-of-Fold (OOF) Stacking</b><br/>Generate uncorrupted Level-1 meta-feature predictions"]
        M2["<b>Final Meta-Learner Regularization</b><br/>Train Level-2 meta-regressor to optimize multi-model blending"]
    end

    P1 ==> P2 ==> P3

    class P1 highlight-blue;
    class P2 highlight-blue;
    class P3 highlight-teal;
    class S1 highlight-blue;
    class S2 highlight-blue;
    class B1 highlight-blue;
    class B2 highlight-blue;
    class M1 highlight-teal;
    class M2 highlight-orange;
```

!!! tip "Out-of-Fold Cross-Validation Safeguard"
    To ensure robust generalization across unseen reaches, 10-fold cross-validation is used across all candidate models. Predictions for meta-learning are strictly generated out-of-fold (OOF), preventing any information leakage between training and testing partitions.
