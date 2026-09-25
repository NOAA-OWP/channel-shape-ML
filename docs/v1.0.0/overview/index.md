---
title: CONUS-FHG v1.0.0 Pipeline Overview
description: Detailed architectural overview of the continental Feature Hydraulic Geometry (CONUS-FHG v1.0.0) machine learning framework.
---

# CONUS-FHG v1.0.0 Pipeline Overview

The **CONUS-FHG v1.0.0** framework provides a robust, physics-grounded machine learning methodology for estimating continental-scale river channel geometry and submerged cross-sectional shape across the Contiguous United States (CONUS). Developed by Modaresi Rad et al. (2024), this framework bridges the gap in derivaiton of river channel geometry as standard DEMs cannot penetrate water.

---

## Deriving Continental Channel Geometry from ADCP Observations

Modern hydrologic and hydrodynamic modeling frameworks such as the NOAA **Next Generation Water Resources Modeling Framework (NextGen)**, **FEMA Flood Insurance Studies (FIS)**, and continental **Flood Inundation Mapping (FIM)** require accurate submerged channel geometry to simulate open-channel conveyance and flood wave propagation:

```mermaid
flowchart TD
    subgraph FIELD ["1. USGS HYDRoSWOT ADCP Field Observation Base"]
        ADCP["<b>3,543 USGS ADCP Gauge Stations (1,432 Distinct River Systems)</b><br/>In-situ cross-sectional depth and acoustic velocity soundings evaluated across flow regimes"]
    end

    subgraph LABELS ["2. Hydrologic Target Derivation & AHG Continuity Fitting"]
        direction LR
        IN_FLOW["<b>In-Channel Targets (100% AEP):</b><br/>• In-Channel Width: <code>TW<sub>in</sub></code><br/>• In-Channel Depth: <code>Y<sub>in</sub></code>"]
        BF_FLOW["<b>Bankfull Targets (50% AEP):</b><br/>• Bankfull Width: <code>TW<sub>bf</sub></code><br/>• Bankfull Depth: <code>Y<sub>bf</sub></code>"]
        AHG_FIT["<b>AHG Continuity Exponents:</b><br/>• Width Exponent: <code>b</code><br/>• Depth Exponent: <code>f</code><br/><i>(b + f + m = 1.0)</i>"]
    end

    subgraph ML ["3. Multi-Tier Machine Learning Generalization"]
        PRED["<b>Continental ML Training & Stacking Meta-Learner:</b><br/>Trained on 30 curated predictors (NWM 2.1, POLARIS Soils, 3DEP DEM, PRISM Climate, StreamCat)<br/>Predicts 100% AEP &amp; 50% AEP dimensions and AHG exponents across 2.8M+ Reference Fabric reaches"]
    end

    subgraph BATH ["4. Synthetic Channel Cross-Sectional Bathymetry"]
        BED["<b>Dingman Power-Law Bed Elevation:</b><br/><code>z(x) = Y<sub>m</sub>* &middot; (2x / W*)<sup>r</sup></code> &nbsp; where &nbsp; <code>r = f / b</code><br/>Reconstructs complete submerged 3D channel geometry for FEMA FIS &amp; NextGen FIM"]
    end

    FIELD ==> LABELS ==> ML ==> BATH

    class FIELD highlight-blue;
    class IN_FLOW highlight-teal;
    class BF_FLOW highlight-orange;
    class AHG_FIT highlight-blue;
    class ML highlight-blue;
    class BATH highlight-teal;
```

The framework derives channel top width, depth, and cross-sectional shape exclusively from **USGS HYDRoSWOT Acoustic Doppler Current Profiler (ADCP)** field surveys and hydro-environmental features, without relying on airborne LiDAR or elevation scans:

1. **In-Situ ADCP Acoustic Soundings**: Uses field-measured cross-sectional velocity and depth profiles from 3,543 USGS gauging stations across 1,432 distinct river systems.
2. **Hydrologic Flow Target Extraction**: Extracts dimensional width and depth at the **100% AEP** (in-channel mean annual flow) and **50% AEP** (bankfull 2-year flood) flow thresholds.
3. **Continuity-Constrained AHG Fitting**: Fits At-a-station Hydraulic Geometry (AHG) scaling exponents ($b, f, m$) under mass conservation continuity constraints ($b + f + m = 1.0, a \cdot c \cdot k = 1.0$).
4. **Machine Learning Generalization**: Trains multi-tier ML models (XGBoost, ExtraTrees, LightGBM, CatBoost, and Stacking Meta-Learners) to predict in-channel dimensions, bankfull dimensions, and AHG exponents across all un-gauged Reference Fabric reaches.
5. **Dingman Power-Law Bathymetry Synthesis**: Couples predicted bankfull dimensions ($TW_{bf}, Y_{bf}$) with the continuous Dingman shape parameter ($r = f/b$) to synthesize complete submerged cross-sections for flood inundation mapping.

---

## From At-a-Station to Feature Hydraulic Geometry (FHG)

### At-a-Station Hydraulic Geometry (AHG)

Classical hydraulic geometry, pioneered by Leopold and Maddock (1953), demonstrates that cross-sectional top width ($W$), mean depth ($Y$), and mean flow velocity ($V$) scale with discharge ($Q$) as power-law functions at a fixed river cross-section:

$$
W = a \cdot Q^b
$$

$$
Y = c \cdot Q^f
$$

$$
V = k \cdot Q^m
$$

where $a, c, k$ are empirical coefficients and $b, f, m$ are dimensionless scaling exponents. 

From the physical principle of mass conservation (continuity equation $Q = W \cdot Y \cdot V$ for incompressible fluid flow in steady state), the power laws must satisfy exact mathematical constraints:

$$
(a \cdot Q^b) \cdot (c \cdot Q^f) \cdot (k \cdot Q^m) = (a \cdot c \cdot k) \cdot Q^{b + f + m} = Q^1
$$

This yields the fundamental AHG continuity criteria:

$$
a \cdot c \cdot k = 1.0 \quad \text{and} \quad b + f + m = 1.0
$$

### Feature Hydraulic Geometry (FHG)

While AHG describes empirical relationships at individual gauge sites, **Feature Hydraulic Geometry (FHG)** expands these relationships to regional and continental scales. FHG treats the power-law parameters ($a, b, c, f, k, m$) as continuous functional responses to holistic catchment and hydrographic features:

$$
\{a, b, c, f, k, m\} = \mathcal{F}\Big(\mathbf{X}_{\text{NWM}}, \mathbf{X}_{\text{StreamCat}}, \mathbf{X}_{\text{DEM}}, \mathbf{X}_{\text{Soil}}, \mathbf{X}_{\text{Climate}}\Big)
$$

where $\mathcal{F}$ represents the machine learning mapping function parameterized across CONUS.

---

## End-to-End Workflow Architecture

The pipeline consists of four tightly integrated phases spanning data acquisition, feature space reduction, multi-tier machine learning, and continental inference:

![CONUS-FHG v1.0.0 End-to-End Machine Learning Workflow](../../assets/images/v1.0.0/overview/Fig2.png){ loading=lazy }

### Workflow Stages:

1. **Hydrographic Data Curation & Preprocessing**:
   > Extraction of raw Acoustic Doppler Current Profiler (ADCP) cross-sectional surveys from the USGS HYDRoSWOT database.
   > </br> Quality control: filtering out tidal/backwater stations, unstable cross-sections, and records with fewer than 5 surveys.
   > </br> Robust optimization fitting of AHG power laws ensuring $R^2 \ge 0.8$ while strictly enforcing continuity constraints ($a \cdot c \cdot k = 1.0$, $b + f + m = 1.0$).
2. **Environmental Feature Extraction**:
   > </br> Extraction of 116 candidate environmental predictors across all gauge locations.
   > </br> Predictor families include National Water Model (NWM v2.1) flood frequencies, StreamCat landscape attributes, POLARIS high-resolution soil physical parameters, 3DEP DEM topographic indices, and PRISM climate statistics.
3. **Dimensionality Reduction & Latent Encoding**:
   > Game-theoretic SHAP (Shapley Additive Explanations) importance screening to prune noisy, uninformative variables. </br>
   > Unsupervised optimal feature detecotr during validation.
4. **Multi-Tier Ensemble Modeling & Continental Generalization**:
   > Systematic benchmarking of 50 candidate regression algorithms.
   > </br> Deployment of a 3-tier predictive cascade: Best Tuned Regressors (Tier 1), Voting Ensemble (Tier 2), and Stacking Meta-Learner (Tier 3).
   > </br> Validation via 10-fold spatial cross-validation and out-of-bag testing.
   > </br> Synthesis of continuous Dingman channel shape exponents ($r = f/b$) for all reaches.

---

## Study Area & Training Domain

The training dataset encompasses **3,543 USGS stream gauging stations** distributed across **1,432 distinct river systems** throughout the Contiguous United States, The most important varaible was identified as discharge from NWM and bellow is comaprison of NWM 2.1 to NWIS. 

![Spatial Distribution of USGS HYDRoSWOT ADCP Training Stations across CONUS River Networks](../../assets/images/v1.0.0/overview/Fig1.png){ loading=lazy }

### Spatial & Hydrographic Diversity:

* **Geographic Breadth**: Spans all major physiographic provinces and Hydrologic Landscape Regions (HLRs) of CONUS, from the humid Appalachian basins to the arid Southwest and Pacific Northwest maritime environments.
* **Stream Order Representation**: Includes 1st through 8th Strahler stream orders, capturing the geomorphic continuum from steep, confined headwaters to large lowland alluvial mainstems (e.g. Mississippi, Ohio, Missouri, Columbia).
* **Station Density & Network Clustering**: Clustered station groups enable robust evaluation of spatial autocorrelation and cross-reach continuity along continuous river networks.

---

## Modeling Strategy: Three-Tier Ensemble

To evaluate whether algorithmic complexity improves hydraulic geometry estimation, v1.0.0 implements a systematic three-tier modeling hierarchy:

```mermaid
flowchart TD
    subgraph BENCH ["Algorithm Benchmarking (50 Candidates)"]
        ALL["Linear &bull; Ridge &bull; Lasso &bull; ElasticNet &bull; SVR &bull; KNN &bull;<br/>Random Forest &bull; Extra Trees &bull; AdaBoost &bull; LightGBM &bull; XGBoost &bull; CatBoost &bull; MLP"]
    end

    subgraph TOP ["Top Tuned Regressors"]
        T_XGB["Tuned XGBoost"]
        T_CAT["Tuned CatBoost"]
        T_RF["Tuned Random Forest"]
        T_ET["Tuned Extra Trees"]
        T_GB["Tuned Gradient Boosting"]
        T_MLP["Tuned Neural Net (MLP)"]
    end

    subgraph T1 ["Tier 1: Best Model"]
        BEST["Single Top-Performing Tuned Algorithm<br/>(e.g., CatBoost / XGBoost)"]
    end

    subgraph T2 ["Tier 2: Voting Ensemble"]
        VOTE["Ensemble Mean Prediction:<br/>&ycirc;<sub>vote</sub> = (1/M) &sum; &ycirc;<sub>m</sub>"]
    end

    subgraph T3 ["Tier 3: Stacking Meta-Learner"]
        META["Level-1 Meta-Regressor<br/>Learns optimal non-linear blending of base model out-of-fold outputs"]
    end

    ALL ==> TOP
    TOP --> T1
    TOP --> T2
    TOP --> T3

    class BEST highlight-blue;
    class VOTE highlight-teal;
    class META highlight-orange;
```

1. **Tier 1 (Best Tuned Model)**: Identifies the single best-performing machine learning architecture for each target parameter following extensive Bayesian hyperparameter optimization.
2. **Tier 2 (Voting Ensemble)**: Computes the unweighted ensemble average of the top hyperparameter-tuned base models, dampening idiosyncratic model variance.
3. **Tier 3 (Stacking Meta-Learner)**: Trains a higher-level regression model on the out-of-fold cross-validation predictions of base models. The meta-learner learns the regional strengths and weaknesses of each base model, achieving superior predictive skill across the full spectrum of river scales.

---

## Section Navigation

For comprehensive technical documentation of each v1.0.0 subsystem, explore the following pages:

* **[Data Sources & Quality Control](data-sources.md)**: HYDRoSWOT ADCP data cleaning, continuity fitting, and environmental predictor PCA decompositions.
* **[Methods & Feature Engineering](methods.md)**: Expert knowledge screening, recursive feature elimination with elbow method skill evaluation, PCA dimensionality reduction, and multi-tier ensembling.
* **[Channel Shape Parameterization ($r$)](../r/index.md)**: Mathematical derivation of Dingman exponent $r$, morphological classifications, and CONUS-scale spatial trends.
