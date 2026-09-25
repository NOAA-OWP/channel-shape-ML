---
title: Shape (R) v1.0.0 Overview
description: Continental channel cross-sectional shape parameterization using Dingman continuous power-law geometry and Feature Hydraulic Geometry (FHG) in v1.0.
---

# Shape ($r$) Parameterization Overview

In 1D and 2D hydrodynamic routing models (such as **HEC-RAS** and the **NextGen National Water Model**), the cross-sectional shape of a river channel controls the relationship between water surface elevation, conveyance area, wetted perimeter, and mean velocity. In the **v1.0.0 ML Parameterization Framework**, channel cross-sectional geometry is continuously parameterized using the power-law formulation introduced by **Dingman (2007)**, derived directly from machine-learned hydraulic geometry exponents.

---

## Derivation of Channel Shape ($r$) from Hydraulic Geometry Exponents

In 1D and 2D hydraulic modeling, cross-sectional conveyance volume is governed by the channel shape exponent $r$. In the **v1.0.0 ML Framework**, $r$ is derived directly from the relationship between at-a-station depth scaling exponent $f$ and width scaling exponent $b$ ($r = f/b$) fitted from **USGS HYDRoSWOT ADCP** field measurements without requiring LiDAR bathymetry:

```mermaid
flowchart LR
    subgraph ADCP ["USGS HYDRoSWOT ADCP Soundings"]
        DATA["<b>3,543 Field Stations</b><br/>Observed Stage-Discharge, Width, Depth & Velocity Profiling"]
    end

    subgraph EXP ["AHG Power-Law Exponents"]
        AHG["<b>Continuity Exponents:</b><br/>• Width scaling: b (W &prop; Q<sup>b</sup>)<br/>• Depth scaling: f (Y &prop; Q<sup>f</sup>)<br/><i>Continuity: b + f + m = 1.0</i>"]
    end

    subgraph META ["Meta-Learner & ML Framework"]
        ML_DERIVE["<b>Shape Exponent Derivation:</b><br/>r = f / b<br/>Generalization across 2.8M+ CONUS COMIDs"]
    end

    subgraph BED_SHAPE ["Dingman Power-Law Bathymetry"]
        BED["<b>Continuous Cross-Sectional Geometry:</b><br/>z(x) = Y<sub>m</sub>* &middot; (2x / W*)<sup>r</sup><br/>Parameterizes full in-channel shape across flow regimes"]
    end

    ADCP ==> EXP ==> META ==> BED_SHAPE

    class DATA highlight-blue;
    class AHG highlight-blue;
    class ML_DERIVE highlight-orange;
    class BED highlight-teal;
```

By linking the rate of channel widening ($b$) and deepening ($f$) with flow to the cross-sectional geometry, the ML pipeline provides physically continuous channel profiles across all CONUS reaches.

---

## Dingman Continuous Power-Law Formulation

Dingman (2007) parameterized symmetrical river cross-sections using a generalized continuous power-law function. The vertical height of the channel bed $z$ above the channel thalweg (assumed to occur at the channel centerline $x = 0$) is defined across the bankfull width $W^*$ as:

$$
z(x) = Y_m^* \cdot \left(\frac{2}{W^*}\right)^r \cdot x^r, \quad 0 \le x \le \frac{W^*}{2}
$$

where:

* $x$ is the lateral horizontal distance from the channel centerline ($0 \le x \le W^*/2$).
* $z(x)$ is the vertical bed elevation above the lowest point (thalweg) of the cross-section.
* $W^*$ is the bankfull top width.
* $Y_m^*$ is the maximum bankfull depth at the thalweg.
* $r$ is the dimensionless cross-sectional shape exponent ($0 < r < \infty$).

Alternatively, expressing local water depth $y(x)$ below the bankfull stage as a function of lateral coordinate $x \in [-W^*/2, W^*/2]$:

$$
y(x) = Y_m^* \left[1 - \left|\frac{2x}{W^*}\right|^r\right]
$$

### Morphological Spectrum of Exponent $r$

The dimensionless parameter $r$ defines the curvature of the channel bed and lateral bank steepness:

* **$r < 1.0$ (Convex)**: Channel bed slopes gently in the center and steepens outward, characteristic of shallow, unconfined multi-thread channels or compound bedrock sills.
* **$r = 1.0$ (Triangular / V-shaped)**: Steep V-shaped profile with zero flat bed width, typical of incised headwaters, gullies, and bedrock-confined canyons.
* **$r \approx 1.75$ (Lane Type B Stable Channel)**: The empirical threshold cross-section identified by Lane (1955) for stable alluvial channels in non-cohesive sediment under critical tractive force equilibrium.
* **$r = 2.0$ (Parabolic)**: Classical parabolic cross-section typical of equilibrium alluvial rivers in mobile gravel and sand-bed environments.
* **$r > 3.0$ (Flat-Bottomed / Trapezoidal)**: Steep vertical banks with a broad, flat bed, common in cohesive silt-clay lowland channels and urban floodways.
* **$r \to \infty$ (Rectangular / Box Channel)**: Vertical sidewalls and flat horizontal bed, characteristic of cohesive slot canyons, wooden flumes, and concrete canal conduits.

![CONUS Model Predicted Channel Shape Exponent Distribution and Morphological Cross-Section Geometries](../../assets/images/v1.0.0/r/R_best_model.png){ loading=lazy }

---

## Hydraulic Geometry Linkage: Connecting AHG to Exponent $r$

In the classical **At-a-station Hydraulic Geometry (AHG)** framework established by Leopold & Maddock (1953), channel width ($W$), mean depth ($Y$), and mean velocity ($V$) scale with discharge ($Q$) as power-law functions:

$$
W = a \cdot Q^b, \quad Y = c \cdot Q^f, \quad V = k \cdot Q^m
$$

subject to the physical continuity constraints:

$$
b + f + m = 1, \quad a \cdot c \cdot k = 1
$$

### Mathematical Derivation of $r = f / b$

For a power-law channel cross-section obeying $y(x) = Y_m^* [1 - |2x/W^*|^r]$, the top width $W(y)$ at any stage height $y$ above the bed expands according to:

$$
W(y) = W^* \cdot \left(\frac{y}{Y_m^*}\right)^{1/r}
$$

Differentiating the stage-discharge power laws with respect to discharge $Q$:

1. Lateral expansion rate: $W \propto Q^b \implies Q \propto W^{1/b}$
2. Vertical stage growth rate: $Y \propto Q^f \implies Q \propto Y^{1/f}$

Equating discharge relations reveals the stage-to-width power law:

$$
W \propto Y^{b/f} \iff Y \propto W^{f/b}
$$

Comparing exponents between Dingman's geometric power law ($W \propto y^{1/r}$) and the hydraulic geometry stage relation ($W \propto Y^{b/f}$) yields the exact fundamental physical relationship:

$$
r = \frac{f}{b}
$$

---

## Morphological Classification Summary

| Exponent Regime | Dominant Profile | Confinement & Bed Material | Hydrologic / Flow Regime | Geographic Settings |
| :--- | :--- | :--- | :--- | :--- |
| **$r < 1.0$** | **Convex / Composite** | Unconfined, non-cohesive gravels and coarse sands; low bank cohesion | High flashiness, ephemeral floods, wide lateral braiding | Arid Southwest washes, braided glacial outwash plains |
| **$1.0 \le r < 1.5$** | **Triangular / V-Shaped** | Bedrock-confined or colluvial boundaries; steep valley walls | Mountainous, high unit stream power, rapid runoff response | Cascade Range, Rocky Mountains, Appalachian headwaters |
| **$1.5 \le r \le 2.2$** | **Parabolic (Lane Type B)** | Cohesive-to-alluvial banks; mobile silt-sand bed equilibrium | Perennial humid-to-subhumid baseflow; balanced sediment load | Midwestern interior basins, Ohio River Valley, Southeast Piedmont |
| **$r > 2.2$** | **Trapezoidal / Rectangular** | High clay-silt bank cohesion; fine cohesive muds or engineered canals | Regulated low-gradient baseflow; low lateral erosion capacity | Mississippi Delta lowlands, Gulf Coastal Plain, urban corridors |

---

## Model Pipeline Navigation

* **[Model Architecture & Meta-Learners](models.md)**: Multi-tier machine learning framework, elbow method & PCA feature reduction, and meta-learner stacking.
* **[Model Skill & Validation](skill.md)**: Continental validation against HYDRoSWOT ADCP surveys, Kling-Gupta Efficiency CDFs, and stream-order bias analysis.
* **[Explainability & Physical Controls (XAI)](xai.md)**: SHAP global importance ranking, hydroclimatic interaction plots, and geomorphic sensitivity analysis.
