---
title: Literature
---

# Single Representative Roughness (n-single)

Manning's equation has been a cornerstone of open-channel hydraulics since its introduction by Robert Manning in 1891 [1], with subsequent theoretical developments linking it to bed material properties by Strickler [2]. For large-scale 1D modeling, it is common to use a single lumped effective value, **n-single**, to represent the bulk flow resistance. Morever a study by Johnson et al (2024) [3] has shown that the ML hydrographic network drived roughness outperforms land cover, stream order based approuches.

## Classical Approaches
Classical estimation of Manning's $n$ relies on empirical tables, visual guides, and additive methods. Cowan (1956) [4] introduced a compositional approach that adds baseline grain roughness to modifiers for channel irregularity, variation, obstructions, vegetation, and meandering. Chow (1959) [5] and Arcement & Schneider (1989) [6] expanded on these guidelines with extensive field data and visual references.

Theoretical foundations, such as the Keulegan equation, demonstrate that flow resistance is not strictly constant but depends dynamically on the hydraulic radius and relative roughness. As flow depth increases, the relative effect of boundary friction decreases.

## Machine Learning in Roughness Estimation
Recent advances have sought to replace subjective lookup tables with data-driven models. Machine learning (ML) approaches offer the ability to leverage massive geospatial datasets to estimate $n$ [6]. However, purely data-driven models can struggle with physical consistency.

### Our Approach: Hybrid Ensemble with Physics-Informed Features
We justify a hybrid ensemble approach that combines data-driven ML with physics-informed staged features. By integrating topological constraints, topological smoothing, and physical scaling laws directly into the feature space (e.g., Fluvial Concavity Deviation, Width-to-Drainage Ratio), our model respects the underlying geomorphic drivers of flow resistance while capturing complex, non-linear geographic variations. 

The choice of a single lumped value (n-single) remains highly relevant due to its simplicity, computational efficiency, and robust historical precedent in 1D hydrodynamic routing.

## References
[1] Manning, R. (1891). *On the flow of water in open channels and pipes*. </br>
[2] Strickler, A. (1923). *Contributions to the question of a velocity formula and roughness data for streams, channels and closed pipelines*.</br>
[3] Johnson JM, Eyelade D, Singh-Mohudpur J, Rad AM, Coll J, Spies R, Yeghiazarian L (2024). *Enhancing Synthetic Rating Curve Development Through Empirical Roughness Built for Hydrofabric Datasets*.</br>
[4] Cowan, W. L. (1956). *Estimating hydraulic roughness coefficients*.</br>
[5] Chow, V. T. (1959). *Open-channel hydraulics*.</br>
[6] Arcement, G. J., & Schneider, V. R. (1989). *Guide for selecting Manning's roughness coefficients for natural channels and flood plains*.</br>
