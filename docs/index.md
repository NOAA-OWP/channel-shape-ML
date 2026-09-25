---
template: main.html
title: Home
hide:
  - toc
---

<div class="pipeline-cards" markdown>
<div class="pipeline-card" markdown>
## TopWidth
**Bankfull Channel Width (\( W \propto Q^b \))**
[Explore →](tw/index.md)
</div>
<div class="pipeline-card" markdown>
## Depth
**Bankfull Depth (\( d \propto Q^f \))**
[Explore →](y/index.md)
</div>
<div class="pipeline-card" markdown>
## Shape
**Dingman Shape Exponent (\( r \))**
[Explore →](r/index.md)
</div>
<div class="pipeline-card" markdown>
## Roughness
**Manning's \( n \)**
[Explore →](n/index.md)
</div>
</div>

<!-- <div class="stage-flow" markdown>
```mermaid
graph LR
    S1[TopWidth] --> S2[Depth]
    S2 --> S3[Shape]
    S3 --> S4[Roughness]
```
</div> -->

<div class="home-capabilities-container" markdown>
<div class="home-capabilities-left" markdown>

## Key Capabilities

<div class="capabilities-grid" markdown>

- **Physics-Informed Features**: Feature engineering grounded in fluvial geomorphology.
- **Hybrid Ensemble**: Attention-augmented deep residual tabular network (PhysicalResTabNet) combined with gradient-boosted decision forest ensemble.
- **Continental Coverage**: Inference across 2.8M+ reaches across US territories (SUPERCONUS, AK, HI, PRVI).
- **Confidence Quantification**: Ensemble epistemic uncertainty with ML Confidence Score.
- **Explainable AI**: Model interpretability via SHAP, PDP/ICE, and Causal DML.
- **Topological Regularization**: Graph based deep learning regularization for network continuity.

</div>

</div>
<div class="home-capabilities-right" markdown>

![ML Pipeline Architecture](assets/images/workflow.jpg){ loading=lazy }

</div>
</div>

This ML pipeline was developed to provide robust, scalable, and physically-consistent estimates of channel geometry and roughness for the National Water Model.
