---
title: Model Skill
---

# Model Skill

---

## Prediction Performance

![Observed vs. Predicted Manning's n-out — Train / Validation / Test](../../assets/images/n/v2.0.0/n-out/skill/pred_resnet_model_1.png){ loading=lazy }

This 3-panel evaluation plot demonstrates the model's agreement with observed overbank roughness across the Training, Validation, and Test partitions. The tight alignment along the 1:1 line confirms high generalization skill across diverse floodplain morphologies without severe overfitting.

---

## Regional Diagnostics

![HLR Performance](../../assets/images/n/v2.0.0/n-out/skill/hlr_resnet_model_1.png){ loading=lazy }

Performance stratified by **Hydrologic Landscape Regions (HLRs)** demonstrates how the overbank model adapts across physiographic and hydro-climatic settings. Performance stratified by Hydrologic Landscape Regions (HLR) shows consistent predictive skill. Simialr to n-single, the areas with higher score are:</br>
- HLR 1: Subhumid plains with permeable soils and bedrock </br>
- HLR 14: Arid playas with permeable soils and bedrock</br>
- HLR 20: Humid mountains with permeable soils and impermeable bedrock</br>
All three regions have permeable soils, which promote high infiltration and groundwater pathways over surface runoff. This produces more stable, baseflow-dominated hydrographs where stage-discharge and hydraulic geometry relationships follow predictable power law scaling.

The areas with lower score are:</br>
- HLR 4: Humid plains with permeable soils and bedrock</br>
- HLR 7: Humid plains with permeable soils and impermeable bedrock</br>
- HLR 8: Semiarid plains with impermeable soils and bedrock</br>
All these regions are Plains characterized by extremely low topographic relief and gentle channel slopes</br>

![HLR Choropleth](../../assets/images/n/v2.0.0/n-out/skill/hlr_map_resnet_model_1.png){ loading=lazy }

The continental choropleth map illustrates spatial skill patterns across all 20 HLRs, highlighting areas of high predictive confidence.

---

## Training Convergence

![Training Curves](../../assets/images/n/v2.0.0/n-out/training_bias/dl_resnet_model_1.png){ loading=lazy }

The training curves illustrate loss convergence and RMSE evolution across optimization epochs. The smooth decay without divergence between training and validation trajectories validates the hyperparameter schedule and structural regularization of the deep tabular residual network.

---

## Bias Analysis

![Residual Bias by Stream Order](../../assets/images/n/v2.0.0/n-out/training_bias/residual_bias_by_streamorder.png){ loading=lazy }

Evaluating prediction residuals mapped against Strahler stream orders (1 through 9) indicates that the overbank roughness model maintains an unbiased zero-centered distribution across network scales, from headwater tributary floodplains to expansive continental lowland valleys. Only in 10th order stream due to lack of training data, there is a slight underestimation of roughness.
