---
title: Model Skill
---

# Model Skill

## Prediction Performance

![Observed vs. Predicted Manning's n-in](../../assets/images/n/v2.0.0/n-in/skill/pred_resnet_model_4.png){ loading=lazy }

The model's ability to predict isolated in-channel roughness is robust, as evidenced by the tight clustering along the 1:1 line in the test set. Compared to n-single, n-in values generally occupy a lower numerical range, accurately reflecting the absence of dense floodplain vegetation resistance.

## Regional Diagnostics

![HLR Performance](../../assets/images/n/v2.0.0/n-in/skill/hlr_resnet_model_4.png){ loading=lazy }

Performance stratified by Hydrologic Landscape Regions (HLR) shows consistent predictive skill. Simialr to n-single, the areas with higher score are:</br>
- HLR 1: Subhumid plains with permeable soils and bedrock </br>
- HLR 14: Arid playas with permeable soils and bedrock</br>
- HLR 20: Humid mountains with permeable soils and impermeable bedrock</br>
All three regions have permeable soils, which promote high infiltration and groundwater pathways over surface runoff. This produces more stable, baseflow-dominated hydrographs where stage-discharge and hydraulic geometry relationships follow predictable power law scaling.

The areas with lower score are:</br>
- HLR 4: Humid plains with permeable soils and bedrock</br>
- HLR 7: Humid plains with permeable soils and impermeable bedrock</br>
- HLR 8: Semiarid plains with impermeable soils and bedrock</br>
All these regions are Plains characterized by extremely low topographic relief and gentle channel slopes</br>

![HLR Choropleth](../../assets/images/n/v2.0.0/n-in/skill/hlr_map_resnet_model_4.png){ loading=lazy }

The choropleth highlights geographic variations, confirming that the model successfully resolves regional geologic differences influencing bed material composition.

## Training Convergence

![Training Curves](../../assets/images/n/v2.0.0/n-in/training_bias/dl_resnet_model_1.png){ loading=lazy }

The neural network training curves indicate stable convergence, validating the hyperparameter selection and structural regularization for the isolated n-in target space.

## Bias Analysis

![Residual Bias by Stream Order](../../assets/images/n/v2.0.0/n-in/training_bias/residual_bias_by_streamorder.png){ loading=lazy }

Residual bias mapped against stream order reveals negligible systemic drift. This proves the model reliably predicts in-channel roughness for both 1st order headwaters and high order lowland rivers, with 10 order having systemtic udnerestiamtion simiallr to n-single due to lack of training data.
