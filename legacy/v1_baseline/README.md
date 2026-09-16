# Legacy v1.0.0 Model Training & Preprocessing Baseline

This directory preserves the operational v1.0.0 training and preprocessing pipelines:
- `channel-WD/`: Bankfull width and depth model training scripts
- `channel-shape/`: Dingman shape parameter (r) model training scripts
- `preprocess/`: HYDRoSWOT, SCAT, and NWM feature extraction scripts

## Production Release Notes (v1.0.0)

<details>
<summary><b>Production-Ready Release Notes (v1.0.0) (Click to expand)</b></summary>

### Motivation and Purpose
The v1.0.0 release establishes an automated machine learning framework to estimate bankfull channel width, depth, and cross-sectional shape across CONUS, addressing the missing bathymetry data gap in DEMs for NextGen and FIM.

### Ingestion and Preprocessing
* Ingestion of acoustic Doppler current profiler (ADCP) soundings from USGS HYDRoSWOT, cleaned with AHGestimation.
* Feature extraction across more than 400 attributes.
* Feature space reduction via RFE and autoencoders to 60 predictors.

### ML Architectures and Ensembles
* Screening of 50 candidate algorithms with out-of-bag validation.
* Hyperparameter optimization using grid and randomized search.
* 3 deployment variants: Best Tuned Single Model, Voting Ensemble, and Stacking Meta-Learner.
* Skew handling via Quantile Transformation and StandardScaler with invertible mappings.
* Feature attribution diagnostics using XGBoost SHAP values.

### PR Changelog
* Model fitting and quantile transformations (#8, #9)
* HydroSWOT validation and geographic mapping (#11, #13)
* Bankfull width and depth ML models (#17)
* Feature importance diagnostics and output persistence (#24, #25)
* Automated HydroSWOT and SCAT data extraction pipelines (#26, #27)
* Conda environment setup for WD modeling (#28)
* PCA groupings, attribute definitions, and hyperparameter search spaces (#29, #30, #31)
* Automated deployment bash runners and execution modules (#35, #37, #38, #39, #41)
* High-throughput batch processing and transformation inversion (#40, #49)
* Data imputation routines and NWIS/NWM training target integration (#43, #46)
* Deployment boundary verification and feature scaling (#50, #51)
* Repository restructuring, license finalization, and cleanup (#54, #55, #56)

</details>

These scripts remain fully functional and serve as the baseline while the modernized v2 high-dimensional preprocessing and distributed training pipelines are completed.