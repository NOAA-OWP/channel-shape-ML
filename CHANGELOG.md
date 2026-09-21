# Changelog

All notable changes to this project will be documented in this file. We follow the [Semantic Versioning 2.0.0](http://semver.org/) format.

---

## [v2.0.0-alpha.1] - 2026-09-04 - [PR#59](https://github.com/NOAA-OWP/channel-shape-ML/pull/59)

This PR implements the first milestone of the v2.0.0 architecture (Epic #57, closes #58) by introducing a 6-stage sequential DAG inference pipeline, Gaussian Markov Random Field (GMRF) topological regularization along stream networks, decoupled model loading with local and Amazon S3 streaming support, and containerized Docker and Conda deployment workflows. The v2.0.0 architecture was developed to use USGS gage observations, HydroSWOT ADCP measurements, and MIP for training and uses hydrofabric network attributes only as predictors. Legacy v1.0.0 training and preprocessing pipelines are reorganized into a transitional legacy directory to keep the root directory clean while maintaining full baseline reproducibility.

### Additions

- `deployment/inference_engine.py`: Batch inference execution engine with boundary checks and distribution scaling inversion.
- `deployment/sequential_orchestrator.py`: Orchestration module linking DAG stages, model loading, and parquet output compilation.
- `entrypoint.sh`: Container entrypoint script managing environment initialization and CLI execution.
- `environment.yml`: Standard Conda environment specification for CPU inference.
- `environment-cuda.yml`: GPU-accelerated Conda environment specification with CUDA support.
- `Dockerfile`: Multi-platform container build definition for production deployment.
- `docker-compose.yml`: Container orchestration service specification for local and server runs.
- `legacy/v1_baseline/README.md`: Documentation explaining legacy v1 baseline directory organization and usage.
- `Makefile`: Build, lint, and execution automation targets.
- `run_inference.py`: Production CLI entrypoint supporting GeoPackage and Parquet inputs for end-to-end channel geometry and roughness prediction.
- `run_pipeline_example.py`: Standalone script demonstrating programmatic Python API execution of the inference pipeline.
- `src/core/io.py`: Data ingestion, column alias resolution, and parquet read and write utilities.
- `src/core/logging.py`: Standardized logging configuration across pipeline modules.
- `src/features/pipeline.py`: Dynamic feature engineering calculating hydraulic and geomorphic covariates at each DAG stage.
- `src/models/architectures.py`: Model architecture definitions for deep learning and ensemble regressors.
- `src/models/wrappers.py`: Decoupled model loaders supporting local filesystem and Amazon S3 streaming URIs with XGBoost and ResNet ensemble support.
- `src/postprocessing/regularization.py`: Gaussian Markov Random Field (GMRF) topological regularization enforcing downstream reach monotonicity along mainstems.

### Changes

- `README.md`: Comprehensive update integrating official v1.0.0 baseline release notes, Leopold and Maddock hydraulic geometry, Dingman power-law cross-sections, 6-stage DAG workflow, input/output schema specifications, and CLI/Docker execution commands.
- `channel-WD/`: Relocated into `legacy/v1_baseline/channel-WD/` to isolate baseline bankfull width and depth training scripts.
- `channel-shape/`: Relocated into `legacy/v1_baseline/channel-shape/` to isolate baseline Dingman shape parameter training scripts.
- `preprocess/`: Relocated into `legacy/v1_baseline/preprocess/` to isolate baseline feature extraction pipelines.

---

## [v1.0.0] - 2026-09-04 - [Release v1.0.0](https://github.com/NOAA-OWP/channel-shape-ML/releases/tag/v1.0.0)

This release establishes the baseline automated machine learning framework to estimate bankfull channel width, depth, and cross-sectional shape across CONUS, addressing the missing bathymetry data gap in DEMs in support of NextGen and FIM.

### Additions

- `channel-WD/deployment/conda_setup.bash`: Environment creation and dependency installation script for bankfull modeling.
- `channel-WD/run_ml.bash`: Operational runner for bankfull width and depth candidate screening, hyperparameter optimization, and stacking meta-learners.
- `channel-shape/run_ml.bash`: Operational runner for Dingman channel shape parameter r modeling using Leopold and Maddock power-law hydraulic geometry relations.
- `preprocess/`: Automated feature extraction and data preparation scripts processing USGS HYDRoSWOT ADCP soundings and catchment attributes across 7 categories.
