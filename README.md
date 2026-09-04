# Channel Bathymetry and Hydraulic Geometry ML Engine (channel-shape-ML)

[![Maintenance: Actively Developed](https://img.shields.io/badge/maintenance-actively--developed-brightgreen.svg)](https://github.com/NOAA-OWP/channel-shape-ML)
[![Python: 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Integration: NextGen and FIM](https://img.shields.io/badge/Integration-NextGen%20%7C%20FIM-teal.svg)](https://water.noaa.gov/)
[![Docker: Ready](https://img.shields.io/badge/Docker-Containerized-2496ED.svg?logo=docker&logoColor=white)](Dockerfile)
[![Keras](https://img.shields.io/badge/Keras-%23D00000.svg?logo=keras&logoColor=white)](https://keras.io/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-%23FF6F00.svg?logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-%2315803D.svg?logo=xgboost&logoColor=white)](https://xgboost.readthedocs.io/)

The NOAA Office of Water Prediction (OWP) automated machine learning framework for estimating bankfull channel width, depth, shape, and Manning's roughness across the Continental United States (CONUS). Developed in direct support of the Next Generation Water Modeling Framework (NextGen) and Flood Inundation Mapping (FIM).

---

## Table of Contents
1. [Motivation and Scientific Background](#1-motivation-and-scientific-background)
2. [Architectural Workflow: 6-Stage Sequential DAG](#2-architectural-workflow-6-stage-sequential-dag)
3. [Input Datasets and Schema Requirements](#3-input-datasets-and-schema-requirements)
4. [Master Consolidated Output Schema](#4-master-consolidated-output-schema)
5. [Inference Pipeline Execution](#5-inference-pipeline-execution)
   - [5.1 Environment Setup](#51-environment-setup)
   - [5.2 CLI Inference Execution](#52-cli-inference-execution)
   - [5.3 Docker Execution](#53-docker-execution)
6. [Production Release Notes (v1.0.0)](#6-production-release-notes-v100)
7. [Evolution and Development Roadmap](#7-evolution-and-development-roadmap)
8. [Contributing and Code Standards](#8-contributing-and-code-standards)
9. [Open Source Licensing and Disclaimer](#9-open-source-licensing-and-disclaimer)
10. [References and Citations](#10-references-and-citations)

---

## 1. Motivation and Scientific Background

Standard Digital Elevation Models (DEMs) cannot penetrate open water surfaces and map river channels as flat planes. This missing in-channel bathymetry causes large errors in river storage volume, channel conveyance capacity, flood wave routing velocity, and overbank inundation thresholds.

`channel-shape-ML` predicts the missing channel geometry and hydraulic roughness across diverse hydrologic regions:

1. **Bankfull and In-Channel Dimensions**: Predicts channel width ($W$) and depth ($D$) associated with the 2-year flood recurrence interval and baseflows.
2. **At-a-Station Functional Hydraulic Geometry (FHG)**: Estimates power-law parameters (Leopold and Maddock, 1953) constrained by physical continuity:
   $$\text{Top Width: } TW = a \cdot Q^b$$
   $$\text{Mean Depth: } Y = c \cdot Q^f$$
   $$\text{Flow Velocity: } V = k \cdot Q^m$$
   $$\text{Continuity Constraints: } a \cdot c \cdot k = 1.0, \quad b + f + m = 1.0$$
3. **Cross-Sectional Shape Curvature Parameter ($r$)**: Analytically derives channel bed curvature using the Dingman (2007) cross-sectional power-law geometry:
   $$Z(x) = Y_m^* \cdot \left(\frac{2}{W^*}\right)^r \cdot x^r, \quad 0 \le x \le \frac{W^*}{2}$$
   - $r = 1.0$: Triangular bed.
   - $r \approx 1.75$: Lane Type B stable regime channel.
   - $r = 2.0$: Parabolic cross-section.
   - $r > 2.5$: Rectangular bed with steep banks.
4. **Manning's Roughness ($n$)**: Estimates overall channel roughness, in-channel roughness ($n_{in}$), and overbank roughness ($n_{out}$) with uncertainty control limits.

These predictions improve upon regional empirical curves previously used in National Water Model (NWM) versions 2.0, 2.1, and 3.0.

---

## 2. Architectural Workflow: 6-Stage Sequential DAG

The inference engine runs predictions across river networks as a Directed Acyclic Graph (DAG). Upstream geometric predictions serve as engineered features for subsequent hydraulic stages (website refrence ....)

---

## 3. Input Datasets and Schema Requirements

The inference pipeline requires two input files: the Flowpaths GeoPackage and the Slope Parquet file. Common column aliases are resolved automatically.

### 3.1 Flowlines GeoPackage (`--flowlines_path`)
* Format: OGC GeoPackage (`.gpkg`)
* Default Layer: `flowpaths` (set via `--flowpath_layer`)

| Expected Column Name | Data Type | Units / Range | Status | Description |
| :--- | :--- | :--- | :--- | :--- |
| `flowpath_id` | `int64` / `int32` | Integer ID | Required | Primary unique flowline identifier. |
| `flowpath_toid` | `int64` / `int32` | Integer ID | Required | Downstream target reach ID for DAG traversal. |
| `mainstemlp` | `int64` / `int32` | Integer ID | Required | Mainstem Level Path ID for GMRF regularization. |
| `hydroseq` | `int64` / `int32` | Integer Sequence | Required | Hydrological routing sequence number. |
| `totdasqkm` | `float64` / `float32` | $\text{km}^2$ ($>0$) | Required | Total upstream contributing drainage area. |
| `arb_sum` | `float64` / `float32` | $\text{km}$ ($\ge 0$) | Required | Total upstream accumulated stream length. |
| `pathlength` | `float64` / `float32` | $\text{km}$ ($\ge 0$) | Required | Flowpath distance to terminal outlet. |
| `lengthkm` | `float64` / `float32` | $\text{km}$ ($>0$) | Required | Segment length (calculated from geometry if missing). |
| `areasqkm` | `float64` / `float32` | $\text{km}^2$ ($\ge 0$) | Required | Local catchment drainage area. |
| `streamorder` | `int64` / `int32` | $1 - 10$ | Required | Strahler stream order. |
| `terminalfl` | `int64` / `int32` | 0 or 1 | Required | Terminal reach flag indicator. |
| `geometry` | `LineString` | Projected or EPSG:4326 | Required | Vector geometry used for reach sinuosity. |

### 3.2 Slope Parquet File (`--slopes_path`)
* Format: Apache Parquet (`.parquet`)

| Expected Column Name | Acceptable Aliases | Data Type | Units | Description |
| :--- | :--- | :--- | :--- | :--- |
| `flowpath_id` | `id` | `int64` / `int32` | Identifier | Reach ID to match against Flowpaths. |
| `slope` | `slope_m_m`, `final_regularized_slope` | `float64` / `float32` | $\text{m/m}$ ($>0$) | Reach energy slope from DEM analysis. |

### 3.3 Model Directory Structure
Model directories (`--tw_model_path`, `--depth_model_path`, `--r_model_path`, `--n_model_path`, `--n_in_model_path`, `--n_out_model_path`) accept local directory paths or `s3://` URIs:
```text
<model_directory>/
  |-- trained_xgboost_model_update_<target>_final.pickle.dat
  |-- transformation_metadata_<target>.json
  |-- final_model_features_<target>.json
  |-- metrics/
  |   \-- median_imput_<target>.parquet
  \-- ensemble/
      |-- ensemble_metadata_n.json
      |-- resnet_model_1.pickle.dat ... resnet_model_N.pickle.dat
      \-- xgb_model_1.pickle.dat ... xgb_model_M.pickle.dat
```

---

## 4. Master Consolidated Output Schema

The prediction engine writes a unified Parquet file to:
`{output_dir}/{process_domain}/deployment/data/final_consolidated_predictions_{process_domain}.parquet`

| Column Header | Data Type | Units / Range | Description |
| :--- | :--- | :--- | :--- |
| `flowpath_id` | `Int64` | Identifier | Primary reach feature identifier. |
| `slope` | `float32` | $\text{m/m}$ ($>0$) | Reach energy slope. |
| `owp_tw_bf_m` | `float32` | meters | Predicted bankfull top width. |
| `owp_y_bf_m` | `float32` | meters | Predicted bankfull channel depth. |
| `owp_r_bf` | `float32` | Dimensionless ($\ge 1.0$) | Predicted Dingman channel shape exponent. |
| `owp_n_single` | `float32` | $\text{s/m}^{1/3}$ ($0.01 - 0.35$) | Regularized overall Manning's roughness. |
| `owp_n_single_lcl` | `float32` | $\text{s/m}^{1/3}$ | Overall roughness lower control limit. |
| `owp_n_single_ucl` | `float32` | $\text{s/m}^{1/3}$ | Overall roughness upper control limit. |
| `confidence_score_n` | `float32` | $\%$ ($0.0 - 100.0$) | Ensemble ML confidence score for roughness. |
| `owp_n_in_channel` | `float32` | $\text{s/m}^{1/3}$ ($0.01 - 0.35$) | Regularized in-channel roughness. |
| `owp_n_in_channel_lcl` | `float32` | $\text{s/m}^{1/3}$ | In-channel roughness lower control limit. |
| `owp_n_in_channel_ucl` | `float32` | $\text{s/m}^{1/3}$ | In-channel roughness upper control limit. |
| `confidence_score_n_in_channel` | `float32` | $\%$ ($0.0 - 100.0$) | In-channel ML confidence score. |
| `owp_n_out_channel` | `float32` | $\text{s/m}^{1/3}$ ($0.01 - 0.35$) | Regularized overbank roughness. |
| `owp_n_out_channel_lcl` | `float32` | $\text{s/m}^{1/3}$ | Overbank roughness lower control limit. |
| `owp_n_out_channel_ucl` | `float32` | $\text{s/m}^{1/3}$ | Overbank roughness upper control limit. |
| `confidence_score_n_out_channel` | `float32` | $\%$ ($0.0 - 100.0$) | Overbank ML confidence score. |

---

## 5. Inference Pipeline Execution

### 5.1 Environment Setup

```bash
# Clone the repository
git clone https://github.com/NOAA-OWP/channel-shape-ML.git
cd channel-shape-ML

# Create Conda environment
conda env create -f environment.yml
conda activate river_ml

# For GPU-accelerated inference:
conda env create -f environment-cuda.yml
conda activate river_ml_cuda
```

### 5.2 CLI Inference Execution

Run inference locally using `run_inference.py`:

```bash
python run_inference.py \
  --flowlines_path "/path/to/flowlines/domain.gpkg" \
  --slopes_path "/path/to/slope.parquet" \
  --output_dir "data/outputs" \
  --process_domain "domain" \
  --tw_model_path "/path/to/tw_model/deployment/models" \
  --y_model_path "/path/to/y_model/deployment/models" \
  --r_model_path "/path/to/r_model/deployment/models" \
  --n_model_path "/path/to/n_model/deployment/models" \
  --n_in_channel_model_path "/path/to/n_in_model/deployment/models" \
  --n_out_channel_model_path "/path/to/n_out_model/deployment/models" \
  --chunk_size 100000
```

To run inference streaming model files directly from Amazon S3:

Option A: With AWS SSO (`aws sso login --profile <profile>`)

```bash
# 1. Authenticate with AWS SSO on host
aws sso login --profile my-sso-profile

# 2. Run inference referencing S3 URIs
python run_inference.py \
  --aws_profile "my-sso-profile" \
  --flowlines_path "s3://your-bucket-name/data/reference.gpkg" \
  --slopes_path "s3://your-bucket-name/data/slope.parquet" \
  --output_dir "s3://your-bucket-name/outputs" \
  --process_domain "domain" \
  --tw_model_path "s3://your-bucket-name/bankfull_topwidth/model" \
  --y_model_path "s3://your-bucket-name/bankfull_depth/model" \
  --r_model_path "s3://your-bucket-name/bankfull_shape/model" \
  --n_model_path "s3://your-bucket-name/manning_single/model" \
  --n_in_channel_model_path "s3://your-bucket-name/manning_in_channel/model" \
  --n_out_channel_model_path "s3://your-bucket-name/manning_out_channel/model" \
  --chunk_size 100000
```

Option B: With Static AWS IAM Keys
```bash
export AWS_ACCESS_KEY_ID="AKIA..."
export AWS_SECRET_ACCESS_KEY="..."
export AWS_DEFAULT_REGION="us-east-1"

python run_inference.py \
  --flowlines_path "s3://your-bucket-name/data/reference.gpkg" \
  --slopes_path "s3://your-bucket-name/data/slope.parquet" \
  --output_dir "s3://your-bucket-name/outputs" \
  --process_domain "domain" \
  --tw_model_path "s3://your-bucket-name/bankfull_topwidth/model" \
  --y_model_path "s3://your-bucket-name/bankfull_depth/model" \
  --r_model_path "s3://your-bucket-name/bankfull_shape/model" \
  --n_model_path "s3://your-bucket-name/manning_single/model" \
  --n_in_channel_model_path "s3://your-bucket-name/manning_in_channel/model" \
  --n_out_channel_model_path "s3://your-bucket-name/manning_out_channel/model" \
  --chunk_size 100000
```
Option C: With AWS EC2 / Batch / ECS IAM Instance Roles
On AWS EC2 or AWS Batch instances with attached IAM instance profile roles, no keys or profiles are passed:

```bash
python run_inference.py \
  --flowlines_path "s3://your-bucket-name/data/reference.gpkg" \
  --slopes_path "s3://your-bucket-name/data/slope.parquet" \
  --output_dir "s3://your-bucket-name/outputs" \
  --process_domain "domain" \
  --tw_model_path "s3://your-bucket-name/bankfull_topwidth/model" \
  --y_model_path "s3://your-bucket-name/bankfull_depth/model" \
  --r_model_path "s3://your-bucket-name/bankfull_shape/model" \
  --n_model_path "s3://your-bucket-name/manning_single/model" \
  --n_in_channel_model_path "s3://your-bucket-name/manning_in_channel/model" \
  --n_out_channel_model_path "s3://your-bucket-name/manning_out_channel/model" \
  --chunk_size 100000
```

### 5.3 Docker Execution

Build and run using Docker:
```bash
docker buildx build -t river_ml_pipeline:latest .
```
Option A: Local Host Volume Mounts (Windows WSL2, Mac, Linux)
```bash
docker run --rm \
  --shm-size=8g \
  -v "/wroking/directory":/data/flowlines:ro \
  -v "/wroking/directory":/data/slopes:ro \
  -v "$(pwd)/data/models/conus":/app/models/conus:ro \
  -v "$(pwd)/data/outputs/oconus":/app/outputs/oconus \
  river_ml_pipeline:latest \
  --flowlines_path "/data/flowlines/path/to/flowlines.gpkg" \
  --slopes_path "/data/slopes/path/to/slope.parquet" \
  --output_dir "/app/outputs/oconus" \
  --process_domain "domain" \
  --tw_model_path "/app/models/conus/superconus/tw_model/models" \
  --y_model_path "/app/models/conus/superconus/y_model/models" \
  --r_model_path "/app/models/conus/superconus/r_model/models" \
  --n_model_path "/app/models/conus/superconus/n_model/models" \
  --n_in_channel_model_path "/app/models/conus/superconus/n_in_model/models" \
  --n_out_channel_model_path "/app/models/conus/superconus/n_out_model/models" \
  --chunk_size 100000
```
Option B: Docker with AWS SSO Mount (`$HOME/.aws`)
```bash
docker run --rm \
  --shm-size=8g \
  -v "$HOME/.aws":/root/.aws:ro \
  -e AWS_PROFILE="my-sso-profile" \
  -e AWS_DEFAULT_REGION="us-east-1" \
  river_ml_pipeline:latest \
  --flowlines_path "/data/flowlines/path/to/flowlines.gpkg" \
  --slopes_path "/data/slopes/path/to/slope.parquet" \
  --output_dir "/app/outputs/oconus" \
  --process_domain "domain" \
  --tw_model_path "s3://your-bucket-name/path/bankfull_topwidth/model" \
  --y_model_path "s3://your-bucket-name/path/bankfull_depth/model" \
  --r_model_path "s3://your-bucket-name/path/bankfull_shape/model" \
  --n_model_path "s3://your-bucket-name/path/manning_single/model" \
  --n_in_channel_model_path "s3://your-bucket-name/path/manning_in_channel/model" \
  --n_out_channel_model_path "s3://your-bucket-name/path/manning_out_channel/model" \
  --chunk_size 100000 
```
---

## 6. Production Release Notes (v1.0.0)

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

---

## 7. Evolution and Development Roadmap

| Version | Status | Architectural Scope | Key Capabilities |
| :--- | :--- | :--- | :--- |
| **v1.0.0** | Current Production | Foundational ML pipelines (`channel-WD`, `channel-shape`, `preprocess`) | - 50 candidate model screening<br>- Distillation of 400+ attributes to 60 predictors<br>- Bankfull width, depth, and Dingman r CLI |
| **v2.0.0-alpha** | Active Integration | Modernized inference engine (`src/`, DAG architecture) | - 6-stage sequential prediction DAG (TW to Y to r to n)<br>- GMRF topological regularization<br>- Decoupled AWS S3 and local model loading<br>- Extention to domains outside CONUS <br>- Using all MIP, HydroSWOT, and USGS gague data|
| **v2.0.0** | Target Milestone | Full Next-Generation ML suite | - Refactored v2 preprocessing pipeline<br>- Distributed multi-model training pipelines<br>- Automated USGS gauge, MIP, and HydroSWOT accuracy benchmarks<br>- Full automated XAI intergration into all pipelines |

---

## 8. Contributing and Code Standards

Contributions from NOAA, academic partners, and the hydrologic modeling community are welcome.
1. Check the [Issue Tracker](https://github.com/NOAA-OWP/channel-shape-ML/issues) for planned milestones.
2. Review [CONTRIBUTING.md](CONTRIBUTING.md) for branch guidelines and pull request instructions.

Project Contacts:
* Lead Developer: Arash Modaresi Rad (arash.rad@noaa.gov)
* Project Oversight: Fernando Salas (fernando.salas@noaa.gov)
* Affiliation: National Oceanic and Atmospheric Administration (NOAA), National Water Center, Office of Water Prediction (OWP)

---

## 9. Open Source Licensing and Disclaimer

This project is licensed under the Apache License, Version 2.0. See the [LICENSE](LICENSE) file for complete details.

### NOAA Scientific Disclaimer
This repository is a scientific product and is not official communication of the National Oceanic and Atmospheric Administration, or the United States Department of Commerce. All NOAA GitHub project code is provided on an 'as is' basis and the user assumes responsibility for its use. Any claims against the Department of Commerce or Department of Commerce bureaus stemming from the use of this GitHub project will be governed by all applicable Federal law. Any reference to specific commercial products, processes, or services by service mark, trademark, manufacturer, or otherwise, does not constitute or imply their endorsement, recommendation or favoring by the Department of Commerce. See [TERMS.md](TERMS.md).

---

## 10. References and Citations

1. Dingman, S. L. (2007). Analytical derivation of at-a-station hydraulic-geometry relations. Journal of Hydrology, 334(1-2), 17-27. https://doi.org/10.1016/j.jhydrol.2006.10.033
2. Leopold, L. B., and Maddock, T. (1953). The hydraulic geometry of stream channels and some physiographic implications. US Geological Survey Professional Paper, 252. https://doi.org/10.3133/pp252
3. Modaresi Rad, A., Johnson, J. M., Ghahremani, Z., Coll, J., & Frazier, N. (2024). Enhancing river channel dimension estimation: A machine learning approach leveraging the National Water Model, hydrographic networks, and landscape characteristics. Journal of Geophysical Research: Machine Learning and Computation, 1(4), e2024JH000173.
4. Lin, P., et al. (2020). High-resolution global channel geometry and its impact on hydrodynamics. Geophysical Research Letters, 47(11), e2019GL086405.
5. Blackburn-Lynch, W., et al. (2017). Development of regional hydraulic geometry curves for the National Water Model. JAWRA, 53(4), 903-918.
