# Data Sources & Coverage

The foundation of our multi-stage pipeline relies on a diverse and comprehensive set of channel geometry and roughness measurements spanning the Contiguous United States (CONUS).

## Training Datasets

| Dataset | Description | Measurements | Domain |
| :--- | :--- | :--- | :--- |
| **USGS Field Measurements** | Streamflow gauging, stage-discharge curves, and channel geometry surveys. | ~thousands | CONUS |
| **HydroSWOT ADCP** | Acoustic Doppler Current Profiler width, depth, and velocity profiles. | ~thousands | CONUS |
| **Ripple HEC-RAS** | Calibrated 1D/2D HEC-RAS rating curves with automated geometry extraction. | ~tens of thousands | CONUS |

!!! note "Data Sparsity & Quality Control"
    While these datasets provide robust coverage across stream orders, observational data in specific geographic regions remains sparse. Rigorous quality control is applied along with a two-stage ML pipline  to filter out unrepresentative or heavily engineered reaches and select only best ones before training.

## Inference Domains

The models trained on CONUS data are operationalized across several diverse domains.

=== "SUPERCONUS"
    The primary deployment domain, encompassing **~2.77 million** flowline reaches across the contiguous United States and parts of Canada/Mexico that drain into the US.
=== "Alaska (AK)"
    Applying the model to Alaska introduces elevated predictive uncertainty due to domain shift from the CONUS training baseline. These arctic and subarctic fluvial systems are governed by distinct hydro-geomorphic processes such as permafrost degradation, seasonal ice-jam dynamics, and braided glacial outwash plains as well as minimal anthropogenic channel modification
=== "Hawaii (HI)"
    Characterized by steep, flashy tropical channels carved into volcanic geology.
=== "PRVI"
    Puerto Rico and the US Virgin Islands feature tropical hydrology often dominated by karst terrain and extreme topographic gradients.

!!! warning "Domain Shift Considerations"
    Applying models trained primarily on CONUS data to domains like Alaska or Hawaii involves transfer learning assumptions. The physics-informed feature engineering (relying on dimensionless ratios and energy proxies) helps mitigate domain shift, but uncertainties in extreme out-of-distribution environments are higher.
