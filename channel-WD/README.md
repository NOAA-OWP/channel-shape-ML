# bankfull_W_D
A machine learning approach for estimation of bankfull width and depth

**Check out our [website](https://sites.google.com/u.boisestate.edu/channel-width-depth?usp=sharing) that provides details of our approuch.** 

- [Repo](#Repository)
  * [Cloning](#Cloning)
  * [Example](#Example)
- [Overview](#Overview)
- [Input data](#Input-data)
  * [Preprocessing](#Preprocess-data)
  * [Discharge] (#Approximation-of-bankfull-and-inchannel-discharge) 


## Repository

### Cloning

```shell
git clone https://github.com/LynkerIntel/bankfull_W_D.git
```

### Example

> you can run the model using following command:
...

## Overview

Here we use the USGS “HYDRoacoustic dataset in support of the Surface Water Oceanographic Topography satellite” mission (HYDRoSWOT) to:

1. Build an ML model that uses climate, land surface, subsurface and topological attributes aggregated over divide and incremental catchements to predict **channel depth and width at bankfull** 

![Channel](assets/images/channel.jpg)

2. Build an ML model that uses climate, land surface, subsurface and topological attributes aggregated over divide and incremental catchements to predict **channel depth and width at inchannel**. This is often the part that is not captured by the DEM or sattelite imagery due to presence of water at all times.

## Input data
The ML model establishes a relation between different attributes that are agrregated at catchment and local area and the FHG coefiencets namely a, b, c, f, k, and m. These include

 1- The reference fabric data 
 - streamorde --> Modified Strahler stream order
 - arbolatesu --> Arbolate sum, the sum of the lengths of all digitized flowlines upstream from the downstream end of the immediate flowline, in kilometers
 - roughness --> Manning's roughness
 - etc.

 2- Soil data
 - clay_mean_0_5 --> % clay 
 - ksat_mean_0_5 --> the effective saturated hydraulic conductivity, (cm hr-1) 
 - theta_s_mean_0_5 --> the saturated soil water content, (cm3 cm-3) 
 - etc.

3- From DEM
- elevation --> elevation (m) 
- slope --> slope 
- US NED Physiographic Diversity
- etc.

4- The StreamCat dataset
- Base flow index
- NLCD
- Road density
- etc.

5- Land surface features
- Leaf area index
- Soil moisture 
- NDVI
- etc.

6- Climate data
- Precipitation 
- Evaporation
- Temperature
- etc.  

7- NWM simulations
- NWM 2.1 quartiles
- NWM 2.1 flood frequnecies


### Preprocess data
- you can filter the dat using following command:
- it will ask for a year date as the lower bound for filttering ADCP data

```shell
python3 ./preprocess/filter_adcp.py
```

- filttering GEE data

```shell
python3 ./preprocess/gee_exports.py
```


### Approximation of bankfull and inchannel discharge

To get an estimate of discharges associated with to bankfull and inchannel conditions we calculate flood frequencies using an algorithem consitent with literature and the flow data is exracted from two USGS query systems. The outputs are is shown below that plots etimated bankfull and inchannel discharge and comapres it to ADCP recorded discharge values:

```shell
python3 ./flood_frequency/ff.py
```

![discharge](assets/images/width_cal.png)
