---
title: Model Architecture 
---

# Model Architecture

The prediction pipeline for **n-in** utilizes the identical feature set and cascaded modeling structure as n-single (Stage 4). 

## Target Derivation via Chow's Equation
The critical difference lies in the target variable ($\mathbf{y}$). Instead of training on an effective roughness optimized over the entire rating curve, the model is trained on `n_in` targets. These targets are derived during data preprocessing using Chow's compound channel equations, isolating the below-bankfull flow resistance.

## Architecture Reuse
Because the fundamental geomorphic drivers (slope, drainage area, topology, hydraulic radius) govern both bulk and in-channel resistance, we retain the hybrid ensemble:
(a) Attention-Augmented Deep Residual Tabular Network
(b) Gradient-Boosted Decision Forest

The models learn a distinct mapping tailored to main-channel bedforms and grain friction without being confounded by floodplain vegetation.
