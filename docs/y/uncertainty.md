---
title: Uncertainty Quantification
---

# Uncertainty Quantification: Depth (Stage 2)

Spatial flowline prediction maps display the reach-scale distribution of bankfull depth ($d_{bf}$) across diverse hydrologic domains.

---

=== "SUPERCONUS"
    ### SUPERCONUS Flowline Predictions
    Predicted bankfull depth across ~2.7M reaches in the contiguous United States.
    
    ![SUPERCONUS Predicted Bankfull Depth Map](../assets/images/y/v2.0.0/uncertainty/superconus/full_network_bankfull_depth_m_predicted_map.png){ loading=lazy }

=== "Alaska"
    ### Alaska Flowline Predictions
    Predicted bankfull depth across Alaskan sub-arctic and glaciated catchments.

    ![Alaska Predicted Bankfull Depth Map](../assets/images/y/v2.0.0/uncertainty/ak/full_network_bankfull_depth_m_predicted_map.png){ loading=lazy }

=== "Hawaii"
    ### Hawaii Flowline Predictions
    Predicted bankfull depth across high-gradient volcanic streams in Hawaii.

    ![Hawaii Predicted Bankfull Depth Map](../assets/images/y/v2.0.0/uncertainty/hi/full_network_bankfull_depth_m_predicted_map.png){ loading=lazy }

=== "PRVI"
    ### Puerto Rico & Virgin Islands (PRVI) Flowline Predictions
    Predicted bankfull depth across tropical montane river networks in PRVI.

    ![PRVI Predicted Bankfull Depth Map](../assets/images/y/v2.0.0/uncertainty/prvi/full_network_bankfull_depth_m_predicted_map.png){ loading=lazy }

---

!!! info "Upcoming Multi-Seed Epistemic Uncertainty Suite"
    The maps above depict flowline predictions from the standalone Stage 2 XGBoost model. The upcoming hybrid ensemble release will deliver full multi-seed epistemic uncertainty quantification, including:
    
    * **ML Confidence Score ($CS$) Distribution Histograms**
    * **Reach-Scale Epistemic Confidence Maps**
    * **Lower ($q_{05}$) and Upper ($q_{95}$) Confidence Limit Flowline Maps**
