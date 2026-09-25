---
title: Uncertainty Quantification
---

# Uncertainty Quantification

Our framework explicitly tracks epistemic and aleatoric uncertainty for in-channel roughness. Model confidence is quantified using an ML Confidence Score ($CS$), defined as:

$$
CS = \exp(-\gamma \cdot CV) \times 100\%
$$

where $\gamma = 3.0$ and $CV = \frac{\sigma}{\mu}$ is the coefficient of variation across the multi-seed ensemble. The ensemble spread yields the median ($q_{50}$), lower confidence limit ($q_{05}$), and upper confidence limit ($q_{95}$).

---

=== "SUPERCONUS"
    ### SUPERCONUS Uncertainty Profiles
    The contiguous United States domain highlights in-channel predictive confidence across diverse physiographic provinces.
    
    ![Confidence Score Distribution](../../assets/images/n/v2.0.0/n-in/uncertainty/superconus/full_network_confidence_score_distribution.png){ loading=lazy }
    ![Reach Confidence](../../assets/images/n/v2.0.0/n-in/uncertainty/superconus/full_network_conus_reach_confidence.png){ loading=lazy }
    ![Median n-in](../../assets/images/n/v2.0.0/n-in/uncertainty/superconus/full_network_conus_n_median.png){ loading=lazy }
    ![Lower Confidence Limit](../../assets/images/n/v2.0.0/n-in/uncertainty/superconus/full_network_conus_n_smooth_lcl.png){ loading=lazy }
    ![Upper Confidence Limit](../../assets/images/n/v2.0.0/n-in/uncertainty/superconus/full_network_conus_n_smooth_ucl.png){ loading=lazy }

=== "Alaska"
    ### Alaska Uncertainty Profiles
    In Alaska, complex glaciated headwaters and braided river channels none existant in CONUS trraining data lower  the predictive confidence bounds.

    ![Confidence Score Distribution](../../assets/images/n/v2.0.0/n-in/uncertainty/ak/full_network_confidence_score_distribution.png){ loading=lazy }
    ![Reach Confidence](../../assets/images/n/v2.0.0/n-in/uncertainty/ak/full_network_conus_reach_confidence.png){ loading=lazy }
    ![Median n-in](../../assets/images/n/v2.0.0/n-in/uncertainty/ak/full_network_conus_n_median.png){ loading=lazy }
    ![Lower Confidence Limit](../../assets/images/n/v2.0.0/n-in/uncertainty/ak/full_network_conus_n_smooth_lcl.png){ loading=lazy }
    ![Upper Confidence Limit](../../assets/images/n/v2.0.0/n-in/uncertainty/ak/full_network_conus_n_smooth_ucl.png){ loading=lazy }

=== "Hawaii"
    ### Hawaii Uncertainty Profiles
    Hawaii's steep volcanic catchments and rapid streamflow response exhibit distinct in-channel hydraulic resistance patterns.

    ![Confidence Score Distribution](../../assets/images/n/v2.0.0/n-in/uncertainty/hi/full_network_confidence_score_distribution.png){ loading=lazy }
    ![Reach Confidence](../../assets/images/n/v2.0.0/n-in/uncertainty/hi/full_network_conus_reach_confidence.png){ loading=lazy }
    ![Median n-in](../../assets/images/n/v2.0.0/n-in/uncertainty/hi/full_network_conus_n_median.png){ loading=lazy }
    ![Lower Confidence Limit](../../assets/images/n/v2.0.0/n-in/uncertainty/hi/full_network_conus_n_smooth_lcl.png){ loading=lazy }
    ![Upper Confidence Limit](../../assets/images/n/v2.0.0/n-in/uncertainty/hi/full_network_conus_n_smooth_ucl.png){ loading=lazy }

=== "PRVI"
    ### Puerto Rico & Virgin Islands (PRVI) Uncertainty Profiles
    Tropical steep montane stream networks in PRVI maintain tight confidence intervals along well-defined channel corridors.

    ![Confidence Score Distribution](../../assets/images/n/v2.0.0/n-in/uncertainty/prvi/full_network_confidence_score_distribution.png){ loading=lazy }
    ![Reach Confidence](../../assets/images/n/v2.0.0/n-in/uncertainty/prvi/full_network_conus_reach_confidence.png){ loading=lazy }
    ![Median n-in](../../assets/images/n/v2.0.0/n-in/uncertainty/prvi/full_network_conus_n_median.png){ loading=lazy }
    ![Lower Confidence Limit](../../assets/images/n/v2.0.0/n-in/uncertainty/prvi/full_network_conus_n_smooth_lcl.png){ loading=lazy }
    ![Upper Confidence Limit](../../assets/images/n/v2.0.0/n-in/uncertainty/prvi/full_network_conus_n_smooth_ucl.png){ loading=lazy }
