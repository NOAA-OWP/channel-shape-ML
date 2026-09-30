---
title: Literature
---

# Overbank Roughness (n-out)

In compound channel hydraulics and 2D flood inundation modeling, distinguishing between main channel conveyance and overbank floodplain resistance is essential. **Overbank roughness ($n_{\text{out}}$)** governs hydraulic friction across the active floodplain once flow exceeds the bankfull stage ($d > d_{bf}$).

---

## Physical Drivers of Floodplain Resistance

Unlike in-channel flow resistance, which is predominantly governed by bed grain size and mobile bedforms, overbank resistance is dictated by macro-scale floodplain attributes and boundary shear mechanisms:

1. **Vegetation Form Drag & Canopy Resistance**: </br>
    As inundation depth increases across the floodplain, relative submergence transitions from unsubmerged vegetation drag to partially submerged boundary layer flow. This structural shift allows fast moving water to skim over the plant canopy, effectively decreasing Manning's roughness ($n_{\text{out}}$)

2. **Floodplain Micro- and Macro-Topography**: </br>
   The complex topography of the floodplain including rolling ridges, abandoned channels, dips, and natural riverbanks acts as a natural brake on moving water. As floodwaters pass over these uneven features, the flow is forced to repeatedly widen, narrow, and swirl into secondary currents. This continuous churning and disruption of the water's path effectively strips the flood of its kinetic energy, slowing it down.

---

## Chow's Compound Channel Method

To parameterize $n_{\text{out}}$ without empirical guesswork, we apply Chow's compound channel method. The rating curve is segmented at the bankfull depth breakpoint:

$$
Q_{\text{total}} = Q_{\text{in}} + Q_{\text{out}} = \frac{1}{n_{\text{in}}} A_{\text{in}} R_{\text{in}}^{2/3} S^{1/2} + \frac{1}{n_{\text{out}}} A_{\text{out}} R_{\text{out}}^{2/3} S^{1/2}
$$

By resolving $Q_{\text{in}}$ from the below-bankfull geometry, the excess discharge and floodplain conveyance are isolated to estimate the true overbank friction coefficient $n_{\text{out}}$ independently.

---

## Key References

1. **Arcement, G. J., & Schneider, V. R. (1989).** *Guide for selecting Manning's roughness coefficients for natural channels and flood plains.* U.S. Geological Survey Water-Supply Paper 2339.
2. **Chow, V. T. (1959).** *Open-Channel Hydraulics.* McGraw-Hill, New York.
3. **Nepf, H. M. (1999).** *Drag, turbulence, and diffusion in flow through emergent vegetation.* Water Resources Research, 35(2), 479-489.
