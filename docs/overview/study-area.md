# Study Area & Inference Domains

The multi-stage ML pipeline relies on extensive data from the Contiguous United States (CONUS) for training, leveraging the robust gauge network and geographic diversity to build generalizable relationships.

## Training Domain: CONUS
CONUS serves as the ideal training environment due abundant availbility of trianing data to its vast array of geomorphic, climatic, and ecological regions ranging from the arid Southwest to the temperate Northeast.

## Inference Domains

We operationalize the pipeline to generate parameters for several massive hydrographic networks:

=== "SUPERCONUS"
    **~2.77M Flowline Reaches**  
    Our primary deployment domain includes the entire CONUS plus transboundary watersheds originating in Canada and Mexico that drain into US territories.

=== "Alaska (AK)"
    The Alaskan domain introduces unique arctic and subarctic hydrological dynamics.  
    **Considerations:** Extensive permafrost affecting groundwater surface water interactions, braided glacial outwash plains, and seasonal ice-jam effects on channel formation, hydraulics.

=== "Hawaii (HI)"
    The Hawaiian islands feature extreme topography and volcanic geology.  
    **Considerations:** Very steep, flashy channels, highly permeable basaltic bedrock, and dense tropical vegetation altering floodplain roughness.

=== "PRVI"
    Puerto Rico and the US Virgin Islands.  
    **Considerations:** Tropical rainfall patterns, steep mountain-to-coast gradients, and karst terrain where surface channels often interact with underground cave systems.

!!! warning "Transfer Learning & Domain Shift"
    Deploying models trained on CONUS to regions like AK, HI, and PRVI relies heavily on the physical basis of our feature engineering. Because features are cast as dimensionless ratios and physics-informed proxies (e.g., Froude number proxies, width-depth ratios), the models are more resilient to domain shift. However, epistemic uncertainty in these out-of-distribution environments remains inherently higher and should be considered when utilizing the confidence scores.
