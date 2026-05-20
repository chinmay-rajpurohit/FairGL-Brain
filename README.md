# Nilearn data folder

This directory is used by Nilearn to store datasets
and atlases downloaded from the internet.
It can be safely deleted.
If you delete it, previously downloaded data will be downloaded again.



## Brain Graph Construction

Each subject is represented as a brain functional connectivity graph constructed from resting-state fMRI using the AAL116 atlas.

- **Nodes:**  
  116 brain Regions of Interest (ROIs), where each node corresponds to one anatomical brain region.

- **Edges:**  
  Functional connectivity between pairs of ROIs computed using Pearson correlation of ROI time-series signals.

.\mlgpu\Scripts\activate