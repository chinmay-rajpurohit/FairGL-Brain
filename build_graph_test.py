from nilearn import datasets
import numpy as np

abide = datasets.fetch_abide_pcp(
    data_dir="data",
    derivatives=["rois_aal"],
    pipeline="cpac",
    band_pass_filtering=True,
    global_signal_regression=False,
    quality_checked=True
)

file_path = abide.rois_aal[0]
ts = np.loadtxt(file_path)

print("First ROI file:", file_path)
print("Time-series shape:", ts.shape)

corr = np.corrcoef(ts.T)

print("Connectivity matrix shape:", corr.shape)
print("First 5x5 values:")
print(corr[:5, :5])