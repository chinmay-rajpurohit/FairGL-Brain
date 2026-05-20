from nilearn import datasets
import os

abide = datasets.fetch_abide_pcp(
    data_dir="data",
    derivatives=["rois_aal"],
    pipeline="cpac",
    band_pass_filtering=True,
    global_signal_regression=False,
    quality_checked=True
)

print("Number:", len(abide.rois_aal))
print("First item:", abide.rois_aal[0])
print("Type:", type(abide.rois_aal[0]))

path = str(abide.rois_aal[0])
print("Exists:", os.path.exists(path))
print("Path:", path)

print("\nFirst 10 lines:")
with open(path, "r", encoding="utf-8", errors="ignore") as f:
    for i in range(10):
        print(repr(f.readline()))