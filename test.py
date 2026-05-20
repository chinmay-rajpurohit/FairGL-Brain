from nilearn import datasets

abide = datasets.fetch_abide_pcp(
    derivatives=['rois_aal'],
    pipeline='cpac',
    band_pass_filtering=True,
    global_signal_regression=False,
    quality_checked=True
)

print(abide.keys())
print(len(abide.rois_aal))
print(abide.phenotypic[0])