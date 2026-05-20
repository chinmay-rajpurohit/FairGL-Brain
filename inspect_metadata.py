from nilearn import datasets

abide = datasets.fetch_abide_pcp(
    data_dir="data",
    derivatives=["rois_aal"],
    pipeline="cpac",
    band_pass_filtering=True,
    global_signal_regression=False,
    quality_checked=True
)

pheno = abide.phenotypic

print("Type:", type(pheno))
print("Shape:", pheno.shape)

print("\nColumns:")
print(list(pheno.columns))

print("\nFirst row:")
print(pheno.iloc[0])

print("\nImportant fields:")
for col in ["DX_GROUP", "SEX", "AGE_AT_SCAN", "SITE_ID"]:
    print(col, "exists?", col in pheno.columns)
    if col in pheno.columns:
        print(col, "first value:", pheno.iloc[0][col])