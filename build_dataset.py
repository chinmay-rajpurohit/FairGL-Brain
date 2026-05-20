from nilearn import datasets
import csv
import numpy as np
import torch
from tqdm import tqdm
import pickle
import os

os.makedirs("processed", exist_ok=True)

abide = datasets.fetch_abide_pcp(
    data_dir="data",
    derivatives=["rois_aal"],
    pipeline="cpac",
    band_pass_filtering=True,
    global_signal_regression=False,
    quality_checked=True
)

pheno = abide.phenotypic

graphs = []
adjs = []
labels = []
sexes = []
ages = []
sites = []
age_groups = []
age_group_ids = []
site_ids = []
metadata_prompts = []
clinical_prompts = []
sensitive_attrs = []
subject_prompts = []

threshold = 0.3


def get_age_group(age):
    if age < 13:
        return "child", 0
    if age <= 18:
        return "adolescent", 1
    return "adult", 2


def safe_float(value):
    try:
        numeric_value = float(value)
    except (TypeError, ValueError):
        return None

    if np.isnan(numeric_value):
        return None

    return numeric_value


def describe_range(value, low_cutoff, high_cutoff, low_label, mid_label, high_label):
    if value is None:
        return "not available"
    if value < low_cutoff:
        return low_label
    if value > high_cutoff:
        return high_label
    return mid_label


def build_clinical_prompt(row, corr, adj):
    upper_mask = np.triu(np.ones(corr.shape, dtype=bool), k=1)
    upper_corr = corr[upper_mask]
    graph_density = float(adj.sum() / (adj.shape[0] * (adj.shape[0] - 1)))
    mean_abs_connectivity = float(np.mean(np.abs(upper_corr)))

    func_mean_fd = safe_float(row.get("func_mean_fd"))
    func_dvars = safe_float(row.get("func_dvars"))
    func_quality = safe_float(row.get("func_quality"))
    func_snr = safe_float(row.get("anat_snr"))
    eye_status = safe_float(row.get("EYE_STATUS_AT_SCAN"))

    motion_text = describe_range(
        func_mean_fd,
        low_cutoff=0.1,
        high_cutoff=0.25,
        low_label="low estimated head motion",
        mid_label="moderate estimated head motion",
        high_label="elevated estimated head motion"
    )
    dvars_text = describe_range(
        func_dvars,
        low_cutoff=1.0,
        high_cutoff=2.0,
        low_label="low temporal signal variation",
        mid_label="moderate temporal signal variation",
        high_label="elevated temporal signal variation"
    )
    quality_text = describe_range(
        func_quality,
        low_cutoff=0.02,
        high_cutoff=0.05,
        low_label="low functional quality artifact",
        mid_label="moderate functional quality artifact",
        high_label="elevated functional quality artifact"
    )
    snr_text = describe_range(
        func_snr,
        low_cutoff=10.0,
        high_cutoff=20.0,
        low_label="low anatomical signal to noise",
        mid_label="moderate anatomical signal to noise",
        high_label="high anatomical signal to noise"
    )

    if eye_status == 1:
        eye_text = "eyes open during scan"
    elif eye_status == 2:
        eye_text = "eyes closed during scan"
    else:
        eye_text = "eye status not available"

    return (
        "Clinical imaging report: resting-state functional connectivity analysis "
        f"shows mean absolute connectivity of {mean_abs_connectivity:.3f} and "
        f"network density of {graph_density:.3f} at the selected threshold. "
        f"Quality summary indicates {motion_text}, {dvars_text}, "
        f"{quality_text}, and {snr_text}. Scan condition indicates {eye_text}."
    )

for i in tqdm(range(len(abide.rois_aal))):
    try:
        ts = np.asarray(abide.rois_aal[i], dtype=np.float32)

        if ts.ndim != 2 or ts.shape[1] < 2:
            print("Skipping invalid subject:", i)
            continue

        corr = np.corrcoef(ts.T)
        corr = np.nan_to_num(corr).astype(np.float32)

        adj = (np.abs(corr) > threshold).astype(np.float32)
        np.fill_diagonal(adj, 0)

        adj_self = adj + np.eye(adj.shape[0], dtype=np.float32)

        degree = np.sum(adj_self, axis=1)
        degree_inv_sqrt = np.power(degree, -0.5)
        degree_inv_sqrt[np.isinf(degree_inv_sqrt)] = 0.0
        d_inv_sqrt = np.diag(degree_inv_sqrt)
        adj_norm = d_inv_sqrt @ adj_self @ d_inv_sqrt

        row = pheno.iloc[i]

        raw_label = int(row["DX_GROUP"])
        label = 1 if raw_label == 1 else 0

        sex_raw = int(row["SEX"])
        age = float(row["AGE_AT_SCAN"])
        site = str(row["SITE_ID"])
        age_group, age_group_id = get_age_group(age)

        metadata_prompt = (
            "Resting-state fMRI subject represented with AAL116 "
            "functional connectivity features."
        )
        clinical_prompt = build_clinical_prompt(row, corr, adj)
        sensitive_attr = {
            "sex": sex_raw,
            "age": age,
            "age_group": age_group,
            "age_group_id": age_group_id,
            "site": site
        }

        graphs.append(torch.tensor(corr, dtype=torch.float32))
        adjs.append(torch.tensor(adj_norm, dtype=torch.float32))
        labels.append(label)
        sexes.append(sex_raw)
        ages.append(age)
        sites.append(site)
        age_groups.append(age_group)
        age_group_ids.append(age_group_id)
        metadata_prompts.append(metadata_prompt)
        clinical_prompts.append(clinical_prompt)
        subject_prompts.append(f"{metadata_prompt} {clinical_prompt}")
        sensitive_attrs.append(sensitive_attr)

    except Exception as e:
        print("Skipping subject", i, "because:", e)

print("Total subjects saved:", len(graphs))

if len(graphs) == 0:
    raise RuntimeError("No subjects were saved.")

site_to_id = {site: idx for idx, site in enumerate(sorted(set(sites)))}
site_ids = [site_to_id[site] for site in sites]

for sensitive_attr, site_id in zip(sensitive_attrs, site_ids):
    sensitive_attr["site_id"] = site_id

print("Graph feature shape:", graphs[0].shape)
print("Adjacency shape:", adjs[0].shape)
print("Example label:", labels[0])
print("Example sex:", sexes[0])
print("Example age:", ages[0])
print("Example age group:", age_groups[0])
print("Example site:", sites[0])
print("Example site id:", site_ids[0])
print("Example metadata prompt:", metadata_prompts[0])
print("Example sensitive attributes:", sensitive_attrs[0])

dataset = {
    "graphs": graphs,
    "adjs": adjs,
    "labels": labels,
    "sexes": sexes,
    "ages": ages,
    "sites": sites,
    "age_groups": age_groups,
    "age_group_ids": age_group_ids,
    "site_ids": site_ids,
    "site_to_id": site_to_id,
    "sensitive_attrs": sensitive_attrs,
    "metadata_prompts": metadata_prompts,
    "clinical_prompts": clinical_prompts,
    "subject_prompts": subject_prompts,
    "prompts": subject_prompts
}

with open("processed/abide_graph_dataset.pkl", "wb") as f:
    pickle.dump(dataset, f)

with open("processed/metadata_prompts.txt", "w", encoding="utf-8") as f:
    for p in metadata_prompts:
        f.write(p + "\n")

with open("processed/subject_prompts.txt", "w", encoding="utf-8") as f:
    for p in subject_prompts:
        f.write(p + "\n")

with open("processed/clinical_prompts.txt", "w", encoding="utf-8") as f:
    for p in clinical_prompts:
        f.write(p + "\n")

with open("processed/sensitive_attributes.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=["sex", "age", "age_group", "age_group_id", "site", "site_id"]
    )
    writer.writeheader()
    writer.writerows(sensitive_attrs)

print("Saved:")
print("processed/abide_graph_dataset.pkl")
print("processed/metadata_prompts.txt")
print("processed/subject_prompts.txt")
print("processed/clinical_prompts.txt")
print("processed/sensitive_attributes.csv")
