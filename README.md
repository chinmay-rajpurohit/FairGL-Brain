# FairGL-Brain

**FairGL-Brain** is a final-year thesis project for fairness-aware graph-language learning in brain functional analysis. The project studies ASD classification on the ABIDE resting-state fMRI dataset using functional connectivity graphs, clinical-style text prompts, ROI-level neurobiological priors, graph-text alignment, and fairness evaluation across sensitive groups.

Full thesis title:

**Fairness-Aware Graph-Language Alignment for Brain Functional Analysis**

## Overview

The framework represents each subject as a brain functional connectivity graph built from AAL116 ROI time-series. It then combines graph representations with BioClinicalBERT-based text embeddings derived from subject-level clinical-style prompts and ROI prior descriptions.

The goal is not only to classify ASD vs Typical Control, but also to evaluate and reduce performance disparities across demographic and acquisition-related groups such as sex, age group, and imaging site.

## Main Components

- **Brain graph construction** from ABIDE resting-state fMRI using Nilearn.
- **AAL116 ROI connectivity matrices** as graph node features.
- **Normalized adjacency matrices** from thresholded functional connectivity.
- **Subject text prompts** based on non-label-leaking imaging and scan-quality information.
- **ROI-level priors** describing neurobiological functions of brain regions.
- **Medical language embeddings** generated using BioClinicalBERT.
- **Graph-text fusion and alignment** using a GCN-style graph encoder and text projection head.
- **Fairness-aware training** using group loss-gap regularization.
- **Fairness evaluation** across sex, age group, and site.
- **Tuned full model** with validation threshold selection and early stopping.

## Repository Structure

```text
FairGL-Brain/
├── build_dataset.py
├── build_population_graph.py
├── generate_text_embeddings.py
├── train_gcn.py
├── train_gcn_fairness.py
├── train_graph_text_fusion.py
├── train_graph_text_alignment.py
├── train_full_fairness_model.py
├── train_full_fairness_model_experimental.py
├── models/
│   ├── gcn_model.py
│   └── graph_text_model.py
├── utils/
│   └── fairness_metrics.py
├── data/
│   └── aal_roi_priors.csv
├── processed/
└── results/
```

Large generated files such as raw ABIDE data, processed tensors, checkpoints, and local virtual environments are intentionally excluded from Git.

## Dataset

The project uses ABIDE through:

```python
nilearn.datasets.fetch_abide_pcp
```

The processed dataset is saved as:

```text
processed/abide_graph_dataset.pkl
```

Each subject contains:

- `graphs`: 116 x 116 functional connectivity matrix
- `adjs`: normalized adjacency matrix
- `labels`: `1 = ASD`, `0 = Typical Control`
- `sexes`: `1 = male`, `2 = female`
- `ages`
- `sites`
- `prompts`

## Pipeline

Run the pipeline in this order:

```powershell
python build_dataset.py
python generate_text_embeddings.py
python build_population_graph.py
```

Then run model experiments:

```powershell
python train_gcn.py
python train_gcn_fairness.py
python train_graph_text_fusion.py
python train_graph_text_alignment.py
python train_full_fairness_model.py
```

## Main Training Script

The main final model is:

```text
train_full_fairness_model.py
```

It includes:

- 70/10/20 train-validation-test split
- early stopping
- validation-based threshold tuning
- fairness-aware threshold score
- graph-text alignment loss
- group fairness loss
- sex, age, and site fairness metrics
- model checkpoint saving

Best checkpoint:

```text
results/best_full_fairness_model_tuned.pt
```

## Fairness Metrics

The project reports:

- Accuracy
- F1 score
- Precision
- Recall
- Male accuracy
- Female accuracy
- Sex accuracy gap
- Sex demographic parity gap
- Sex equal opportunity gap
- Age group accuracy gap
- Age worst-group accuracy
- Site accuracy gap
- Site worst-group accuracy

Site groups with very small test counts are ignored in the tuned site-gap calculation to reduce unstable estimates.

## Environment

Recommended environment:

```text
Python 3.11
PyTorch
TorchVision / TorchAudio if needed
Nilearn
NumPy
Pandas
scikit-learn
Transformers
Matplotlib
TQDM
```

Activate the local environment if using the included local setup:

```powershell
.\mlgpu\Scripts\activate
```

## Notes

- Diagnosis labels are not inserted into text prompts.
- Sensitive attributes are used for fairness evaluation and regularization, not as direct prediction targets.
- Processed tensors and ABIDE data are not committed because they are large/generated artifacts.
- Results can vary slightly across runs because neural training is stochastic.

## Citation

If referencing this project, use:

```text
FairGL-Brain: Fairness-Aware Graph-Language Alignment for Brain Functional Analysis
```
