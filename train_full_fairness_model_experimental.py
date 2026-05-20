import pickle
import os
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
import pandas as pd
import numpy as np

from models.graph_text_model import FairnessAwareGraphLanguageModel
from utils.fairness_metrics import binary_classification_metrics, group_metrics

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


class BrainGraphTextDataset(Dataset):
    def __init__(
        self,
        graphs,
        adjs,
        text_embeddings,
        labels,
        sexes,
        age_group_ids,
        site_ids,
        indices
    ):
        self.graphs = graphs
        self.adjs = adjs
        self.text_embeddings = text_embeddings
        self.labels = labels
        self.sexes = sexes
        self.age_group_ids = age_group_ids
        self.site_ids = site_ids
        self.indices = indices

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return (
            self.graphs[idx],
            self.adjs[idx],
            self.text_embeddings[idx],
            torch.tensor(self.labels[idx], dtype=torch.long),
            torch.tensor(self.sexes[idx], dtype=torch.long),
            torch.tensor(self.age_group_ids[idx], dtype=torch.long),
            torch.tensor(self.site_ids[idx], dtype=torch.long),
            torch.tensor(self.indices[idx], dtype=torch.long)
        )


def contrastive_alignment_loss(graph_z, text_z, temperature=0.1):
    graph_z = F.normalize(graph_z, dim=1)
    text_z = F.normalize(text_z, dim=1)

    logits = torch.matmul(graph_z, text_z.T) / temperature
    labels = torch.arange(graph_z.size(0), device=graph_z.device)

    loss_g2t = F.cross_entropy(logits, labels)
    loss_t2g = F.cross_entropy(logits.T, labels)

    return (loss_g2t + loss_t2g) / 2


def fairness_loss_by_group(logits, labels, group_ids):
    group_losses = []

    for group_id in torch.unique(group_ids):
        group_mask = group_ids == group_id

        if group_mask.sum() == 0:
            continue

        group_losses.append(F.cross_entropy(logits[group_mask], labels[group_mask]))

    if len(group_losses) < 2:
        return torch.tensor(0.0, device=logits.device)

    group_losses = torch.stack(group_losses)

    return torch.mean(torch.abs(group_losses - group_losses.mean()))


def fairness_loss_all_attributes(logits, labels, sex, age_group, site_id):
    sex_loss = fairness_loss_by_group(logits, labels, sex)
    age_loss = fairness_loss_by_group(logits, labels, age_group)
    site_loss = fairness_loss_by_group(logits, labels, site_id)

    return 0.6 * sex_loss + 0.3 * age_loss + 0.1 * site_loss


def adversarial_sensitive_loss(sensitive_logits, sex, age_group, site_id):
    sex_targets = sex - 1
    sex_loss = F.cross_entropy(sensitive_logits["sex"], sex_targets)
    age_loss = F.cross_entropy(sensitive_logits["age_group"], age_group)
    site_loss = F.cross_entropy(sensitive_logits["site"], site_id)

    return (sex_loss + age_loss + site_loss) / 3


def population_smoothness_loss(shared_z, batch_indices, population_adj):
    shared_z = F.normalize(shared_z, dim=1)
    batch_population_adj = population_adj[batch_indices][:, batch_indices]
    edge_mask = batch_population_adj > 0

    if edge_mask.sum() == 0:
        return torch.tensor(0.0, device=shared_z.device)

    similarity = torch.matmul(shared_z, shared_z.T)
    weighted_distance = batch_population_adj[edge_mask] * (1.0 - similarity[edge_mask])

    return weighted_distance.mean()


def collect_predictions(model, loader, threshold=0.5, return_probs=False):
    all_probs = []
    all_preds = []
    all_labels = []
    all_sexes = []
    all_age_groups = []
    all_site_ids = []

    model.eval()

    with torch.no_grad():
        for x, adj, text_emb, y, sex, age_group, site_id, batch_idx in loader:
            x = x.to(device)
            adj = adj.to(device)
            text_emb = text_emb.to(device)

            outputs = model(x, adj, text_emb, roi_prior_embeddings)
            logits = outputs["logits"]
            probs = torch.softmax(logits, dim=1)[:, 1]
            preds = (probs >= threshold).long()

            all_probs.extend(probs.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.numpy())
            all_sexes.extend(sex.numpy())
            all_age_groups.extend(age_group.numpy())
            all_site_ids.extend(site_id.numpy())

    if return_probs:
        return all_probs, all_preds, all_labels, all_sexes, all_age_groups, all_site_ids

    return all_preds, all_labels, all_sexes, all_age_groups, all_site_ids


def choose_fairness_threshold(probs, labels, sexes, age_groups, site_ids):
    best_threshold = 0.5
    best_score = -1.0
    best_summary = {}

    for threshold_int in range(10, 91):
        threshold = threshold_int / 100
        preds = [1 if prob >= threshold else 0 for prob in probs]
        acc = accuracy_score(labels, preds)
        f1 = f1_score(labels, preds, zero_division=0)
        sex_gap = group_metrics(labels, preds, sexes)["accuracy_gap"]
        age_gap = group_metrics(labels, preds, age_groups)["accuracy_gap"]
        site_gap = group_metrics(labels, preds, site_ids)["accuracy_gap"]
        score = acc + 0.25 * f1 - 0.75 * sex_gap - 0.25 * age_gap - 0.05 * site_gap

        if score > best_score:
            best_score = score
            best_threshold = threshold
            best_summary = {
                "accuracy": acc,
                "f1": f1,
                "sex_gap": sex_gap,
                "age_gap": age_gap,
                "site_gap": site_gap,
                "score": score
            }

    return best_threshold, best_summary


with open("processed/abide_graph_dataset.pkl", "rb") as f:
    data = pickle.load(f)

text_embeddings = torch.load("processed/text_embeddings.pt")
roi_prior_embeddings = torch.load("processed/roi_prior_embeddings.pt")
population_graph = torch.load("processed/population_similarity_graph.pt", map_location="cpu")

graphs = data["graphs"]
adjs = data["adjs"]
labels = data["labels"]
sexes = data["sexes"]
age_group_ids = data["age_group_ids"]
site_ids = data["site_ids"]

print("Total samples:", len(labels))
print("Text embeddings:", text_embeddings.shape)
print("ROI prior embeddings:", roi_prior_embeddings.shape)
print("Population graph adjacency:", population_graph["adjacency"].shape)

indices = list(range(len(labels)))

train_idx, test_idx = train_test_split(
    indices,
    test_size=0.2,
    stratify=labels,
    random_state=42
)

train_dataset = BrainGraphTextDataset(
    [graphs[i] for i in train_idx],
    [adjs[i] for i in train_idx],
    text_embeddings[train_idx],
    [labels[i] for i in train_idx],
    [sexes[i] for i in train_idx],
    [age_group_ids[i] for i in train_idx],
    [site_ids[i] for i in train_idx],
    train_idx
)

test_dataset = BrainGraphTextDataset(
    [graphs[i] for i in test_idx],
    [adjs[i] for i in test_idx],
    text_embeddings[test_idx],
    [labels[i] for i in test_idx],
    [sexes[i] for i in test_idx],
    [age_group_ids[i] for i in test_idx],
    [site_ids[i] for i in test_idx],
    test_idx
)

train_label_tensor = torch.tensor([labels[i] for i in train_idx], dtype=torch.long)
class_counts = torch.bincount(train_label_tensor, minlength=2).float()

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
train_eval_loader = DataLoader(train_dataset, batch_size=16, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

model = FairnessAwareGraphLanguageModel(
    node_feat_dim=116,
    graph_hidden_dim=64,
    text_dim=768,
    shared_dim=64,
    num_classes=2,
    roi_prior_dim=768,
    roi_hidden_dim=32,
    num_sex_groups=2,
    num_age_groups=3,
    num_sites=len(set(site_ids))
).to(device)
roi_prior_embeddings = roi_prior_embeddings.to(device)
population_adj = population_graph["adjacency"].to(device)

print("Train class counts:", class_counts.tolist())

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

epochs = 30
lambda_align = 0.005
lambda_fair = 0.1
lambda_pop = 0.0
lambda_adv = 0.0

print("Starting training...")

for epoch in range(epochs):
    model.train()

    total_loss = 0.0
    total_cls_loss = 0.0
    total_align_loss = 0.0
    total_fair_loss = 0.0
    total_pop_loss = 0.0
    total_adv_loss = 0.0

    for x, adj, text_emb, y, sex, age_group, site_id, batch_idx in train_loader:
        x = x.to(device)
        adj = adj.to(device)
        text_emb = text_emb.to(device)
        y = y.to(device)
        sex = sex.to(device)
        age_group = age_group.to(device)
        site_id = site_id.to(device)
        batch_idx = batch_idx.to(device)

        optimizer.zero_grad()

        outputs = model(
            x,
            adj,
            text_emb,
            roi_prior_embeddings,
            adv_lambda=lambda_adv
        )
        logits = outputs["logits"]
        graph_z = outputs["graph_z"]
        text_z = outputs["text_z"]
        shared_z = outputs["shared_z"]

        cls_loss = F.cross_entropy(logits, y)
        align_loss = contrastive_alignment_loss(graph_z, text_z)
        fair_loss = fairness_loss_all_attributes(logits, y, sex, age_group, site_id)
        pop_loss = population_smoothness_loss(shared_z, batch_idx, population_adj)
        adv_loss = adversarial_sensitive_loss(
            outputs["sensitive_logits"],
            sex,
            age_group,
            site_id
        )

        loss = (
            cls_loss
            + lambda_align * align_loss
            + lambda_fair * fair_loss
            + lambda_pop * pop_loss
            + lambda_adv * adv_loss
        )

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_cls_loss += cls_loss.item()
        total_align_loss += align_loss.item()
        total_fair_loss += fair_loss.item()
        total_pop_loss += pop_loss.item()
        total_adv_loss += adv_loss.item()

    print(
        f"Epoch {epoch + 1}/{epochs} | "
        f"Loss: {total_loss:.4f} | "
        f"Cls: {total_cls_loss:.4f} | "
        f"Align: {total_align_loss:.4f} | "
        f"Fair: {total_fair_loss:.4f} | "
        f"Pop: {total_pop_loss:.4f} | "
        f"Adv: {total_adv_loss:.4f}"
    )

print("Training finished.")

model.eval()

train_probs, _, train_labels, train_sexes, train_age_groups, train_site_ids = collect_predictions(
    model,
    train_eval_loader,
    return_probs=True
)
best_threshold, threshold_summary = choose_fairness_threshold(
    train_probs,
    train_labels,
    train_sexes,
    train_age_groups,
    train_site_ids
)

all_preds, all_labels, all_sexes, all_age_groups, all_site_ids = collect_predictions(
    model,
    test_loader,
    threshold=best_threshold
)

overall_metrics = binary_classification_metrics(all_labels, all_preds)
overall_acc = overall_metrics["accuracy"]
overall_f1 = overall_metrics["f1"]

sex_metrics = group_metrics(all_labels, all_preds, all_sexes)
age_metrics = group_metrics(all_labels, all_preds, all_age_groups)
site_metrics = group_metrics(all_labels, all_preds, all_site_ids)

sex_stats = sex_metrics["per_group"]
age_stats = age_metrics["per_group"]
site_stats = site_metrics["per_group"]
sex_acc_gap = sex_metrics["accuracy_gap"]
age_acc_gap = age_metrics["accuracy_gap"]
site_acc_gap = site_metrics["accuracy_gap"]

male_acc = sex_stats.get(1, {"accuracy": 0.0})["accuracy"]
female_acc = sex_stats.get(2, {"accuracy": 0.0})["accuracy"]
male_n = sex_stats.get(1, {"n": 0})["n"]
female_n = sex_stats.get(2, {"n": 0})["n"]

print("\n========== RESULTS ==========")
print("Model: Full Fairness-Aware Model")
print("Overall Accuracy:", overall_acc)
print("Overall F1:", overall_f1)
print("Overall Precision:", overall_metrics["precision"])
print("Overall Recall:", overall_metrics["recall"])
print("Selected Threshold:", best_threshold)
print("Train Threshold Summary:", threshold_summary)
print("Prediction counts:", pd.Series(all_preds).value_counts().sort_index().to_dict())
print("Male Accuracy:", male_acc)
print("Female Accuracy:", female_acc)
print("Sex Accuracy Gap:", sex_acc_gap)
print("Age Group Accuracy Gap:", age_acc_gap)
print("Site Accuracy Gap:", site_acc_gap)
print("Sex Demographic Parity Gap:", sex_metrics["demographic_parity_gap"])
print("Sex Equal Opportunity Gap:", sex_metrics["equal_opportunity_gap"])
print("Age Worst Group Accuracy:", age_metrics["worst_group_accuracy"])
print("Site Worst Group Accuracy:", site_metrics["worst_group_accuracy"])
print("Male test samples:", male_n)
print("Female test samples:", female_n)
print("Age group stats:", age_stats)
print("Site stats:", site_stats)

os.makedirs("results", exist_ok=True)

new_result = pd.DataFrame({
    "Model": ["Full Fairness-Aware Model"],
    "Accuracy": [overall_acc],
    "F1": [overall_f1],
    "Male Accuracy": [male_acc],
    "Female Accuracy": [female_acc],
    "Accuracy Gap": [sex_acc_gap],
    "Sex Accuracy Gap": [sex_acc_gap],
    "Age Group Accuracy Gap": [age_acc_gap],
    "Site Accuracy Gap": [site_acc_gap],
    "Precision": [overall_metrics["precision"]],
    "Recall": [overall_metrics["recall"]],
    "Threshold": [best_threshold],
    "Sex Demographic Parity Gap": [sex_metrics["demographic_parity_gap"]],
    "Sex Equal Opportunity Gap": [sex_metrics["equal_opportunity_gap"]],
    "Age Worst Group Accuracy": [age_metrics["worst_group_accuracy"]],
    "Site Worst Group Accuracy": [site_metrics["worst_group_accuracy"]],
    "Male Test Samples": [male_n],
    "Female Test Samples": [female_n]
})

result_path = "results/experiment_results.csv"

if os.path.exists(result_path):
    old_results = pd.read_csv(result_path)
    old_results = old_results[old_results["Model"] != "Full Fairness-Aware Model"]
    final_results = pd.concat([old_results, new_result], ignore_index=True)
else:
    final_results = new_result

final_results.to_csv(result_path, index=False)

print("\nSaved results to:", result_path)
print("\nCurrent Experiment Table:")
print(final_results)
