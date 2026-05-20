import copy
import os
import pickle
import random

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

from models.graph_text_model import GraphTextFusionModel


batch_size = 16
learning_rate = 5e-4
weight_decay = 1e-4
lambda_align = 0.01
lambda_fair = 0.05
hidden_dim = 128
shared_dim = 128
dropout = 0.3

epochs = 100
patience = 15
random_seed = 42
model_name = "Full Fairness-Aware Model Tuned"
checkpoint_path = "results/best_full_fairness_model_tuned.pt"
result_path = "results/experiment_results.csv"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

random.seed(random_seed)
np.random.seed(random_seed)
torch.manual_seed(random_seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(random_seed)


class BrainGraphTextDataset(Dataset):
    def __init__(self, graphs, adjs, text_embeddings, labels, sexes, age_groups, site_ids):
        self.graphs = graphs
        self.adjs = adjs
        self.text_embeddings = text_embeddings
        self.labels = labels
        self.sexes = sexes
        self.age_groups = age_groups
        self.site_ids = site_ids

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return (
            self.graphs[idx],
            self.adjs[idx],
            self.text_embeddings[idx],
            torch.tensor(self.labels[idx], dtype=torch.long),
            torch.tensor(self.sexes[idx], dtype=torch.long),
            torch.tensor(self.age_groups[idx], dtype=torch.long),
            torch.tensor(self.site_ids[idx], dtype=torch.long),
        )


def make_age_group(age):
    if age < 12:
        return 0
    if age < 18:
        return 1
    return 2


def stratified_split(indices, labels, test_size, seed):
    try:
        return train_test_split(
            indices,
            test_size=test_size,
            stratify=[labels[i] for i in indices],
            random_state=seed,
        )
    except ValueError:
        return train_test_split(indices, test_size=test_size, random_state=seed)


def contrastive_alignment_loss(graph_z, text_z, temperature=0.1):
    graph_z = F.normalize(graph_z, dim=1)
    text_z = F.normalize(text_z, dim=1)
    logits = torch.matmul(graph_z, text_z.T) / temperature
    targets = torch.arange(graph_z.size(0), device=graph_z.device)
    loss_g2t = F.cross_entropy(logits, targets)
    loss_t2g = F.cross_entropy(logits.T, targets)
    return (loss_g2t + loss_t2g) / 2


def fairness_loss_by_group(logits, labels, group_ids):
    group_losses = []

    for group_id in torch.unique(group_ids):
        mask = group_ids == group_id
        if mask.sum() == 0:
            continue
        group_losses.append(F.cross_entropy(logits[mask], labels[mask]))

    if len(group_losses) < 2:
        return torch.tensor(0.0, device=logits.device)

    group_losses = torch.stack(group_losses)
    return torch.mean(torch.abs(group_losses - group_losses.mean()))


def fairness_loss(logits, labels, sexes, age_groups):
    sex_loss = fairness_loss_by_group(logits, labels, sexes)
    age_loss = fairness_loss_by_group(logits, labels, age_groups)
    return 0.7 * sex_loss + 0.3 * age_loss


def collect_probabilities(model, loader):
    model.eval()
    probs = []
    labels = []
    sexes = []
    age_groups = []
    site_ids = []

    with torch.no_grad():
        for x, adj, text_emb, y, sex, age_group, site_id in loader:
            x = x.to(device)
            adj = adj.to(device)
            text_emb = text_emb.to(device)

            logits, graph_z, text_z = model(x, adj, text_emb)
            batch_probs = torch.softmax(logits, dim=1)[:, 1]

            probs.extend(batch_probs.cpu().numpy())
            labels.extend(y.numpy())
            sexes.extend(sex.numpy())
            age_groups.extend(age_group.numpy())
            site_ids.extend(site_id.numpy())

    return probs, labels, sexes, age_groups, site_ids


def group_accuracy_stats(labels, preds, group_ids, min_group_size=1):
    stats = {}

    for group_id in sorted(set(int(g) for g in group_ids)):
        group_labels = [
            y for y, g in zip(labels, group_ids)
            if int(g) == group_id
        ]
        group_preds = [
            p for p, g in zip(preds, group_ids)
            if int(g) == group_id
        ]

        if len(group_labels) < min_group_size:
            continue

        stats[group_id] = {
            "accuracy": accuracy_score(group_labels, group_preds),
            "n": len(group_labels),
        }

    accuracies = [value["accuracy"] for value in stats.values()]
    gap = max(accuracies) - min(accuracies) if len(accuracies) >= 2 else 0.0
    worst = min(accuracies) if accuracies else 0.0

    return stats, gap, worst


def sex_fairness_stats(labels, preds, sexes):
    sex_stats, sex_gap, _ = group_accuracy_stats(labels, preds, sexes)

    male_acc = sex_stats.get(1, {"accuracy": 0.0})["accuracy"]
    female_acc = sex_stats.get(2, {"accuracy": 0.0})["accuracy"]

    positive_rates = {}
    true_positive_rates = {}

    for group_id in [1, 2]:
        group_labels = [
            y for y, g in zip(labels, sexes)
            if int(g) == group_id
        ]
        group_preds = [
            p for p, g in zip(preds, sexes)
            if int(g) == group_id
        ]

        if len(group_labels) == 0:
            positive_rates[group_id] = 0.0
            true_positive_rates[group_id] = 0.0
            continue

        positive_rates[group_id] = sum(int(p) == 1 for p in group_preds) / len(group_preds)
        true_positive_rates[group_id] = recall_score(
            group_labels,
            group_preds,
            zero_division=0,
        )

    demographic_parity_gap = abs(positive_rates[1] - positive_rates[2])
    equal_opportunity_gap = abs(true_positive_rates[1] - true_positive_rates[2])

    return {
        "male_accuracy": male_acc,
        "female_accuracy": female_acc,
        "sex_accuracy_gap": sex_gap,
        "sex_demographic_parity_gap": demographic_parity_gap,
        "sex_equal_opportunity_gap": equal_opportunity_gap,
    }


def evaluate_with_threshold(probs, labels, sexes, age_groups, site_ids, threshold):
    preds = [1 if prob >= threshold else 0 for prob in probs]

    sex_metrics = sex_fairness_stats(labels, preds, sexes)
    age_stats, age_gap, age_worst = group_accuracy_stats(labels, preds, age_groups)
    site_stats, site_gap, site_worst = group_accuracy_stats(
        labels,
        preds,
        site_ids,
        min_group_size=5,
    )

    metrics = {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds, zero_division=0),
        "precision": precision_score(labels, preds, zero_division=0),
        "recall": recall_score(labels, preds, zero_division=0),
        "age_group_accuracy_gap": age_gap,
        "age_worst_group_accuracy": age_worst,
        "site_accuracy_gap": site_gap,
        "site_worst_group_accuracy": site_worst,
        "age_group_stats": age_stats,
        "site_stats": site_stats,
        "prediction_counts": pd.Series(preds).value_counts().sort_index().to_dict(),
    }
    metrics.update(sex_metrics)
    return metrics, preds


def tune_threshold(probs, labels, sexes, age_groups, site_ids):
    best_threshold = 0.5
    best_score = -float("inf")
    best_metrics = None

    for threshold in np.arange(0.1, 0.91, 0.01):
        metrics, preds = evaluate_with_threshold(
            probs,
            labels,
            sexes,
            age_groups,
            site_ids,
            float(threshold),
        )
        score = (
            metrics["f1"]
            - 0.3 * metrics["sex_accuracy_gap"]
            - 0.2 * metrics["age_group_accuracy_gap"]
        )

        if score > best_score:
            best_score = score
            best_threshold = float(threshold)
            best_metrics = metrics

    return best_threshold, best_score, best_metrics


def make_dataset(split_indices):
    return BrainGraphTextDataset(
        [graphs[i] for i in split_indices],
        [adjs[i] for i in split_indices],
        text_embeddings[split_indices],
        [labels[i] for i in split_indices],
        [sexes[i] for i in split_indices],
        [age_groups[i] for i in split_indices],
        [site_ids[i] for i in split_indices],
    )


with open("processed/abide_graph_dataset.pkl", "rb") as f:
    data = pickle.load(f)

text_embeddings = torch.load("processed/text_embeddings.pt")

graphs = data["graphs"]
adjs = data["adjs"]
labels = data["labels"]
sexes = data["sexes"]
ages = data["ages"]
sites = data["sites"]

age_groups = [make_age_group(age) for age in ages]
site_to_id = {site: idx for idx, site in enumerate(sorted(set(sites)))}
site_ids = [site_to_id[site] for site in sites]

print("Total samples:", len(labels))
print("Text embeddings:", text_embeddings.shape)
print("Sites:", len(site_to_id))

all_indices = list(range(len(labels)))
train_val_idx, test_idx = stratified_split(
    all_indices,
    labels,
    test_size=0.2,
    seed=random_seed,
)
train_idx, val_idx = stratified_split(
    train_val_idx,
    labels,
    test_size=0.125,
    seed=random_seed,
)

print("Train samples:", len(train_idx))
print("Validation samples:", len(val_idx))
print("Test samples:", len(test_idx))

train_dataset = make_dataset(train_idx)
val_dataset = make_dataset(val_idx)
test_dataset = make_dataset(test_idx)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


model = GraphTextFusionModel(
    node_feat_dim=116,
    graph_hidden_dim=hidden_dim,
    text_dim=768,
    shared_dim=shared_dim,
    num_classes=2,
    dropout=dropout,
    classifier_hidden_dim=hidden_dim,
).to(device)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=learning_rate,
    weight_decay=weight_decay,
)


os.makedirs("results", exist_ok=True)

best_val_score = -float("inf")
best_epoch = 0
best_threshold = 0.5
best_state = None
epochs_without_improvement = 0

print("Starting tuned training...")

for epoch in range(1, epochs + 1):
    model.train()

    total_loss = 0.0
    total_cls_loss = 0.0
    total_align_loss = 0.0
    total_fair_loss = 0.0

    for x, adj, text_emb, y, sex, age_group, site_id in train_loader:
        x = x.to(device)
        adj = adj.to(device)
        text_emb = text_emb.to(device)
        y = y.to(device)
        sex = sex.to(device)
        age_group = age_group.to(device)

        optimizer.zero_grad()

        logits, graph_z, text_z = model(x, adj, text_emb)

        cls_loss = F.cross_entropy(logits, y)
        align_loss = contrastive_alignment_loss(graph_z, text_z)
        fair_loss = fairness_loss(logits, y, sex, age_group)
        loss = cls_loss + lambda_align * align_loss + lambda_fair * fair_loss

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_cls_loss += cls_loss.item()
        total_align_loss += align_loss.item()
        total_fair_loss += fair_loss.item()

    val_probs, val_labels, val_sexes, val_age_groups, val_site_ids = collect_probabilities(
        model,
        val_loader,
    )
    val_threshold, val_score, val_metrics = tune_threshold(
        val_probs,
        val_labels,
        val_sexes,
        val_age_groups,
        val_site_ids,
    )

    print(
        f"Epoch {epoch:03d}/{epochs} | "
        f"Loss: {total_loss:.4f} | "
        f"Cls: {total_cls_loss:.4f} | "
        f"Align: {total_align_loss:.4f} | "
        f"Fair: {total_fair_loss:.4f} | "
        f"Val F1: {val_metrics['f1']:.4f} | "
        f"Val Sex Gap: {val_metrics['sex_accuracy_gap']:.4f} | "
        f"Val Age Gap: {val_metrics['age_group_accuracy_gap']:.4f} | "
        f"Val Score: {val_score:.4f} | "
        f"Threshold: {val_threshold:.2f}"
    )

    if val_score > best_val_score:
        best_val_score = val_score
        best_epoch = epoch
        best_threshold = val_threshold
        best_state = copy.deepcopy(model.state_dict())
        epochs_without_improvement = 0

        torch.save(
            {
                "model_state_dict": best_state,
                "epoch": best_epoch,
                "threshold": best_threshold,
                "validation_score": best_val_score,
                "hyperparameters": {
                    "batch_size": batch_size,
                    "learning_rate": learning_rate,
                    "weight_decay": weight_decay,
                    "lambda_align": lambda_align,
                    "lambda_fair": lambda_fair,
                    "hidden_dim": hidden_dim,
                    "shared_dim": shared_dim,
                    "dropout": dropout,
                },
                "site_to_id": site_to_id,
            },
            checkpoint_path,
        )
    else:
        epochs_without_improvement += 1

    if epochs_without_improvement >= patience:
        print(f"Early stopping at epoch {epoch}. Best epoch: {best_epoch}")
        break


if best_state is None:
    raise RuntimeError("No model checkpoint was saved.")

model.load_state_dict(best_state)

test_probs, test_labels, test_sexes, test_age_groups, test_site_ids = collect_probabilities(
    model,
    test_loader,
)
test_metrics, test_preds = evaluate_with_threshold(
    test_probs,
    test_labels,
    test_sexes,
    test_age_groups,
    test_site_ids,
    best_threshold,
)

print("\n========== TUNED RESULTS ==========")
print("Model:", model_name)
print("Best Epoch:", best_epoch)
print("Best Validation Score:", best_val_score)
print("Selected Threshold:", best_threshold)
print("Accuracy:", test_metrics["accuracy"])
print("F1:", test_metrics["f1"])
print("Precision:", test_metrics["precision"])
print("Recall:", test_metrics["recall"])
print("Male Accuracy:", test_metrics["male_accuracy"])
print("Female Accuracy:", test_metrics["female_accuracy"])
print("Sex Accuracy Gap:", test_metrics["sex_accuracy_gap"])
print("Sex Demographic Parity Gap:", test_metrics["sex_demographic_parity_gap"])
print("Sex Equal Opportunity Gap:", test_metrics["sex_equal_opportunity_gap"])
print("Age Group Accuracy Gap:", test_metrics["age_group_accuracy_gap"])
print("Age Worst-Group Accuracy:", test_metrics["age_worst_group_accuracy"])
print("Site Accuracy Gap:", test_metrics["site_accuracy_gap"])
print("Site Worst-Group Accuracy:", test_metrics["site_worst_group_accuracy"])
print("Prediction Counts:", test_metrics["prediction_counts"])
print("Age group stats:", test_metrics["age_group_stats"])
print("Site stats with n >= 5:", test_metrics["site_stats"])
print("Saved checkpoint:", checkpoint_path)


new_result = pd.DataFrame({
    "Model": [model_name],
    "Accuracy": [test_metrics["accuracy"]],
    "F1": [test_metrics["f1"]],
    "Precision": [test_metrics["precision"]],
    "Recall": [test_metrics["recall"]],
    "Male Accuracy": [test_metrics["male_accuracy"]],
    "Female Accuracy": [test_metrics["female_accuracy"]],
    "Accuracy Gap": [test_metrics["sex_accuracy_gap"]],
    "Sex Accuracy Gap": [test_metrics["sex_accuracy_gap"]],
    "Sex Demographic Parity Gap": [test_metrics["sex_demographic_parity_gap"]],
    "Sex Equal Opportunity Gap": [test_metrics["sex_equal_opportunity_gap"]],
    "Age Group Accuracy Gap": [test_metrics["age_group_accuracy_gap"]],
    "Age Worst Group Accuracy": [test_metrics["age_worst_group_accuracy"]],
    "Site Accuracy Gap": [test_metrics["site_accuracy_gap"]],
    "Site Worst Group Accuracy": [test_metrics["site_worst_group_accuracy"]],
    "Threshold": [best_threshold],
    "Best Epoch": [best_epoch],
})

if os.path.exists(result_path):
    old_results = pd.read_csv(result_path)
    old_results = old_results[old_results["Model"] != model_name]
    final_results = pd.concat([old_results, new_result], ignore_index=True)
else:
    final_results = new_result

final_results.to_csv(result_path, index=False)

print("\nSaved results to:", result_path)
print("\nCurrent Experiment Table:")
print(final_results)
