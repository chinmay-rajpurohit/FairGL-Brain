import pickle
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
import pandas as pd

from models.graph_text_model import GraphTextFusionModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


class BrainGraphTextDataset(Dataset):
    def __init__(self, graphs, adjs, text_embeddings, labels, sexes):
        self.graphs = graphs
        self.adjs = adjs
        self.text_embeddings = text_embeddings
        self.labels = labels
        self.sexes = sexes

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return (
            self.graphs[idx],
            self.adjs[idx],
            self.text_embeddings[idx],
            torch.tensor(self.labels[idx], dtype=torch.long),
            torch.tensor(self.sexes[idx], dtype=torch.long)
        )


def contrastive_alignment_loss(graph_z, text_z, temperature=0.1):
    graph_z = F.normalize(graph_z, dim=1)
    text_z = F.normalize(text_z, dim=1)

    logits = torch.matmul(graph_z, text_z.T) / temperature
    labels = torch.arange(graph_z.size(0), device=graph_z.device)

    loss_graph_to_text = F.cross_entropy(logits, labels)
    loss_text_to_graph = F.cross_entropy(logits.T, labels)

    return (loss_graph_to_text + loss_text_to_graph) / 2


def compute_group_accuracy(labels, preds, sexes):
    male_labels, male_preds = [], []
    female_labels, female_preds = [], []

    for y, p, s in zip(labels, preds, sexes):
        if int(s) == 1:
            male_labels.append(y)
            male_preds.append(p)
        elif int(s) == 2:
            female_labels.append(y)
            female_preds.append(p)

    male_acc = accuracy_score(male_labels, male_preds) if len(male_labels) > 0 else 0.0
    female_acc = accuracy_score(female_labels, female_preds) if len(female_labels) > 0 else 0.0
    acc_gap = abs(male_acc - female_acc)

    return male_acc, female_acc, acc_gap, len(male_labels), len(female_labels)


with open("processed/abide_graph_dataset.pkl", "rb") as f:
    data = pickle.load(f)

text_embeddings = torch.load("processed/text_embeddings.pt")

graphs = data["graphs"]
adjs = data["adjs"]
labels = data["labels"]
sexes = data["sexes"]

print("Total samples:", len(labels))
print("Text embeddings:", text_embeddings.shape)

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
    [sexes[i] for i in train_idx]
)

test_dataset = BrainGraphTextDataset(
    [graphs[i] for i in test_idx],
    [adjs[i] for i in test_idx],
    text_embeddings[test_idx],
    [labels[i] for i in test_idx],
    [sexes[i] for i in test_idx]
)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

print("Train samples:", len(train_dataset))
print("Test samples:", len(test_dataset))
print("Train batches:", len(train_loader))
print("Test batches:", len(test_loader))

model = GraphTextFusionModel(
    node_feat_dim=116,
    graph_hidden_dim=64,
    text_dim=768,
    shared_dim=64,
    num_classes=2
).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
criterion = nn.CrossEntropyLoss()

epochs = 30
lambda_align = 0.1

print("Starting training...")

for epoch in range(epochs):
    model.train()
    total_loss = 0.0
    total_cls_loss = 0.0
    total_align_loss = 0.0

    for x, adj, text_emb, y, sex in train_loader:
        x = x.to(device)
        adj = adj.to(device)
        text_emb = text_emb.to(device)
        y = y.to(device)

        optimizer.zero_grad()

        logits, graph_z, text_z = model(x, adj, text_emb)

        cls_loss = criterion(logits, y)
        align_loss = contrastive_alignment_loss(graph_z, text_z)
        loss = cls_loss + lambda_align * align_loss

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_cls_loss += cls_loss.item()
        total_align_loss += align_loss.item()

    print(
        f"Epoch {epoch + 1}/{epochs} | "
        f"Loss: {total_loss:.4f} | "
        f"Cls: {total_cls_loss:.4f} | "
        f"Align: {total_align_loss:.4f}"
    )

print("Training finished.")

model.eval()

all_preds = []
all_labels = []
all_sexes = []

with torch.no_grad():
    for x, adj, text_emb, y, sex in test_loader:
        x = x.to(device)
        adj = adj.to(device)
        text_emb = text_emb.to(device)

        logits, graph_z, text_z = model(x, adj, text_emb)
        preds = torch.argmax(logits, dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(y.numpy())
        all_sexes.extend(sex.numpy())

overall_acc = accuracy_score(all_labels, all_preds)
overall_f1 = f1_score(all_labels, all_preds)

male_acc, female_acc, acc_gap, male_n, female_n = compute_group_accuracy(
    all_labels,
    all_preds,
    all_sexes
)

print("\n========== RESULTS ==========")
print("Model: Graph + Text + Alignment")
print("Overall Accuracy:", overall_acc)
print("Overall F1:", overall_f1)
print("Male Accuracy:", male_acc)
print("Female Accuracy:", female_acc)
print("Accuracy Gap:", acc_gap)
print("Male test samples:", male_n)
print("Female test samples:", female_n)

os.makedirs("results", exist_ok=True)

new_result = pd.DataFrame({
    "Model": ["Graph + Text + Alignment"],
    "Accuracy": [overall_acc],
    "F1": [overall_f1],
    "Male Accuracy": [male_acc],
    "Female Accuracy": [female_acc],
    "Accuracy Gap": [acc_gap],
    "Male Test Samples": [male_n],
    "Female Test Samples": [female_n]
})

result_path = "results/experiment_results.csv"

if os.path.exists(result_path):
    old_results = pd.read_csv(result_path)
    old_results = old_results[old_results["Model"] != "Graph + Text + Alignment"]
    final_results = pd.concat([old_results, new_result], ignore_index=True)
else:
    final_results = new_result

final_results.to_csv(result_path, index=False)

print("\nSaved results to:", result_path)
print("\nCurrent Experiment Table:")
print(final_results)