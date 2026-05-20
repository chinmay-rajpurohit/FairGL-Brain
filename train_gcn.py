import pickle
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

from models.gcn_model import SimpleGCN


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


class BrainDataset(Dataset):
    def __init__(self, graphs, adjs, labels):
        self.graphs = graphs
        self.adjs = adjs
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return (
            self.graphs[idx],
            self.adjs[idx],
            self.labels[idx]
        )


with open("processed/abide_graph_dataset.pkl", "rb") as f:
    data = pickle.load(f)

graphs = data["graphs"]
adjs = data["adjs"]
labels = data["labels"]

print("Total samples:", len(labels))


indices = list(range(len(labels)))

train_idx, test_idx = train_test_split(
    indices,
    test_size=0.2,
    stratify=labels,
    random_state=42
)

train_graphs = [graphs[i] for i in train_idx]
train_adjs = [adjs[i] for i in train_idx]
train_labels = [labels[i] for i in train_idx]

test_graphs = [graphs[i] for i in test_idx]
test_adjs = [adjs[i] for i in test_idx]
test_labels = [labels[i] for i in test_idx]


train_dataset = BrainDataset(
    train_graphs,
    train_adjs,
    train_labels
)

test_dataset = BrainDataset(
    test_graphs,
    test_adjs,
    test_labels
)

train_loader = DataLoader(
    train_dataset,
    batch_size=8,
    shuffle=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=8,
    shuffle=False
)


model = SimpleGCN().to(device)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=1e-3
)

criterion = nn.CrossEntropyLoss()


epochs = 20

for epoch in range(epochs):

    model.train()

    total_loss = 0

    for x, adj, y in train_loader:

        x = x.to(device)
        adj = adj.to(device)
        y = torch.tensor(y).long().to(device)

        optimizer.zero_grad()

        logits, embeddings = model(x, adj)

        loss = criterion(logits, y)

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1} Loss: {total_loss:.4f}")


model.eval()

all_preds = []
all_labels = []

with torch.no_grad():

    for x, adj, y in test_loader:

        x = x.to(device)
        adj = adj.to(device)

        logits, embeddings = model(x, adj)

        preds = torch.argmax(logits, dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(y)

acc = accuracy_score(all_labels, all_preds)
f1 = f1_score(all_labels, all_preds)

print("\nTest Accuracy:", acc)
print("Test F1:", f1)
