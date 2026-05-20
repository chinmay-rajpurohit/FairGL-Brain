import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleGCN(nn.Module):
    def __init__(self, input_dim=116, hidden_dim=64, num_classes=2):
        super(SimpleGCN, self).__init__()

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def gcn_layer(self, x, adj, layer):
        x = torch.bmm(adj, x)
        x = layer(x)
        x = F.relu(x)
        return x

    def forward(self, x, adj):

        x = self.gcn_layer(x, adj, self.fc1)
        x = self.gcn_layer(x, adj, self.fc2)

        x = x.mean(dim=1)

        logits = self.classifier(x)

        return logits, x
