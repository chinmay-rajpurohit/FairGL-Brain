import pickle
import torch
import torch.nn.functional as F


DATASET_FILE = "processed/abide_graph_dataset.pkl"
TEXT_EMBEDDING_FILE = "processed/text_embeddings.pt"
OUTPUT_FILE = "processed/population_similarity_graph.pt"
K_NEIGHBORS = 50


def upper_triangle_vector(matrix):
    mask = torch.triu(torch.ones_like(matrix, dtype=torch.bool), diagonal=1)
    return matrix[mask]


with open(DATASET_FILE, "rb") as f:
    data = pickle.load(f)

text_embeddings = torch.load(TEXT_EMBEDDING_FILE, map_location="cpu")
graphs = data["graphs"]
labels = data["labels"]

graph_features = torch.stack([
    upper_triangle_vector(graph.float())
    for graph in graphs
])

graph_features = F.normalize(graph_features, dim=1)
text_features = F.normalize(text_embeddings.float(), dim=1)

population_features = torch.cat([graph_features, text_features], dim=1)
population_features = F.normalize(population_features, dim=1)

similarity = torch.matmul(population_features, population_features.T)
similarity.fill_diagonal_(-1.0)

k = min(K_NEIGHBORS, similarity.size(0) - 1)
top_values, top_indices = torch.topk(similarity, k=k, dim=1)

population_adj = torch.zeros_like(similarity)
row_indices = torch.arange(similarity.size(0)).unsqueeze(1).expand(-1, k)
population_adj[row_indices, top_indices] = top_values.clamp(min=0.0)

population_adj = torch.maximum(population_adj, population_adj.T)
population_adj.fill_diagonal_(0.0)

edge_index = population_adj.nonzero(as_tuple=False).T
edge_weight = population_adj[edge_index[0], edge_index[1]]

output = {
    "adjacency": population_adj,
    "edge_index": edge_index,
    "edge_weight": edge_weight,
    "k_neighbors": k,
    "feature_sources": ["connectivity_upper_triangle", "subject_text_embedding"],
    "num_subjects": len(labels)
}

torch.save(output, OUTPUT_FILE)

print("Population graph saved to:", OUTPUT_FILE)
print("Subjects:", len(labels))
print("K neighbors:", k)
print("Undirected weighted edges:", edge_weight.numel())
print("Average degree:", edge_weight.numel() / len(labels))
print("Mean edge weight:", edge_weight.mean().item())
