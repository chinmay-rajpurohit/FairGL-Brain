import torch
import torch.nn as nn
import torch.nn.functional as F


class GradientReversalFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambda_adv):
        ctx.lambda_adv = lambda_adv
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambda_adv * grad_output, None


def gradient_reverse(x, lambda_adv):
    return GradientReversalFunction.apply(x, lambda_adv)


class GraphTextFusionModel(nn.Module):
    def __init__(
        self,
        node_feat_dim=116,
        graph_hidden_dim=64,
        text_dim=768,
        shared_dim=64,
        num_classes=2,
        roi_prior_dim=0,
        roi_hidden_dim=32,
        dropout=0.3,
        classifier_hidden_dim=64
    ):
        super(GraphTextFusionModel, self).__init__()

        self.roi_proj = None
        graph_input_dim = node_feat_dim

        if roi_prior_dim > 0:
            self.roi_proj = nn.Linear(roi_prior_dim, roi_hidden_dim)
            graph_input_dim += roi_hidden_dim

        self.gcn_fc1 = nn.Linear(graph_input_dim, graph_hidden_dim)
        self.gcn_fc2 = nn.Linear(graph_hidden_dim, graph_hidden_dim)

        self.graph_proj = nn.Linear(graph_hidden_dim, shared_dim)
        self.text_proj = nn.Linear(text_dim, shared_dim)

        self.classifier = nn.Sequential(
            nn.Linear(shared_dim * 2, classifier_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(classifier_hidden_dim, num_classes)
        )

    def gcn_layer(self, x, adj, layer):
        x = torch.bmm(adj, x)
        x = layer(x)
        x = F.relu(x)
        return x

    def encode_graph(self, x, adj, roi_prior_embeddings=None):
        if self.roi_proj is not None:
            if roi_prior_embeddings is None:
                raise ValueError("roi_prior_embeddings must be provided when roi_prior_dim > 0.")

            roi_features = self.roi_proj(roi_prior_embeddings)
            roi_features = F.relu(roi_features)
            roi_features = roi_features.unsqueeze(0).expand(x.size(0), -1, -1)
            x = torch.cat([x, roi_features], dim=2)

        x = self.gcn_layer(x, adj, self.gcn_fc1)
        x = self.gcn_layer(x, adj, self.gcn_fc2)

        graph_embedding = x.mean(dim=1)

        graph_z = self.graph_proj(graph_embedding)
        graph_z = F.relu(graph_z)

        return graph_z

    def encode_text(self, text_embedding):
        text_z = self.text_proj(text_embedding)
        text_z = F.relu(text_z)

        return text_z

    def forward(self, x, adj, text_embedding, roi_prior_embeddings=None):
        graph_z = self.encode_graph(x, adj, roi_prior_embeddings)
        text_z = self.encode_text(text_embedding)

        fused = torch.cat([graph_z, text_z], dim=1)

        logits = self.classifier(fused)

        return logits, graph_z, text_z


class FairnessAwareGraphLanguageModel(nn.Module):
    def __init__(
        self,
        node_feat_dim=116,
        graph_hidden_dim=64,
        text_dim=768,
        shared_dim=64,
        num_classes=2,
        roi_prior_dim=0,
        roi_hidden_dim=32,
        num_sex_groups=3,
        num_age_groups=3,
        num_sites=20
    ):
        super(FairnessAwareGraphLanguageModel, self).__init__()

        self.roi_proj = None
        graph_input_dim = node_feat_dim

        if roi_prior_dim > 0:
            self.roi_proj = nn.Linear(roi_prior_dim, roi_hidden_dim)
            graph_input_dim += roi_hidden_dim

        self.gcn_fc1 = nn.Linear(graph_input_dim, graph_hidden_dim)
        self.gcn_fc2 = nn.Linear(graph_hidden_dim, graph_hidden_dim)

        self.graph_proj = nn.Sequential(
            nn.Linear(graph_hidden_dim, shared_dim),
            nn.ReLU(),
            nn.LayerNorm(shared_dim)
        )
        self.text_proj = nn.Sequential(
            nn.Linear(text_dim, shared_dim),
            nn.ReLU(),
            nn.LayerNorm(shared_dim)
        )

        self.shared_fusion = nn.Sequential(
            nn.Linear(shared_dim * 2, shared_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.LayerNorm(shared_dim)
        )

        self.classifier = nn.Sequential(
            nn.Linear(shared_dim * 3, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )
        self.graph_residual_classifier = nn.Linear(shared_dim, num_classes)

        self.sex_head = nn.Linear(shared_dim, num_sex_groups)
        self.age_head = nn.Linear(shared_dim, num_age_groups)
        self.site_head = nn.Linear(shared_dim, num_sites)

    def gcn_layer(self, x, adj, layer):
        x = torch.bmm(adj, x)
        x = layer(x)
        x = F.relu(x)
        return x

    def encode_graph(self, x, adj, roi_prior_embeddings=None):
        if self.roi_proj is not None:
            if roi_prior_embeddings is None:
                raise ValueError("roi_prior_embeddings must be provided when roi_prior_dim > 0.")

            roi_features = self.roi_proj(roi_prior_embeddings)
            roi_features = F.relu(roi_features)
            roi_features = roi_features.unsqueeze(0).expand(x.size(0), -1, -1)
            x = torch.cat([x, roi_features], dim=2)

        x = self.gcn_layer(x, adj, self.gcn_fc1)
        x = self.gcn_layer(x, adj, self.gcn_fc2)
        graph_embedding = x.mean(dim=1)

        return self.graph_proj(graph_embedding)

    def encode_text(self, text_embedding):
        return self.text_proj(text_embedding)

    def forward(self, x, adj, text_embedding, roi_prior_embeddings=None, adv_lambda=0.0):
        graph_z = self.encode_graph(x, adj, roi_prior_embeddings)
        text_z = self.encode_text(text_embedding)
        shared_z = self.shared_fusion(torch.cat([graph_z, text_z], dim=1))

        prediction_features = torch.cat([graph_z, text_z, shared_z], dim=1)
        logits = self.classifier(prediction_features)
        logits = logits + self.graph_residual_classifier(graph_z)
        adversarial_z = gradient_reverse(shared_z, adv_lambda)
        sensitive_logits = {
            "sex": self.sex_head(adversarial_z),
            "age_group": self.age_head(adversarial_z),
            "site": self.site_head(adversarial_z)
        }

        return {
            "logits": logits,
            "graph_z": graph_z,
            "text_z": text_z,
            "shared_z": shared_z,
            "sensitive_logits": sensitive_logits
        }
