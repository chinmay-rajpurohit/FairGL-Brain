from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


def binary_classification_metrics(labels, preds):
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds, zero_division=0),
        "precision": precision_score(labels, preds, zero_division=0),
        "recall": recall_score(labels, preds, zero_division=0)
    }


def group_metrics(labels, preds, group_ids):
    groups = sorted(set(int(group_id) for group_id in group_ids))
    per_group = {}

    for group_id in groups:
        group_labels = [
            label
            for label, candidate_group in zip(labels, group_ids)
            if int(candidate_group) == group_id
        ]
        group_preds = [
            pred
            for pred, candidate_group in zip(preds, group_ids)
            if int(candidate_group) == group_id
        ]

        positives = sum(1 for pred in group_preds if int(pred) == 1)
        actual_positives = sum(1 for label in group_labels if int(label) == 1)

        per_group[group_id] = {
            "accuracy": accuracy_score(group_labels, group_preds) if group_labels else 0.0,
            "f1": f1_score(group_labels, group_preds, zero_division=0) if group_labels else 0.0,
            "positive_rate": positives / len(group_preds) if group_preds else 0.0,
            "true_positive_rate": recall_score(
                group_labels,
                group_preds,
                zero_division=0
            ) if actual_positives > 0 else 0.0,
            "n": len(group_labels)
        }

    accuracies = [stats["accuracy"] for stats in per_group.values()]
    positive_rates = [stats["positive_rate"] for stats in per_group.values()]
    true_positive_rates = [stats["true_positive_rate"] for stats in per_group.values()]

    return {
        "per_group": per_group,
        "accuracy_gap": max(accuracies) - min(accuracies) if len(accuracies) >= 2 else 0.0,
        "demographic_parity_gap": (
            max(positive_rates) - min(positive_rates)
            if len(positive_rates) >= 2
            else 0.0
        ),
        "equal_opportunity_gap": (
            max(true_positive_rates) - min(true_positive_rates)
            if len(true_positive_rates) >= 2
            else 0.0
        ),
        "worst_group_accuracy": min(accuracies) if accuracies else 0.0
    }
