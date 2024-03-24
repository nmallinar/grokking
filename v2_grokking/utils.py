import torch

def compute_correlation(labels, predictions):
    """
    Compute the correlation between label vector and prediction vector.

    Args:
    - labels (torch.Tensor): 1D tensor containing the true labels.
    - predictions (torch.Tensor): 1D tensor containing the predicted values.

    Returns:
    - correlation (torch.Tensor): Scalar tensor containing the correlation.
    """

    # Calculate mean of labels and predictions
    mean_labels = torch.mean(labels)
    mean_predictions = torch.mean(predictions)

    # Compute covariance and variances
    covariance = torch.mean((labels - mean_labels) * (predictions - mean_predictions))
    variance_labels = torch.mean((labels - mean_labels)**2)
    variance_predictions = torch.mean((predictions - mean_predictions)**2)

    # Compute correlation
    correlation = covariance / (torch.sqrt(variance_labels) * torch.sqrt(variance_predictions))
    correlations[torch.isnan(correlations)] = 0.0

    return correlation

def compute_correlation_one_hot(labels_one_hot, predictions):
    """
    Compute the correlation between one-hot encoded label matrix and prediction matrix.

    Args:
    - labels_one_hot (torch.Tensor): 2D tensor containing the one-hot encoded labels.
                                      Shape: [batch_size, num_classes]
    - predictions (torch.Tensor): 2D tensor containing the predicted values.
                                  Shape: [batch_size, num_classes]

    Returns:
    - correlations (torch.Tensor): 1D tensor containing the correlations for each column of predictions.
    """

    # Calculate mean of predictions
    mean_predictions = torch.mean(predictions, dim=0)

    # Compute covariance and variances
    covariance_matrix = torch.zeros(labels_one_hot.shape[1], labels_one_hot.shape[1])
    variance_labels = torch.mean(labels_one_hot - torch.mean(labels_one_hot, dim=0, keepdim=True), dim=0)
    variance_predictions = torch.mean(predictions - mean_predictions, dim=0)

    for i in range(labels_one_hot.shape[1]):
        for j in range(labels_one_hot.shape[1]):
            covariance_matrix[i, j] = torch.mean(
                (labels_one_hot[:, i] - torch.mean(labels_one_hot[:, i])) *
                (predictions[:, j] - mean_predictions[j])
            )

    # Compute correlations
    correlations = covariance_matrix / (torch.sqrt(variance_labels.unsqueeze(1)) * torch.sqrt(variance_predictions.unsqueeze(0)))
    correlations[torch.isnan(correlations)] = 0.0
    correlations[correlations == float("Inf")] = 0.0
    return correlations
