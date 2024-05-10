import torch
import math

import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.cm import hsv
import matplotlib.pyplot as plt

# def get_umap_embeddings(loader, model, mapper, args, is_train=True):
#     embeddings = []
#     all_labels = []
#     for idx, batch in enumerate(loader):
#         batch = tuple(t.to(args.device) for t in batch)
#         inputs, labels = batch
#
#         hid = model(inputs, act=args.act_fn, return_layer='lin1')
#         embeddings.append(hid.detach().cpu())
#         all_labels.append(labels.detach().cpu())
#
#     embeddings = torch.cat(embeddings, dim=0).numpy()
#     all_labels = torch.cat(all_labels).numpy()
#     u_embeddings = mapper.fit_transform(embeddings)

def scatter_umap_embeddings(embeddings, labels, cmap, wandb, caption, wandb_key, global_step):
    plt.clf()
    plt.scatter(
        embeddings[:,0],
        embeddings[:,1],
        c=[cmap(x) for x in labels]
    )
    img = wandb.Image(
        plt,
        caption=caption
    )
    wandb.log({wandb_key: img}, step=global_step)

def generate_colormap(number_of_distinct_colors: int = 80):
    if number_of_distinct_colors == 0:
        number_of_distinct_colors = 80

    number_of_shades = 7
    number_of_distinct_colors_with_multiply_of_shades = int(math.ceil(number_of_distinct_colors / number_of_shades) * number_of_shades)

    # Create an array with uniformly drawn floats taken from <0, 1) partition
    linearly_distributed_nums = np.arange(number_of_distinct_colors_with_multiply_of_shades) / number_of_distinct_colors_with_multiply_of_shades

    # We are going to reorganise monotonically growing numbers in such way that there will be single array with saw-like pattern
    #     but each saw tooth is slightly higher than the one before
    # First divide linearly_distributed_nums into number_of_shades sub-arrays containing linearly distributed numbers
    arr_by_shade_rows = linearly_distributed_nums.reshape(number_of_shades, number_of_distinct_colors_with_multiply_of_shades // number_of_shades)

    # Transpose the above matrix (columns become rows) - as a result each row contains saw tooth with values slightly higher than row above
    arr_by_shade_columns = arr_by_shade_rows.T

    # Keep number of saw teeth for later
    number_of_partitions = arr_by_shade_columns.shape[0]

    # Flatten the above matrix - join each row into single array
    nums_distributed_like_rising_saw = arr_by_shade_columns.reshape(-1)

    # HSV colour map is cyclic (https://matplotlib.org/tutorials/colors/colormaps.html#cyclic), we'll use this property
    initial_cm = hsv(nums_distributed_like_rising_saw)

    lower_partitions_half = number_of_partitions // 2
    upper_partitions_half = number_of_partitions - lower_partitions_half

    # Modify lower half in such way that colours towards beginning of partition are darker
    # First colours are affected more, colours closer to the middle are affected less
    lower_half = lower_partitions_half * number_of_shades
    for i in range(3):
        initial_cm[0:lower_half, i] *= np.arange(0.2, 1, 0.8/lower_half)

    # Modify second half in such way that colours towards end of partition are less intense and brighter
    # Colours closer to the middle are affected less, colours closer to the end are affected more
    for i in range(3):
        for j in range(upper_partitions_half):
            modifier = np.ones(number_of_shades) - initial_cm[lower_half + j * number_of_shades: lower_half + (j + 1) * number_of_shades, i]
            modifier = j * modifier / upper_partitions_half
            initial_cm[lower_half + j * number_of_shades: lower_half + (j + 1) * number_of_shades, i] += modifier

    return ListedColormap(initial_cm)

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
