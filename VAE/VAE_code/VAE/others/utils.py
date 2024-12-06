'''Define some useful functions !!!'''
################################################
import numpy as np
import torch
import torch.nn as nn
################################################

# Shuffle the first shuffle_size columns of a 2D numpy array X
def shuffle(X, shuffle_parts=1):
    chunk_size = int(len(X) / shuffle_parts)
    shuffled_range = list(range(chunk_size))

    X_buffer = np.copy(X[0:chunk_size])
    for k in range(shuffle_parts):
        np.random.shuffle(shuffled_range)
    for i in range(chunk_size):
        X_buffer[i] = X[k * chunk_size + shuffled_range[i]]
    X[k * chunk_size:(k + 1) * chunk_size] = X_buffer

    return X


# Loss function for VAE.
def loss_vae(X_hat, X_input, mu, sigma):
    loss_fn = nn.MSELoss()
    loss = loss_fn(X_hat, X_input)
    kl_divergence = -0.5 * torch.sum(1 + sigma - mu.pow(2) - sigma.exp())

    loss_tot = loss + kl_divergence

    return loss_tot


# Randomly choose test data from test dataset with a specified quantity.
def randomly_choose_from_test(dataset, n):
    dataset_size = dataset.shape[0]
    
    if n > dataset_size:
        raise ValueError("Number n Should Not Be Bigger Than Total Number of Dataset!!! Please Check!!!")
    
    random_indices = np.random.choice(dataset.shape[0], n, replace=False)
    selected_samples = dataset[random_indices]
    selected_samples = torch.from_numpy(selected_samples)

    return selected_samples