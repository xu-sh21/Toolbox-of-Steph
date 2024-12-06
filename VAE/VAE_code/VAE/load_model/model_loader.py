'''Load the model according to your definition!!!'''
#######################################################################
import os
import torch
import torch.nn as nn
import torch.optim as optim

from vae_core.network import myVAE
from others.utils import loss_vae
#######################################################################

# Define the model, loss function and optimizer.
def load_model(config):
    # Device.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    # Model.
    # Read model parameters from config_dict.
    lr = config['params']['lr']
    channels = config['params']['channels']
    encoder_kernel_size = config['params']['encoder_kernel_size']
    decoder_kernel_size = config['params']['decoder_kernel_size']
    latent_dim = config['params']['latent_dim']
    act_func = config['params']['act_func']
    # Choose activation function.
    if act_func.lower() == 'relu':
        activation_fn = nn.ReLU
    elif act_func.lower() == 'tanh':
        activation_fn = nn.Tanh
    elif act_func.lower() == 'sigmoid':
        activation_fn = nn.Sigmoid
    else:
        raise ValueError('Invalid activation function.')
    # Settings.
    # Choose model parameters according to dataset.
    dataset = config['settings']['dataset']
    if dataset == 'mnist':
        fig_channels = 1
        input_dim = [28,28]
    elif dataset == 'cifar10':
        fig_channels = 3
        input_dim = [32,32]
    else:
        raise ValueError('Unknown Dataset!!! Supported Datasets Are: "mnist", "cifar10"!!!')
    
    if channels[0] != fig_channels:
        raise ValueError('Invalid number of channels for the given dataset.')


    # Define the model.
    model = myVAE(input_dim,
                  channels,
                  encoder_kernel_size,
                  latent_dim, 
                  decoder_kernel_size,
                  activation_fn)
    model.to(device)
    loss_func = loss_vae
    optimizer = optim.Adam(model.parameters(), lr=lr)

    return device, model, loss_func, optimizer


# Load the model if exists.
def load_old_model(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_path = config['settings']['train_dir'] + '/checkpoint_' + str(config['settings']['inference_version']) + '.pth.tar'
    if os.path.exists(model_path):
        model = torch.load(model_path)
        loss_func = loss_vae

        return device, model, loss_func
    else:
        raise FileNotFoundError('Model Not Found!!! Please Check!!!')