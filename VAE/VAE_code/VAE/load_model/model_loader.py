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
    lr = config['params']['lr']
    encoder_channels = config['params']['encoder_channels']
    decoder_channels = config['params']['decoder_channels']
    encoder_kernel_size = config['params']['encoder_kernel_size']
    decoder_kernel_size = config['params']['decoder_kernel_size']
    latent_dim = config['params']['latent_dim']
    act_func = config['params']['act_func']

    if act_func == 'relu':
        activation_fn = nn.ReLU
    elif act_func == 'tanh':
        activation_fn = nn.Tanh
    elif act_func == 'sigmoid':
        activation_fn = nn.Sigmoid
    else:
        raise ValueError('Invalid activation function.')
    
    # Settings.
    dataset = config['settings']['dataset']

    model = myVAE(encoder_channels,
                 encoder_kernel_size,
                 latent_dim,
                 decoder_channels,
                 decoder_kernel_size,
                 activation_fn,
                 dataset)
    model.to(device)
    loss_func = loss_vae
    optimizer = optim.Adam(model.parameters(), lr=lr)

    return device, model, loss_func, optimizer


# Load the model if exists.
def load_old_model(config):
    model_path = config['settings']['train_dir'] + 'checkpoint_' + str(config['settings']['inference_version']) + '.pth.tar'
    if os.path.exists(model_path):
        model = torch.load(model_path)

        return model
    else:
        raise FileNotFoundError('Model Not Found!!! Please Check!!!')