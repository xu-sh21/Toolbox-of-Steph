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
    encoder_channels = config['params']['encoder_channels']
    decoder_channels = config['params']['decoder_channels']
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

    # Check channels of encoders and decoders.
    encoder_channels_len = len(encoder_channels)
    decoder_channels_len = len(decoder_channels)
    if encoder_channels_len != decoder_channels_len:
        raise ValueError('Encoder and Decoder channels should be the same!!!')
    channels_len = encoder_channels_len


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

    # Define the model.
    model = myVAE(fig_channels,
                  input_dim,
                  encoder_channels,
                  encoder_kernel_size,
                  latent_dim, 
                  decoder_channels,
                  decoder_kernel_size,
                  channels_len,
                  activation_fn)
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