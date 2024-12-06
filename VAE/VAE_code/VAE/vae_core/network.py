'''Define the model of VAE.'''
################################################
import torch
import torch.nn as nn
import torch.nn.functional as nn_functional
################################################

class myVAE(nn.Module):
    def __init__(self,
                 input_dim=[28, 28],
                 channels=[1, 32],
                 encoder_kernel_size=3,
                 latent_dim=10,
                 decoder_kernel_size=3,
                 activation_fn=nn.ReLU):
        # Parameter instructions:
        # input_dim: The dimension of input images, e.g. [28, 28] for MNIST, and [32, 32, 3] for CIFAR10.
        # channels: The list of channels of each layer for encoder, reversed list for decoder.
        # encoder_kernel_size: The size of convolution kernel in encoder.
        # latent_dim: The dimension of latent variable.
        # decoder_kernel_size: The size of convolution kernel in decoder.
        # activation_fn: The activation function for neural network.
        super().__init__()

        # Channels for encoder and decoder layers.
        self.encoder_channels = channels
        self.decoder_channels = channels[::-1]
        self.channel_num = len(self.encoder_channels)

        # Encoder part.
        encoder_layers = []
        in_channels = self.encoder_channels[0]
        for i in range(1, self.channel_num):
            encoder_layers.extend([
                nn.Conv2d(in_channels=in_channels, out_channels=self.encoder_channels[i], 
                          kernel_size=encoder_kernel_size, padding=1),
                activation_fn(),
                nn.MaxPool2d(kernel_size=2, stride=2)
            ])
            in_channels = self.encoder_channels[i]
        encoder_layers.append(nn.Flatten())
        self.encoder_part = nn.Sequential(*encoder_layers)


        # Linear part for getting mu and sigma.
        H_linear = int(input_dim[0] / (2 ** (self.channel_num - 1)))
        W_linear = int(input_dim[1] / (2 ** (self.channel_num - 1)))
        self.get_mu = nn.Linear(in_features=self.encoder_channels[-1] * H_linear * W_linear, out_features=latent_dim)
        self.get_sigma = nn.Linear(in_features=self.encoder_channels[-1] * H_linear * W_linear, out_features=latent_dim)


        # Decoder part.
        decoder_layers = [
            nn.Linear(in_features=latent_dim, out_features=self.decoder_channels[0] * H_linear * W_linear),
            activation_fn(),
            nn.Unflatten(dim=1, unflattened_size=(self.decoder_channels[0], H_linear, W_linear))
        ]
        in_channels = self.decoder_channels[0]
        for i in range(1, self.channel_num):
            decoder_layers.extend([
                nn.ConvTranspose2d(in_channels=in_channels, out_channels=self.decoder_channels[i], 
                                   kernel_size=decoder_kernel_size, stride=2, padding=1, output_padding=1),
                activation_fn() if i < len(self.decoder_channels)-1 else nn.Sigmoid() # Sigmoid for the last layer to get values in [0, 1]
            ])
            in_channels = self.decoder_channels[i]
        self.decoder_part = nn.Sequential(*decoder_layers)


    def encoder(self, x):
        x = self.encoder_part(x)
        x = nn_functional.relu(x)

        # Get 10 mu and sigma for normal distributions.
        mu = self.get_mu(x)
        sigma = self.get_sigma(x)

        return mu, sigma


    def reparam(self, mus, sigmas):
        eps = torch.randn(size=(mus.shape))
        std = torch.sqrt(torch.exp(sigmas))

        z = mus + eps * std

        return z


    def decoder(self, x):
        x = self.decoder_part(x)

        return x
    

    def forward(self, x):
        mu, sigma = self.encoder(x)
        z = self.reparam(mu, sigma)
        x_hat = self.decoder(z)

        return x_hat, mu, sigma
