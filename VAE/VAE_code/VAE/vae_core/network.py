'''Define the model of VAE.'''
################################################
import torch
import torch.nn as nn
################################################

class myVAE(nn.Module):
    def __init__(self,
                 encoder_channels=(32, 64),
                 encoder_kernel_size=3,
                 latent_dim=10,
                 decoder_channels=(32, 3),
                 decoder_kernel_size=3,
                 activation_fn=nn.ReLU,
                 dataset=None):
        super().__init__()

        # Encoder part.
        encoder_layers = []
        if dataset == 'mnist':
            in_channels = 1
        elif dataset == 'cifar10':
            in_channels = 3
        else:
            raise ValueError('Unknown Dataset!!! Supported Datasets Are: "mnist", "cifar10"!!!')

        for out_channels in encoder_channels:
            encoder_layers.extend([
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels, 
                          kernel_size=encoder_kernel_size, padding=1),
                activation_fn(),
                nn.MaxPool2d(kernel_size=2, stride=2)
            ])
            in_channels = out_channels
        encoder_layers.append(nn.Flatten())
        self.encoder_part = nn.Sequential(*encoder_layers)

        # Linear part for getting mu and sigma.
        self.get_mu = nn.Linear(in_features=encoder_channels[-1]*7*7, out_features=latent_dim)
        self.get_sigma = nn.Linear(in_features=encoder_channels[-1]*7*7, out_features=latent_dim)

        # Decoder part.
        # self.decoder_part = nn.Sequential(
        #     nn.Linear(in_features=10, out_features=512),
        #     nn.ReLU(),
        #     nn.Linear(in_features=512, out_features=64*7*7),
        #     nn.Unflatten(dim=1, shape=(64, 7, 7)),
        #     nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=3, stride=2, padding=1, output_padding=1), # (b, 64, 14, 14)
        #     nn.ReLU(),
        #     nn.ConvTranspose2d(in_channels=32, out_channels=3, kernel_size=3, stride=2, padding=1, output_padding=1) # (b, 3, 28, 28)
        # )
        decoder_layers = [
            nn.Linear(in_features=latent_dim, out_features=encoder_channels[-1]*7*7),
            activation_fn(),
            nn.Unflatten(dim=1, unflattened_size=(encoder_channels[-1], 7, 7))
        ]
        for i, out_channels in enumerate(decoder_channels):
            decoder_layers.extend([
                nn.ConvTranspose2d(in_channels=encoder_channels[-i-1], out_channels=out_channels, 
                                   kernel_size=decoder_kernel_size, stride=2, padding=1, output_padding=1),
                activation_fn() if i < len(decoder_channels)-1 else nn.Sigmoid()  # Sigmoid for the last layer to get values in [0, 1]
            ])
        self.decoder_part = nn.Sequential(*decoder_layers)


    def encoder(self, x):
        x = self.encoder_part(x)
        x = nn.functional.relu(x)

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
