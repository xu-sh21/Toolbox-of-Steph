import torch
import torch.nn as nn
import torch.nn.functional as F

class myVAE(nn.Module):
    def __init__(self, 
                 encoder_channels=(32, 64), 
                 encoder_kernel_size=3, 
                 latent_dim=10, 
                 decoder_channels=(32, 3), 
                 decoder_kernel_size=3,
                 activation_fn=nn.ReLU):
        super().__init__()
        
        # Encoder part.
        encoder_layers = []
        in_channels = 3
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
        decoder_layers = [
            nn.Linear(in_features=latent_dim, out_features=encoder_channels[-1]*7*7),
            activation_fn(),
            nn.Unflatten(dim=1, shape=(encoder_channels[-1], 7, 7))
        ]
        for i, out_channels in enumerate(decoder_channels):
            decoder_layers.extend([
                nn.ConvTranspose2d(in_channels=encoder_channels[-i-1], out_channels=out_channels, 
                                   kernel_size=decoder_kernel_size, stride=2, padding=1, output_padding=1),
                activation_fn() if i < len(decoder_channels)-1 else nn.Sigmoid()  # Sigmoid for the last layer to get values in [0, 1]
            ])
        self.decoder_part = nn.Sequential(*decoder_layers)

    # Forward method...
    # Rest of the class...
