import torch
from torch import nn
from torch.nn import Linear, ReLU, Sigmoid, Sequential

class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, z_dim):
        '''
        Args:
            input_dim: A integer indicating the size of input (in case of MNIST 28 * 28).
            hidden_dim: A integer indicating the size of hidden dimension.
            z_dim: A integer indicating the latent dimension.
        '''
        super().__init__()
        self.fc =  Sequential(Linear(input_dim, hidden_dim), ReLU())
        self.mu =  Linear(hidden_dim, z_dim)
        self.var = Linear(hidden_dim, z_dim)

    def forward(self, x):               # x shape [batch_size, input_dim]
        hidden = self.fc(x)             # hidden shape [batch_size, hidden_dim]
        z_mu = self.mu(hidden)          # z_mu   shape [batch_size, latent_dim]
        z_var = self.var(hidden)        # z_var  shape [batch_size, latent_dim]
        return z_mu, z_var

class Decoder(nn.Module):
    def __init__(self, z_dim, hidden_dim, output_dim, output_bounded=True):
        '''
        Args:
            z_dim: A integer indicating the latent size.
            hidden_dim: A integer indicating the size of hidden dimension.
            output_dim: A integer indicating the output dimension (in case of MNIST 28 * 28)
        '''
        super().__init__()
        self.linear = Sequential(Linear(z_dim, hidden_dim), ReLU())
        if output_bounded:
            self.out = Sequential(Linear(hidden_dim, output_dim), Sigmoid())
        else:
            self.out = Linear(hidden_dim, output_dim)

    def forward(self, x):            # x shape [batch_size, latent_dim]
        hidden = self.linear(x)      # hidden shape [batch_size, hidden_dim]
        predicted = self.out(hidden) # predicted shape [batch_size, output_dim]
        return predicted
    
class VAE(nn.Module):
    def __init__(self, enc, dec):
        super().__init__()
        self.enc = enc
        self.dec = dec

    def forward(self, x):
        z_mu, z_var = self.enc(x) # encode

        # sample from the distribution having latent parameters z_mu, z_var and reparameterize
        std = torch.exp(z_var / 2)
        eps = torch.randn_like(std)
        x_sample = eps.mul(std).add_(z_mu)

        predicted = self.dec(x_sample) # decode
        return predicted, z_mu, z_var