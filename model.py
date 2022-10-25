# -*- coding: utf-8 -*-
"""Model definition for Benzaiten Starter Kit ver. 1.0

Copyright (C) 2022 by Akira TAMAMORI

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
import torch
from torch import nn


class Encoder(nn.Module):
    """RNN(LSTM)-based Encoder class."""

    def __init__(self, input_dim, emb_dim, hidden_dim, n_layers=1):
        """Initialize class.

        input_dim: #dimensions of input mixed-hot vectors (melody + chord)
        emb_dim: Embedded dimension for LSTM cell and hidden
        hidden_dim: number of dimensions of LSTM output vectors
        n_layers: number of hidden layers
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.embedding = nn.Linear(input_dim, emb_dim)
        self.rnn = nn.LSTM(emb_dim, hidden_dim, n_layers, batch_first=True)

    def forward(self, source):
        """Forward propagation."""
        embedded = self.embedding(source)
        outputs, (hidden, cell) = self.rnn(embedded)
        return outputs, (hidden, cell)


class Decoder(nn.Module):
    """RNN(LSTM)-based Decoder class."""

    def __init__(self, output_dim, hidden_dim, n_layers=1):
        """Initialize class.

        input_dim: #dimensions of one-hot vectors (melody)
        hidden_dim: number of dimensions of LSTM output vectors
        n_layers: number of hidden layers
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.output_dim = output_dim
        self.rnn = nn.LSTM(
            hidden_dim, hidden_dim, num_layers=n_layers, batch_first=True
        )
        self.fc_out = nn.Linear(hidden_dim, output_dim)

    def forward(self, inputs):
        """Forward propagation."""
        output, (hidden, cell) = self.rnn(inputs)
        prediction = self.fc_out(output)
        return prediction, (hidden, cell)


class VariationalAutoEncoder(nn.Module):
    """VariationalAutoEncoder Class."""

    def __init__(self, input_dim, hidden_dim, latent_dim, n_hidden=0):
        """Initialize class.

        input_dim: #dimensions of observations
        hidden_dim: #dimensions of hidden units
        latent_dim: #dimensions of latent vectors
        n_hidden: number of hidden-to-hidden layers
        """
        super().__init__()
        self.n_hidden = n_hidden  # number of hidden-to-hidden layers

        layers = nn.ModuleList([])
        layers += [nn.Linear(input_dim, hidden_dim)]
        layers += [nn.Linear(hidden_dim, hidden_dim) for _ in range(self.n_hidden)]
        layers += [nn.Linear(hidden_dim, latent_dim), nn.Linear(hidden_dim, latent_dim)]
        self.enc_layers = layers

        layers = nn.ModuleList([])
        layers += [nn.Linear(latent_dim, hidden_dim)]
        layers += [nn.Linear(hidden_dim, hidden_dim) for _ in range(self.n_hidden)]
        layers += [nn.Linear(hidden_dim, input_dim)]
        self.dec_layers = layers

        self.activation = nn.ReLU()

    def encode(self, inputs):
        """Drive encoder."""
        hidden = self.activation(self.enc_layers[0](inputs))
        for i in range(self.n_hidden):
            hidden = self.activation(self.enc_layers[i + 1](hidden))
        mean = self.enc_layers[-2](hidden)
        logvar = self.enc_layers[-1](hidden)
        return mean, logvar

    def reparameterization(self, mean, logvar):
        """Sample latent vector from inputs via reparameterization trick."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        latent = mean + eps * std  # reparameterization trick
        return latent

    def decode(self, latent):
        """Drive decoder."""
        hidden = self.activation(self.dec_layers[0](latent))
        for i in range(self.n_hidden):
            hidden = self.activation(self.dec_layers[i + 1](hidden))
        reconst = self.dec_layers[-1](hidden)
        return reconst

    def forward(self, inputs):
        """Reconstruct inputs through VAE."""
        mean, logvar = self.encode(inputs)
        latent = self.reparameterization(mean, logvar)
        reconst = self.decode(latent)
        return reconst, mean, logvar


class Seq2SeqMelodyComposer(nn.Module):
    """Seq2SeqMelodyComposer class for ad-lib melody composition."""

    def __init__(self, config, device):
        """Initialize class."""
        super().__init__()
        self.encoder = Encoder(
            config.model.encoder.input_dim,
            config.model.encoder.emb_dim,
            config.model.encoder.hidden_dim,
            config.model.encoder.n_layers,
        ).to(device)
        self.decoder = Decoder(
            config.model.decoder.output_dim,
            config.model.decoder.hidden_dim,
            config.model.decoder.n_layers,
        ).to(device)
        self.vae = VariationalAutoEncoder(
            config.model.encoder.hidden_dim,
            config.model.vae.hidden_dim,
            config.model.vae.latent_dim,
            config.model.vae.n_hidden,
        ).to(device)

    def forward(self, inputs):
        """Forward propagation.

        Args:
            inputs (Tensor) : sequence of mixed-hot vectors (melody + chord)

        Returns:
            outputs (Tensor) : sequence of one-hot vectors (melody)
        """
        seq_len = inputs.shape[1]
        _, encoder_state = self.encoder(inputs)
        hiddens = torch.squeeze(encoder_state[0])
        hiddens = hiddens.unsqueeze(0)
        reconst_state, _, _ = self.vae(hiddens)
        inputs = reconst_state.unsqueeze(1)
        inputs = inputs.repeat(1, seq_len, 1)
        outputs, _ = self.decoder(inputs)
        return outputs


def get_model(config, device):
    """Instantiate model."""
    model = Seq2SeqMelodyComposer(config, device)
    return model
