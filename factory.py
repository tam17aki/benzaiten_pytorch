# -*- coding: utf-8 -*-
"""A Python module which provides optimizer, scheduler, and customized loss.

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
from omegaconf import DictConfig
from torch import nn, optim


def get_optimizer(cfg: DictConfig, model):
    """Instantiate optimizer."""
    optimizer_class = getattr(optim, cfg.training.optim.optimizer.name)
    optimizer = optimizer_class(
        model.parameters(), **cfg.training.optim.optimizer.params
    )
    return optimizer


def get_lr_scheduler(cfg: DictConfig, optimizer):
    """Instantiate scheduler."""
    lr_scheduler_class = getattr(
        optim.lr_scheduler, cfg.training.optim.lr_scheduler.name
    )
    lr_scheduler = lr_scheduler_class(
        optimizer, **cfg.training.optim.lr_scheduler.params
    )
    return lr_scheduler


class CustomLoss(nn.Module):
    """Custom loss."""

    def __init__(self, cfg: DictConfig, model):
        """Initialize class."""
        super().__init__()
        self.cfg = cfg
        self.model = model
        self.criterion = nn.CrossEntropyLoss()
        self.kl_weight = cfg.model.vae.kl_weight

    def forward(self, batch):
        """Compute loss."""
        vae = {"reconst": 0.0, "mean": 0.0, "logvar": 0.0}
        loss = {"xent": 0.0, "kl": 0.0}
        loss_func = {"xent": nn.CrossEntropyLoss(), "mse": nn.MSELoss()}

        source = batch["source"]  # melody + chord
        target = batch["target"]  # melody

        _, encoder_state = self.model.encoder(source)
        hiddens = torch.squeeze(encoder_state[0])  # [32, 1024]
        vae["reconst"], vae["mean"], vae["logvar"] = self.model.vae(hiddens)
        inputs = vae["reconst"].unsqueeze(1)
        inputs = inputs.repeat(1, target.shape[1], 1)
        outputs, _ = self.model.decoder(inputs)
        outputs = torch.permute(outputs, (0, 2, 1))

        loss["xent"] = loss_func["xent"](outputs, target)
        loss["kl"] = (-0.5) * torch.sum(
            1 + vae["logvar"] - vae["mean"].pow(2) - vae["logvar"].exp()
        )
        return loss["xent"] + self.kl_weight * loss["kl"]


def get_loss(cfg: DictConfig, model):
    """Instantiate customized loss."""
    custom_loss = CustomLoss(cfg, model)
    return custom_loss
