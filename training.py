# -*- coding: utf-8 -*-
"""Training script for Benzaiten Starter Kit ver. 1.0.

Copyright (C) 2022 by ballforest

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
import os
from collections import namedtuple

import torch
from hydra import compose, initialize
from omegaconf import DictConfig, OmegaConf
from progressbar import progressbar as prg
from torch.nn.utils import clip_grad_norm_

from dataset import get_dataloader
from factory import get_loss, get_lr_scheduler, get_optimizer
from model import get_model


def setup_modules(cfg: DictConfig, device: torch.device):
    """Instantiate modules for training."""
    dataloader = get_dataloader(cfg)
    model = get_model(cfg, device)
    loss_func = get_loss(cfg, model)
    optimizer = get_optimizer(cfg, model)
    lr_scheduler = None
    if cfg.training.use_scheduler:
        lr_scheduler = get_lr_scheduler(cfg, optimizer)
    TrainingModules = namedtuple(
        "TrainingModules",
        ["dataloader", "model", "loss_func", "optimizer", "lr_scheduler"],
    )
    modules = TrainingModules(
        dataloader=dataloader,
        model=model,
        loss_func=loss_func,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
    )
    return modules


def training_step(batch, loss_func, device: torch.device):
    """Perform a training step."""
    source, target = batch
    batch = {"source": source.to(device).float(), "target": target.to(device).long()}
    loss = loss_func(batch)
    return loss


def training_loop(cfg: DictConfig, modules, device: torch.device):
    """Perform training loop."""
    dataloader, model, loss_func, optimizer, lr_scheduler = modules
    model.train()  # turn on train mode
    n_epoch = cfg.training.n_epoch + 1
    for epoch in prg(range(1, n_epoch), prefix="Model training: ", suffix="\n"):
        epoch_loss = 0.0
        for batch in dataloader:
            optimizer.zero_grad()
            loss = training_step(batch, loss_func, device)
            epoch_loss += loss.item()
            loss.backward()
            if cfg.training.use_grad_clip:
                clip_grad_norm_(model.parameters(), cfg.training.grad_max_norm)
            optimizer.step()
        if lr_scheduler is not None:
            lr_scheduler.step()
        epoch_loss = epoch_loss / len(dataloader)
        print(f"Epoch {epoch}: loss = {epoch_loss:.6f}")


def save_checkpoint(cfg: DictConfig, modules):
    """Save checkpoint."""
    model = modules.model
    model_dir = os.path.join(cfg.benzaiten.root_dir, cfg.benzaiten.model_dir)
    model_file = os.path.join(model_dir, cfg.training.model_file)
    torch.save(model.state_dict(), model_file)


def main(cfg: DictConfig):
    """Perform model training."""
    print(OmegaConf.to_yaml(cfg), flush=True)  # dump configuration

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # instantiate modules for training
    modules = setup_modules(cfg, device)

    # perform training loop
    training_loop(cfg, modules, device)

    # save checkpoint
    save_checkpoint(cfg, modules)


if __name__ == "__main__":
    with initialize(version_base=None, config_path="."):
        config = compose(config_name="config")

    main(config)
