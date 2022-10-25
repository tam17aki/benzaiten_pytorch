# -*- coding: utf-8 -*-
"""Dataset script for Benzaiten Starter Kit ver. 1.0

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
import os
import random

import joblib
from torch.utils.data import DataLoader, Dataset


class BenzaitenDataset(Dataset):
    """Build Dataset for training."""

    def __init__(self, cfg):
        """Initialize class."""
        super().__init__()
        self.cfg = cfg
        feats_dir = os.path.join(
            self.cfg.benzaiten.root_dir, self.cfg.benzaiten.feat_dir
        )
        feat_file = os.path.join(feats_dir, self.cfg.preprocess.feat_file)
        features = joblib.load(feat_file)
        self.data_all = features["data"]
        self.label_all = features["label"]

    def __len__(self):
        """Return dataset size."""
        return self.data_all.shape[0]

    def __getitem__(self, idx):
        """Fetch items."""
        return self.data_all[idx], self.label_all[idx]


def _worker_init_fn(worker_id):
    """Initialize worker functions."""
    random.seed(worker_id)


def get_dataloader(cfg):
    """Return dataloader from dataset."""
    dataset = BenzaitenDataset(cfg)
    dataloader = DataLoader(
        dataset,
        batch_size=cfg.training.n_batch,
        shuffle=True,
        drop_last=True,
        num_workers=2,  # for faster computation
        pin_memory=True,  # for faster computation
        worker_init_fn=_worker_init_fn,
    )
    return dataloader
