"""Behavioural cloning of the expert demonstrator."""

import numpy as np
import torch
import torch.nn as nn

from . import config as C
from .policy import MLPPolicy


def _loader(obs, act, batch, seed, shuffle=True):
    ds = torch.utils.data.TensorDataset(torch.as_tensor(obs), torch.as_tensor(act))
    g = torch.Generator().manual_seed(seed)
    return torch.utils.data.DataLoader(ds, batch_size=batch, shuffle=shuffle, generator=g)


def fit(model, obs, act, epochs, lr, batch, seed, val=None, log_every=0):
    """Generic supervised fit. Used for both BC and post-training so that the
    optimiser type and loss are identical across every arm of the experiment."""
    torch.manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    dl = _loader(obs, act, batch, seed)
    history = []
    n_grad_steps = 0
    for ep in range(epochs):
        model.train()
        tot = 0.0
        for xb, yb in dl:
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
            tot += loss.item() * len(xb)
            n_grad_steps += 1
        tr = tot / max(len(obs), 1)
        vl = float("nan")
        if val is not None and len(val[0]) > 0:
            model.eval()
            with torch.no_grad():
                vl = loss_fn(model(torch.as_tensor(val[0])), torch.as_tensor(val[1])).item()
        history.append({"epoch": ep, "train_mse": tr, "val_mse": vl})
        if log_every and (ep + 1) % log_every == 0:
            print(f"      epoch {ep+1:3d}/{epochs}  train_mse={tr:.5f}  val_mse={vl:.5f}")
    model.eval()
    return history, n_grad_steps


def train_baseline(train_xy, val_xy, seed, log_every=20):
    """Train the baseline policy from scratch with behavioural cloning."""
    obs, act = train_xy
    # Seed BEFORE constructing the model: weight initialisation must be part of
    # what the training seed controls, otherwise runs are not reproducible.
    torch.manual_seed(seed)
    model = MLPPolicy()
    model.set_normalizer(obs.mean(axis=0), obs.std(axis=0))
    history, n_steps = fit(model, obs, act, C.BC_EPOCHS, C.BC_LR, C.BC_BATCH,
                           seed, val=val_xy, log_every=log_every)
    return model, history, n_steps
