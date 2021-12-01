import torch
import numpy as np
import torch.nn.functional as F
import pytorch_lightning as pl
from torch import nn
from sam import SAM


class BaseModel(pl.LightningModule):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.save_hyperparameters()

    def configure_optimizers(self):
        return torch.optim.Adam(
            self.parameters(), lr=self.hparams.lr, weight_decay=1e-5
        )

    def training_step(self, batch, batch_idx):
        outputs = self.forward(batch[0], batch[1])
        self.log("loss", outputs["loss"], on_step=True, on_epoch=False, prog_bar=True)
        self.log("acc", outputs["acc"], on_step=True, on_epoch=False, prog_bar=True)
        return outputs

    def validation_step(self, batch, batch_idx):
        outputs = self.forward(batch[0], batch[1])
        self.log("val/loss", outputs["loss"], on_epoch=True, prog_bar=True)
        self.log("val/acc", outputs["acc"], on_epoch=True, prog_bar=True)


class NoiseNet(BaseModel):
    def __init__(self, *args, noise_level=0.0, dropout=0.0, **kwargs):
        super().__init__(*args, **kwargs)

        self.fc1 = nn.Sequential(nn.Linear(102, 200), nn.ReLU())
        self.fc2 = nn.Linear(200, 1)
        self.noise_level = noise_level
        self.dropout = dropout

    def predict(self, x):
        with torch.no_grad():
            o = torch.sigmoid(self.fc2(self.fc1(x)))
        return o

    def feats(self, x):
        with torch.no_grad():
            return self.fc1(x)

    def forward(self, x, y):
        h = self.fc1(x)
        y = y.squeeze()

        if self.training:
            if self.noise_level > 0.0:
                h = (
                    h
                    + self.noise_level * torch.randn_like(h) * h.norm(2, dim=1)[:, None]
                )
        if self.dropout > 0.0:
            h = F.dropout(h, self.dropout)
        py = torch.sigmoid(self.fc2(h)).squeeze()

        outputs = {}
        loss = F.binary_cross_entropy(py, y)
        acc = py.gt(0.5).float().eq(y).float().mean()

        outputs["acc"] = acc
        outputs["loss"] = loss
        return outputs


class Sam(NoiseNet):
    def __init__(self, *args, noise_level=0, dropout=0, **kwargs):
        super().__init__(*args, noise_level=noise_level, dropout=dropout, **kwargs)
        self.automatic_optimization = False

    def configure_optimizers(self):
        return SAM(
            list(self.parameters()),
            torch.optim.Adam,
            rho=0.5,
            lr=self.hparams.lr,
            weight_decay=1e-5,
        )

    def forward(self, x, y):
        h = self.fc1(x)
        y = y.squeeze()

        if self.training:
            if self.noise_level > 0.0:
                h = (
                    h
                    + self.noise_level * torch.randn_like(h) * h.norm(2, dim=1)[:, None]
                )
        if self.dropout > 0.0:
            h = F.dropout(h, self.dropout)
        py = torch.sigmoid(self.fc2(h)).squeeze()

        outputs = {}
        loss = F.binary_cross_entropy(py, y)
        acc = py.gt(0.5).float().eq(y).float().mean()

        outputs["acc"] = acc
        outputs["loss"] = loss
        return outputs

    def training_step(self, batch, batch_idx):
        if self.hparams.method == "sam_min":
            ex_idxs = torch.where(batch[2] == 1)[0]
            max_batch = batch[0][ex_idxs], batch[1][ex_idxs]
        else:
            max_batch = batch[0], batch[1]

        # max part of sam
        outputs = self.forward(*max_batch)
        outputs["loss"].backward()

        opt = self.optimizers()
        opt.first_step(zero_grad=True)

        # min part of sam
        outputs = self.forward(batch[0], batch[1])
        outputs["loss"].backward()
        opt.second_step(zero_grad=True)

        self.log("loss", outputs["loss"], on_step=True, on_epoch=False, prog_bar=True)
        self.log("acc", outputs["acc"], on_step=True, on_epoch=False, prog_bar=True)


def model_chooser(args):
    method = args.method

    if method in ["base", "reweight"]:
        f = NoiseNet(**args.__dict__, noise_level=0.0)
    elif method in ["sam", "sam_min", "sam_maj"]:
        f = Sam(**args.__dict__, noise_level=0.0)
    return f
