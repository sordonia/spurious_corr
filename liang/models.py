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

    def get_accs(self, preds, targets, groups):
        acc = preds.gt(0.5).float().eq(targets).float()
        maj_idx = groups == 0
        min_idx = groups == 1
        acc_min = acc[min_idx].mean()
        acc_maj = acc[maj_idx].mean()
        return acc.mean(), acc_min, acc_maj

    def training_step(self, batch, batch_idx):
        outputs = self.forward(batch[0], batch[1], batch[2])
        acc, acc_min, acc_maj = self.get_accs(outputs["preds"], batch[1], batch[2])
        self.log("loss", outputs["loss"], on_step=True, on_epoch=False, prog_bar=True)
        self.log("acc", acc, on_step=True, on_epoch=False, prog_bar=True)
        self.log("acc_min", acc_min, on_step=True, on_epoch=False, prog_bar=True)
        self.log("acc_maj", acc_maj, on_step=True, on_epoch=False, prog_bar=True)
        return outputs

    def training_epoch_end(self, outputs) -> None:
        self.trainer.datamodule.plot_model(
            self, f"{self.hparams.output_dir}/boundary_e{self.current_epoch}.png"
        )

    def validation_step(self, batch, batch_idx):
        outputs = self.forward(batch[0], batch[1], batch[2])
        acc, acc_min, acc_maj = self.get_accs(outputs["preds"], batch[1], batch[2])

        self.log("val/loss", outputs["loss"], on_epoch=True, prog_bar=True)
        self.log("val/acc", acc, on_epoch=True, prog_bar=True)
        self.log("val/acc_min", acc_min, on_epoch=True, prog_bar=True)
        self.log("val/acc_maj", acc_maj, on_epoch=True, prog_bar=True)


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

    def forward(self, x, y, g=None):
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

        outputs["preds"] = py
        outputs["loss"] = loss
        return outputs


class Reweight(NoiseNet):
    def forward(self, x, y, g=None):
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

        # group indices
        maj_idx = g == 0
        min_idx = g == 1
        loss = 0.5 * (
            F.binary_cross_entropy(py[maj_idx], y[maj_idx])
            + 0.5 * self.hparams.min_reweight_factor * F.binary_cross_entropy(py[min_idx], y[min_idx])
        )

        outputs["preds"] = py
        outputs["loss"] = loss
        return outputs


class FeatureSelectionNet(BaseModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fc1 = nn.Sequential(nn.Linear(102, 200), nn.ReLU())
        self.fc2 = nn.Linear(200, 1)
        self.gs = nn.Parameter(torch.ones(102))

    def predict(self, x):
        with torch.no_grad():
            o = torch.sigmoid(self.fc2(self.fc1(x)))
        return o

    def forward(self, x, y, g=None):
        g = F.softmax(self.gs)
        # this applies a consistent mask for each input
        h = self.fc1(x * g[None, :])
        py = torch.sigmoid(self.fc2(h)).squeeze()

        outputs = {}
        loss = F.binary_cross_entropy(py, y)
        ent = -(g * torch.log(g + 1e-6)).sum()
        outputs['preds'] = py
        outputs['ent'] = ent
        outputs['loss'] = loss + ent
        return outputs


class Sam(NoiseNet):
    def __init__(self, *args, noise_level=0, dropout=0, **kwargs):
        super().__init__(*args, noise_level=noise_level, dropout=dropout, **kwargs)
        self.automatic_optimization = False

    def configure_optimizers(self):
        return SAM(
            list(self.parameters()),
            torch.optim.Adam,
            rho=self.hparams.rho,
            lr=self.hparams.lr,
            weight_decay=1e-5,
        )

    def forward(self, x, y, g=None):
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

        outputs["preds"] = py
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
        outputs = self.forward(batch[0], batch[1], batch[2])
        outputs["loss"].backward()
        opt.second_step(zero_grad=True)

        acc, acc_min, acc_maj = self.get_accs(outputs["preds"], batch[1], batch[2])
        self.log("loss", outputs["loss"], on_step=True, on_epoch=False, prog_bar=True)
        self.log("acc", acc, on_step=True, on_epoch=False, prog_bar=True)
        self.log("acc_min", acc_min, on_step=True, on_epoch=False, prog_bar=True)
        self.log("acc_maj", acc_maj, on_step=True, on_epoch=False, prog_bar=True)


def model_chooser(args):
    method = args.method

    if method in ["base"]:
        f = NoiseNet(**args.__dict__, noise_level=0.0)
    elif method in ["sam", "sam_min", "sam_maj"]:
        f = Sam(**args.__dict__, noise_level=0.0)
    elif method in ["feat_selection_x"]:
        f = FeatureSelectionNet(**args.__dict__)
    elif method in ["reweight"]:
        f = Reweight(**args.__dict__)
    return f
