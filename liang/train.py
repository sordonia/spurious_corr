from pytorch_lightning import callbacks
import torch
from dataset import LiangDataset
import pytorch_lightning as pl
import argparse
from models import model_chooser
from matplotlib import pyplot as plt
import datetime


def train():
    parser = argparse.ArgumentParser()

    parser.add_argument("--batch_size", type=int, default=-1)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--rho", type=float, default=0.5, help="sam rho")
    parser.add_argument("--method", type=str, default="base")
    parser.add_argument("--name", type=str, default=f"run")
    parser.add_argument("--min_reweight_factor", type=float, default=1., help="Reweighting approach minorities weight")

    pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()
    output_dir = f"./outputs/run_{args.name}_{datetime.datetime.now()}"
    args.output_dir = output_dir

    data = LiangDataset(args)
    model = model_chooser(args)

    tb_logger = pl.loggers.TensorBoardLogger(save_dir=output_dir)
    csv_logger = pl.loggers.CSVLogger(output_dir)
    model_checkpoint = pl.callbacks.ModelCheckpoint(
        dirpath=output_dir,
        mode="max",
        monitor="val/acc",
        auto_insert_metric_name=False,
        filename='epoch{epoch:02d}-acc{val/acc:.2f}'
    )

    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        gpus=args.gpus,
        logger=[tb_logger, csv_logger],
        callbacks=model_checkpoint
    )
    data.plot_model(model, "test.png")
    trainer.fit(model, datamodule=data)


train()