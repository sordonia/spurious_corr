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
    parser.add_argument("--lr", type=float, default=0.0005)
    parser.add_argument("--method", type=str, default="base")
    parser.add_argument("--name", type=str, default=f"run")
    pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    data = LiangDataset(args)
    model = model_chooser(args)
    output_dir = f"./outputs/{args.name}_{datetime.datetime.now()}"
    
    tb_logger = pl.loggers.TensorBoardLogger(output_dir)
    csv_logger = pl.loggers.CSVLogger(output_dir)
    model_checkpoint = pl.callbacks.ModelCheckpoint(output_dir)

    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        gpus=args.gpus,
        logger=[tb_logger, csv_logger],
        callbacks=model_checkpoint
    )
    trainer.fit(model, datamodule=data)
    data.plot_model(model, f"{output_dir}/boundary.png")


train()