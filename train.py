import argparse
import os
import sys
import warnings

import cv2
import numpy as np
import torch
import wandb
import yaml

from torchvision import transforms
from pytorch_lightning import LightningModule, Trainer, seed_everything
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint

from target_metric.metric import count_ced
from models.mobileone import reparameterize_model
from utils.utils import (get_loss,
                         get_optimizer,
                         get_scheduler,
                         get_train_dataloader,
                         get_val_dataloader,
                         get_model)


class FacialLandmarkDetection(LightningModule):
    def __init__(self, config, dataloader_train, dataloader_val):
        super(FacialLandmarkDetection, self).__init__()
        self.config = config
        self.dataloader_train = dataloader_train
        self.dataloader_val = dataloader_val
        self.land_gt, self.land_pred = dict(), dict()

        self.net = get_model(config=config)
        self.net_val = None  # only for MobileOne arch
        self.loss = get_loss(loss_name=config["train"]["loss"])

    def forward(self, x):
        return self.net(x)

    def mo_val_forward(self, x):
        return self.net_val(x)

    def training_step(self, batch, batch_idx):
        face_images, landmarks_gt = batch
        landmarks = self(face_images)
        loss = self.loss(landmarks, landmarks_gt)
        return loss

    def training_epoch_end(self, training_step_outputs):
        avg_train_loss = torch.stack(
            [x["loss"] for x in training_step_outputs]
        ).mean()
        avg_train_loss = {
            "epoch": self.current_epoch,
            "train_loss": avg_train_loss,
        }
        wandb.log(avg_train_loss)

    def validation_step(self, batch, batch_nb):
        face_images, landmarks_gt = batch
        if self.config["train"]["model_arch"].lower() == "mobileone":
            if batch_nb == 0:
                self.net_val = reparameterize_model(self.net)
            landmarks = self.mo_val_forward(face_images)
        else:
            landmarks = self(face_images)

        loss = self.loss(landmarks, landmarks_gt)
        self.log("val_loss_log", loss, on_epoch=True)

        if batch_nb % data["logging"]["wandb"]["img_log_every"] == 0:
            image_gt, image_pred = self.visualize_images_wandb(
                image=face_images[0],
                landmark_pred=landmarks[0],
                landmarks_gt=landmarks_gt[0],
            )

            wandb.log(
                {
                    "epoch": self.current_epoch,
                    "ground_true:": wandb.Image(image_gt),
                    "predictions": wandb.Image(image_pred),
                }
            )

        landmarks = (
            landmarks.view(self.config["dataset"]["num_of_landmarks"], 2)
            .cpu()
            .numpy()
        )
        landmarks_gt = (
            landmarks_gt.view(self.config["dataset"]["num_of_landmarks"], 2)
            .cpu()
            .numpy()
        )
        landmarks = (landmarks * self.config["dataset"]["input_size"]).astype(
            int
        )
        landmarks_gt = (
            landmarks_gt * self.config["dataset"]["input_size"]
        ).astype(int)
        self.land_pred[batch_nb] = (landmarks[:, 0], landmarks[:, 1])
        self.land_gt[batch_nb] = (landmarks_gt[:, 0], landmarks_gt[:, 1])
        return loss

    def visualize_images_wandb(self, image, landmark_pred, landmarks_gt):
        # Function for correct rendering of images in wandb
        inv_trans = transforms.Compose(
            [
                transforms.Normalize(
                    mean=[0.0, 0.0, 0.0], std=[1 / 0.229, 1 / 0.224, 1 / 0.225]
                ),
                transforms.Normalize(
                    mean=[-0.485, -0.456, -0.406], std=[1.0, 1.0, 1.0]
                ),
            ]
        )
        inv_tensor = inv_trans(image)
        image_gt = np.array(
            inv_tensor.permute(1, 2, 0).cpu() * 255, dtype=np.uint8
        ).copy()
        landmark_pred = (
            landmark_pred.view(self.config["dataset"]["num_of_landmarks"], 2)
            .cpu()
            .numpy()
        )
        landmarks_gt = (
            landmarks_gt.view(self.config["dataset"]["num_of_landmarks"], 2)
            .cpu()
            .numpy()
        )
        landmark_pred = (
            landmark_pred * self.config["dataset"]["input_size"]
        ).astype(int)
        landmarks_gt = (
            landmarks_gt * self.config["dataset"]["input_size"]
        ).astype(int)
        image_pred = image_gt.copy()
        image_pred = np.ascontiguousarray(image_pred, dtype=np.uint8)

        for index, point in enumerate(landmark_pred):
            image_pred = cv2.circle(
                image_pred, point, radius=0, color=(0, 255, 255), thickness=2
            )

        for index, point in enumerate(landmarks_gt):
            image_gt = cv2.circle(
                image_gt, point, radius=0, color=(0, 255, 255), thickness=2
            )
        return image_gt, image_pred

    def validation_epoch_end(self, outputs):
        avg_val_loss = torch.stack([x for x in outputs]).mean()
        val_ced_auc = count_ced(
            predicted_points=self.land_pred, gt_points=self.land_gt
        )
        avg_val_loss = {
            "epoch": self.current_epoch,
            "val_loss": avg_val_loss,
            "val_ced_auc": val_ced_auc,
        }
        wandb.log(avg_val_loss)
        self.land_gt.clear()
        self.land_pred.clear()

    def configure_optimizers(self):
        optimizer = get_optimizer(config=self.config, model=self.net)
        if "lr_scheduler" not in self.config["train"]:
            message = 'Training without scheduler'
            warnings.warn(message, Warning)
            return optimizer
        scheduler = get_scheduler(config=self.config, optimizer=optimizer)
        return [optimizer], [scheduler]

    def train_dataloader(self):
        return self.dataloader_train

    def val_dataloader(self):
        return self.dataloader_val


def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cfg", type=str, default="./configs/default.yml",
        help="Path to config file.",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = parse_arguments(sys.argv[1:])
    with open(args.cfg) as f:
        data = yaml.safe_load(f)

    if not data["logging"]["wandb"]["logging"]:
        wandb.init(mode="disabled")
    seed_everything(data["dataset"]["seed"])

    model = FacialLandmarkDetection(
        config=data,
        dataloader_train=get_train_dataloader(config=data),
        dataloader_val=get_val_dataloader(config=data),
    )

    wandb_logger = WandbLogger(
        project="FacialLandmarkDetection", name=data["exp_name"]
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(data["out_dir"], data["exp_name"]),
        filename=data["train"]["model_arch"] + "-{epoch}-{val_loss_log:.4f}",
        save_top_k=3,
        monitor="val_loss_log",
    )

    trainer = Trainer(
        num_sanity_val_steps=0,
        gpus=1,
        fast_dev_run=False,  # only for debug
        logger=wandb_logger,
        max_epochs=data["train"]["n_epoch"],
        precision=16 if data["train"]["fp16"] else 32,
        callbacks=[checkpoint_callback],
    )
    _ = trainer.fit(model)
