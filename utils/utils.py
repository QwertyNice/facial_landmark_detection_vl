import warnings

import torch

from torch import nn
from torch.utils.data import DataLoader

from data.augmentations import get_train_aug, get_val_aug
from data.dataset import FacesDataset


def weights_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight.data)
        nn.init.constant_(m.bias, 0.1)


def get_model(config):
    model_arch = config["train"]["model_arch"]
    pretrained_weights = config["train"]["pretrained_weights"]
    if model_arch.lower() == "onet":
        from models.onet import ONet
        model = ONet(
            input_size=config["dataset"]["input_size"],
            total_landmarks=config["dataset"]["num_of_landmarks"],
        )
        if pretrained_weights:
            message = 'Current architecture does not have pretrained weights'
            warnings.warn(message, Warning)
    elif model_arch.lower() == "mini_pfld":
        from models.mini_pfld import MiniPFLDInference
        model = MiniPFLDInference(
            input_size=config["dataset"]["input_size"],
            total_landmarks=config["dataset"]["num_of_landmarks"],
        )
        if pretrained_weights:
            message = 'Current architecture does not have pretrained weights'
            warnings.warn(message, Warning)
    elif model_arch.lower() == "mobileone":
        from models.mobileone import mobileone
        model = mobileone(
            variant="s1",
            num_classes=config["dataset"]["num_of_landmarks"] * 2,
        )

        if pretrained_weights:
            checkpoint = torch.load(pretrained_weights)
            checkpoint.pop("linear.weight")
            checkpoint.pop("linear.bias")
            model.load_state_dict(checkpoint, strict=False)
    else:
        message = (
            f'Selected arch ({model_arch}) '
            f"does not exist."
        )
        raise Exception(message)
    return model


def get_loss(loss_name):
    if loss_name.lower() == 'mse':
        loss = nn.MSELoss()
    elif loss_name.lower() == 'wingloss':
        from utils.loss import WingLoss
        loss = WingLoss()
    elif loss_name.lower() == 'adawingloss':
        from utils.loss import AdaptiveWingLoss
        loss = AdaptiveWingLoss()
    elif loss_name.lower() == 'smoothl1loss':
        loss = nn.SmoothL1Loss()
    else:
        message = f'Selected loss ({loss_name}) does not exist.'
        raise Exception(message)
    return loss


def get_optimizer(config, model):
    optimizer_name = config['train']['optimizer']['name']
    if optimizer_name.lower() == 'adam':
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config['train']['learning_rate'],
            betas=(config['train']['optimizer']['beta1'],
                   config['train']['optimizer']['beta2']),
            weight_decay=config['train']['optimizer']['weight_decay']
        )
    elif optimizer_name.lower() == 'adamw':
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config['train']['learning_rate'],
            betas=(config['train']['optimizer']['beta1'],
                   config['train']['optimizer']['beta2']),
            weight_decay=config['train']['optimizer']['weight_decay']
        )
    elif optimizer_name.lower() == 'sgd':
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=config['train']['learning_rate'],
            momentum=config['train']['optimizer']['momentum'],
            weight_decay=config['train']['optimizer']['weight_decay']
        )
    else:
        message = f'Selected optimizer ({optimizer_name}) does not exist.'
        raise Exception(message)
    return optimizer


def get_scheduler(config, optimizer):
    scheduler_name = config['train']['lr_scheduler']['name']
    if scheduler_name.lower() == 'steplr':
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=config['train']['lr_scheduler']['step_size'],
            gamma=config['train']['lr_scheduler']['gamma']
        )
    elif scheduler_name.lower() == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config['train']['n_epoch']
        )
    else:
        message = f'Selected scheduler ({scheduler_name}) does not exist.'
        raise Exception(message)
    return scheduler


def get_train_dataloader(config):
    train_dataset = FacesDataset(
        root=config["dataset"]["root"],
        is_train=True,
        padding=(
            config["dataset"]["padding"]["left"],
            config["dataset"]["padding"]["top"],
            config["dataset"]["padding"]["right"],
            config["dataset"]["padding"]["bottom"],
        ),
        img_size=config["dataset"]["input_size"],
        transforms=get_train_aug(config=config),
        num_of_landmarks=config["dataset"]["num_of_landmarks"],
        num_threads=config["dataset"]["num_workers"],
    )
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config["dataset"]["batch_size"],
        shuffle=True,
        num_workers=config["dataset"]["num_workers"],
        drop_last=False,
    )
    return train_dataloader


def get_val_dataloader(config, batch_size=1):
    val_dataset = FacesDataset(
        root=config["dataset"]["root"],
        is_train=False,
        padding=(
            config["dataset"]["padding"]["left"],
            config["dataset"]["padding"]["top"],
            config["dataset"]["padding"]["right"],
            config["dataset"]["padding"]["bottom"],
        ),
        img_size=config["dataset"]["input_size"],
        transforms=get_val_aug(config=config),
        num_of_landmarks=config["dataset"]["num_of_landmarks"],
        num_threads=config["dataset"]["num_workers"],
    )

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=config["dataset"]["num_workers"],
        drop_last=False,
    )
    return val_dataloader
