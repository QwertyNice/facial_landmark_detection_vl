import argparse
import os
import sys

import cv2
import numpy as np
import pandas as pd
import torch
import tqdm
import yaml

from collections import OrderedDict

from models.mobileone import reparameterize_model
from data.augmentations import get_test_aug
from utils.utils import get_model


def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cfg",
        type=str,
        default="./configs/default.yml",
        help="Path to config file.",
    )
    parser.add_argument(
        "--weights", type=str, required=True, help="Path to model weights."
    )
    parser.add_argument(
        "--save_path", type=str, required=True, help="Path to output files."
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        required=True,
        help="Path to dataset with test.csv inside.",
    )
    parser.add_argument(
        "--dataset_type",
        type=str,
        choices=["Menpo", "300W", "all"],
        required=True,
        help="Name of the dataset to process.",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = parse_arguments(sys.argv[1:])
    with open(args.cfg) as f:
        data = yaml.safe_load(f)

    model = get_model(config=data)
    checkpoint = torch.load(args.weights)
    dataset_type = (
        "" if args.dataset_type.lower() == "all" else args.dataset_type
    )

    new_state_dict = OrderedDict()
    for key, value in checkpoint["state_dict"].items():
        new_key = key.replace("net.", "").replace("net_val.", "")
        new_state_dict[new_key] = value
    model.load_state_dict(new_state_dict, strict=False)
    model.eval()

    if data["train"]["model_arch"] == "mobileone":
        model = reparameterize_model(model)

    ants = pd.read_csv(os.path.join(args.dataset_path, "test.csv"))
    transforms = get_test_aug(img_size=data["dataset"]["input_size"])

    with torch.no_grad():
        for i in tqdm.tqdm(range(len(ants))):
            img_path, *bbox, _, img_height, img_width = ants.iloc[i]
            if dataset_type in img_path.split("/"):
                img = cv2.imread(
                    (
                        args.dataset_path[:-1]
                        if args.dataset_path.endswith("/")
                        else args.dataset_path
                    )
                    + img_path
                )[..., ::-1]

                bbox_height, bbox_width = bbox[3] - bbox[1], bbox[2] - bbox[0]
                bbox[0] = int(
                    max(
                        0,
                        bbox[0]
                        - data["dataset"]["padding"]["left"]
                        / 100
                        * bbox_width,
                    )
                )
                bbox[1] = int(
                    max(
                        0,
                        bbox[1]
                        - data["dataset"]["padding"]["top"]
                        / 100
                        * bbox_height,
                    )
                )
                bbox[2] = int(
                    min(
                        img_width,
                        bbox[2]
                        + data["dataset"]["padding"]["right"]
                        / 100
                        * bbox_width,
                    )
                )
                bbox[3] = int(
                    min(
                        img_height,
                        bbox[3]
                        + data["dataset"]["padding"]["bottom"]
                        / 100
                        * bbox_height,
                    )
                )
                img = img[bbox[1] : bbox[3], bbox[0] : bbox[2], :]
                dx = img.shape[1] / data["dataset"]["input_size"]
                dy = img.shape[0] / data["dataset"]["input_size"]

                img = transforms(image=img)["image"].unsqueeze(0)
                landmarks = model(img)

                points = (
                    (landmarks * data["dataset"]["input_size"])
                    .view(data["dataset"]["num_of_landmarks"], 2)
                    .numpy()
                )
                points[:, 0] *= dx
                points[:, 1] *= dy
                points[:, 0] += bbox[0]
                points[:, 1] += bbox[1]

                path2save = os.path.join(
                    args.save_path,
                    f'{data["train"]["model_arch"]}_{args.dataset_type.lower()}',
                )
                if not os.path.exists(path2save):
                    os.makedirs(path2save)

                np.savetxt(
                    f'{path2save}/{img_path.split("/")[-1].split(".")[0]}.pts',
                    points,
                    delimiter=" ",
                    fmt="%1.3f",
                    header="version: 1\nn_points: 68\n{",
                    footer="}",
                    comments="",
                )
