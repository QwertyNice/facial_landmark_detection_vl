import os

import cv2
import numpy as np
import pandas as pd
import torch
import torch.utils.data as data


def read_image(image_file):
    img = cv2.imread(
        image_file, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION
    )
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if img is None:
        raise ValueError("Failed to read {}".format(image_file))
    return img


def read_pts(filename):
    return np.loadtxt(filename, comments=("version:", "n_points:", "{", "}"))


class FacesDataset(data.Dataset):
    def __init__(
        self,
        root,
        is_train,
        padding,
        img_size,
        transforms=None,
        num_of_landmarks=68,
        num_threads=12,
    ):
        self.root = root
        self.padding = padding
        self.img_size = img_size
        self.num_of_landmarks = num_of_landmarks
        if is_train:
            self.images_pd = pd.read_csv(os.path.join(root, "train.csv"))
        else:
            self.images_pd = pd.read_csv(os.path.join(root, "test.csv"))

        self.num_threads = num_threads
        self.transforms = transforms

    def __getitem__(self, index):
        cv2.setNumThreads(self.num_threads)

        img_path, *bbox, _, img_height, img_width = self.images_pd.loc[index]
        full_img_path = os.path.join(
            self.root, img_path[1:] if img_path.startswith("/") else img_path
        )
        img = read_image(full_img_path)
        points = read_pts("".join(full_img_path.split(".")[:-1]) + ".pts")

        bbox_height, bbox_width = bbox[3] - bbox[1], bbox[2] - bbox[0]
        bbox[0] = int(max(0, bbox[0] - self.padding[0] / 100 * bbox_width))
        bbox[1] = int(max(0, bbox[1] - self.padding[1] / 100 * bbox_height))
        bbox[2] = int(
            min(img_width, bbox[2] + self.padding[2] / 100 * bbox_width)
        )
        bbox[3] = int(
            min(img_height, bbox[3] + self.padding[3] / 100 * bbox_height)
        )
        img = img[bbox[1] : bbox[3], bbox[0] : bbox[2], :]
        points[:, 0] -= bbox[0]
        points[:, 1] -= bbox[1]

        if self.transforms is not None:
            transformed = self.transforms(image=img, keypoints=points)
            img, points = transformed["image"], transformed["keypoints"]

        if not torch.is_tensor(points):
            points = torch.tensor(points, dtype=torch.float)

        points /= self.img_size

        return img, points.view(-1)

    def __len__(self):
        return len(self.images_pd)
