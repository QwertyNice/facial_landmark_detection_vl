import argparse
import pathlib
import sys

import dlib
import numpy as np
import pandas as pd

from tqdm import tqdm


def read_pts(filename):
    return np.loadtxt(filename, comments=("version:", "n_points:", "{", "}"))


def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, help="Path to dataset.")
    parser.add_argument(
        "--nose-point", type=int, default=33, help="Center point of nose."
    )
    parser.add_argument(
        "--total_landmarks",
        type=int,
        default=68,
        help="Total number of landmarks.",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = parse_arguments(sys.argv[1:])
    path2dataset = args.path

    all_files_path = list(pathlib.Path(path2dataset).glob("**/*.*"))
    all_images_path = sorted(
        list(
            filter(
                lambda x: x.__str__()
                .lower()
                .endswith((".png", ".jpg", "jpeg")),
                all_files_path,
            )
        )
    )

    detector = dlib.get_frontal_face_detector()
    output_table_train = list()
    output_table_test = list()
    photos_without_face = list()

    for image_path in tqdm(all_images_path):
        img = dlib.load_rgb_image(image_path.__str__())  # (H, W, C)
        img_h, img_w, _ = img.shape
        relative_image_path = image_path.__str__().split(path2dataset)[-1]
        markup_path = "".join(image_path.__str__().split(".")[:-1]) + ".pts"
        markup = read_pts(filename=markup_path)
        if len(markup) != args.total_landmarks:
            continue
        cntr_point = markup[args.nose_point]

        detections = detector.run(img, 1)

        if len(detections[0]) == 0:
            photos_without_face.append([relative_image_path, 0])
        elif len(detections[0]) == 1:
            if image_path.parent.name == "train":
                output_table_train.append(
                    [
                        relative_image_path,
                        detections[0][0].left(),
                        detections[0][0].top(),
                        detections[0][0].right(),
                        detections[0][0].bottom(),
                        round(detections[1][0], 4),
                        img_h,
                        img_w,
                    ]
                )
            elif image_path.parent.name == "test":
                output_table_test.append(
                    [
                        relative_image_path,
                        detections[0][0].left(),
                        detections[0][0].top(),
                        detections[0][0].right(),
                        detections[0][0].bottom(),
                        round(detections[1][0], 4),
                        img_h,
                        img_w,
                    ]
                )
        else:
            rectangle_centers = np.array(
                [[i.dcenter().x, i.dcenter().y] for i in detections[0]]
            )
            dists = np.linalg.norm(cntr_point - rectangle_centers, axis=1)
            closest_rect_idx = np.argmin(dists)

            if not (
                (
                    detections[0][closest_rect_idx].left()
                    < cntr_point[0]
                    < detections[0][closest_rect_idx].right()
                )
                and (
                    detections[0][closest_rect_idx].top()
                    < cntr_point[1]
                    < detections[0][closest_rect_idx].bottom()
                )
            ):
                photos_without_face.append([relative_image_path, 1])
                continue

            if image_path.parent.name == "train":
                output_table_train.append(
                    [
                        relative_image_path,
                        detections[0][closest_rect_idx].left(),
                        detections[0][closest_rect_idx].top(),
                        detections[0][closest_rect_idx].right(),
                        detections[0][closest_rect_idx].bottom(),
                        round(detections[1][closest_rect_idx], 4),
                        img_h,
                        img_w,
                    ]
                )
            elif image_path.parent.name == "test":
                output_table_test.append(
                    [
                        relative_image_path,
                        detections[0][closest_rect_idx].left(),
                        detections[0][closest_rect_idx].top(),
                        detections[0][closest_rect_idx].right(),
                        detections[0][closest_rect_idx].bottom(),
                        round(detections[1][closest_rect_idx], 4),
                        img_h,
                        img_w,
                    ]
                )

    train_dataset_pd = pd.DataFrame(
        output_table_train,
        columns=[
            "img_name",
            "left",
            "top",
            "right",
            "bottom",
            "confidence",
            "frame_height",
            "frame_width",
        ],
    )
    test_dataset_pd = pd.DataFrame(
        output_table_test,
        columns=[
            "img_name",
            "left",
            "top",
            "right",
            "bottom",
            "confidence",
            "frame_height",
            "frame_width",
        ],
    )
    empty_photos_pd = pd.DataFrame(
        photos_without_face, columns=["img_name", "error_type"]
    )

    train_dataset_pd.to_csv(
        pathlib.Path(path2dataset) / "train.csv", index=False
    )
    test_dataset_pd.to_csv(
        pathlib.Path(path2dataset) / "test.csv", index=False
    )
    empty_photos_pd.to_csv(
        pathlib.Path(path2dataset) / "empty_photos.csv", index=False
    )

    print(
        f"Congrats! You have created 3 csv files:\n "
        f'1) {pathlib.Path(path2dataset) / "train.csv"}\n '
        f'2) {pathlib.Path(path2dataset) / "test.csv"}\n '
        f'3) {pathlib.Path(path2dataset) / "empty_photos.csv"}'
    )
