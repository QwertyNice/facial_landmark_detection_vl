import numpy as np


def count_ced_auc(error):
    auc = 0
    proportions = np.arange(error.shape[0], dtype=np.float32) / error.shape[0]
    step = 0.01
    for thr in np.arange(0.0, 1.0, step):
        gt_indexes = [idx for idx, e in enumerate(error) if e >= thr]
        if len(gt_indexes) > 0:
            first_gt_idx = gt_indexes[0]
        else:
            first_gt_idx = len(error) - 1
        auc += proportions[first_gt_idx] * step
    return auc


def count_ced(predicted_points, gt_points):
    ceds = list()
    for index in predicted_points.keys():
        x_pred, y_pred = predicted_points[index]
        x_gt, y_gt = gt_points[index]
        n_points = x_pred.shape[0]

        w = np.max(x_gt) - np.min(x_gt)
        h = np.max(y_gt) - np.min(y_gt)
        normalization_factor = np.sqrt(h * w)

        diff_x = [x_gt[i] - x_pred[i] for i in range(n_points)]
        diff_y = [y_gt[i] - y_pred[i] for i in range(n_points)]
        dist = np.sqrt(np.square(diff_x) + np.square(diff_y))
        avg_norm_dist = np.sum(dist) / (n_points * normalization_factor)
        ceds.append(avg_norm_dist)
    return count_ced_auc(np.sort(ceds))
