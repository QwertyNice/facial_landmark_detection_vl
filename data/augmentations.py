import albumentations as alb

from albumentations.pytorch import ToTensorV2
from albumentations_experimental import HorizontalFlipSymmetricKeypoints


def get_train_aug(config):
    if config['dataset']['augmentations_train'] == 'default':
        train_augs = alb.Compose([
            alb.Resize(width=config['dataset']['input_size'],
                       height=config['dataset']['input_size']),
            HorizontalFlipSymmetricKeypoints(symmetric_keypoints={
                0: 16, 1: 15, 2: 14, 3: 13, 4: 12, 5: 11, 6: 10, 7: 9, 8: 8,
                17: 26, 18: 25, 19: 24, 20: 23, 21: 22, 36: 45, 37: 44, 38: 43,
                39: 42, 40: 47, 41: 46, 27: 27, 28: 28, 29: 29, 30: 30, 31: 35,
                32: 34, 33: 33, 50: 52, 51: 51, 49: 53, 61: 63, 62: 62, 48: 54,
                60: 64, 67: 65, 66: 66, 59: 55, 58: 56, 57: 57
            }, p=0.5),
            alb.Rotate(limit=45, p=0.5),
            alb.RandomBrightnessContrast(p=0.25),
            alb.Normalize(mean=(0.485, 0.456, 0.406),
                          std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ], keypoint_params=alb.KeypointParams(format='xy',
                                              remove_invisible=False))
    else:
        raise Exception("Unknown type of augs: {}".format(
            config['dataset']['augmentations_train']
        ))
    return train_augs


def get_val_aug(config):
    if config['dataset']['augmentations_val'] == 'default':
        val_augs = alb.Compose([
            alb.Resize(width=config['dataset']['input_size'],
                       height=config['dataset']['input_size']),
            alb.Normalize(mean=(0.485, 0.456, 0.406),
                          std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ], keypoint_params=alb.KeypointParams(format='xy',
                                              remove_invisible=False))
    else:
        raise Exception("Unknown type of augs: {}".format(
            config['dataset']['augmentations_val']
        ))
    return val_augs


def get_test_aug(img_size):
    test_aug = alb.Compose([
        alb.Resize(width=img_size, height=img_size),
        alb.Normalize(mean=(0.485, 0.456, 0.406),
                      std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])
    return test_aug
