import cv2
import albumentations as A

GWHD_PAD_VALUE = [82, 82, 54] # Average values per channel from GWHD dataset

def get_gwhd_train_augmentations(img_width=640, img_height=640):
    return A.Compose(
        [
            A.Resize(width=img_width, height=img_height, interpolation=cv2.INTER_CUBIC, always_apply=True),
            A.Flip(p=0.5),
            A.OneOf([
                A.MotionBlur(p=0.9),
                A.GaussNoise(p=0.1),
            ], p=0.6),
            A.RandomShadow(p=0.5),
            A.RandomBrightnessContrast(p=0.5),
            A.FancyPCA(alpha=1),
        ],
        bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels'], min_area=0, min_visibility=0)
    )


def get_gwhd_test_augmentations(img_width=640, img_height=640):
    return A.Compose(
        [
            A.Resize(width=img_width, height=img_height, interpolation=cv2.INTER_CUBIC, always_apply=True),
        ],
        bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels'], min_area=0, min_visibility=0)
    )


def get_gwhd_val_augmentations(img_width=640, img_height=640):
    return A.Compose(
        [
            A.Resize(width=img_width, height=img_height, interpolation=cv2.INTER_CUBIC, always_apply=True),
        ],
        bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels'], min_area=0, min_visibility=0)
    )


if __name__ == '__main__':
    import argparse
    import yaml
    import os
    from data.adapter import GwhdToYoloAdapter, load_gwhd_df
    from yolov7.dataset import Yolov7Dataset
    from yolov7.plotting import show_image

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to config file')

    hyperparameters_config_file = parser.parse_args().config
    with open(hyperparameters_config_file) as f:
        config = yaml.safe_load(f)

    DATASET_ROOT_DIR = config['dataset']['path']
    train_subset = os.path.join(DATASET_ROOT_DIR, 'competition_train.csv')
    train_df = load_gwhd_df(os.path.join(DATASET_ROOT_DIR, train_subset))
    images_dir = os.path.join(config['dataset']['path'], config['dataset']['images_dir'])

    transforms = get_gwhd_train_augmentations()

    train_adapter = GwhdToYoloAdapter(images_dir, train_df, transforms)
    yolo_train_ds = Yolov7Dataset(train_adapter)

    for i in range(len(yolo_train_ds)):
        image_tensor, labels, image_id, image_size = yolo_train_ds[i]
        boxes = labels[:, 2:]
        boxes[:, [0, 2]] *= image_size[1]
        boxes[:, [1, 3]] *= image_size[0]

        show_image(image_tensor.permute(1, 2, 0), boxes.tolist(), None, 'cxcywh')