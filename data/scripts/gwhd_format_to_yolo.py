import argparse
import sys
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os

DATASET_ROOT_DIR = os.environ['DATASET_ROOT_DIR']

def create_gwhd_yolo_dir_tree():
    os.makedirs(f'{DATASET_ROOT_DIR}/yolo-format-dataset', exist_ok=True)
    os.makedirs(f'{DATASET_ROOT_DIR}/yolo-format-dataset/images/train', exist_ok=True)
    os.makedirs(f'{DATASET_ROOT_DIR}/yolo-format-dataset/images/test', exist_ok=True)
    os.makedirs(f'{DATASET_ROOT_DIR}/yolo-format-dataset/images/val', exist_ok=True)
    os.makedirs(f'{DATASET_ROOT_DIR}/yolo-format-dataset/labels/train', exist_ok=True)
    os.makedirs(f'{DATASET_ROOT_DIR}/yolo-format-dataset/labels/test', exist_ok=True)
    os.makedirs(f'{DATASET_ROOT_DIR}/yolo-format-dataset/labels/val', exist_ok=True)

def create_yolo_dir_tree(name):
    os.makedirs(f'{DATASET_ROOT_DIR}/yolo-format-dataset', exist_ok=True)
    os.makedirs(f'{DATASET_ROOT_DIR}/yolo-format-dataset/images/{name}', exist_ok=True)
    os.makedirs(f'{DATASET_ROOT_DIR}/yolo-format-dataset/labels/{name}', exist_ok=True)


def gwhd_coords_to_yolo_coords(
    x_min: int, y_min: int, x_max: int, y_max: int) -> tuple[float, float, float, float]:
    img_width = 1024
    img_height = 1024
    x_center = (x_min + x_max) / 2
    y_center = (y_min + y_max) / 2
    width = x_max - x_min
    height = y_max - y_min
    return x_center / img_width, y_center / img_height, width / img_width, height / img_height


def create_yolo_label_file(img_name: str, subset_name: str, bboxes: list[list[float]]):
    label_filename = img_name.split('.')[0] + '.txt'
    with open(f'{DATASET_ROOT_DIR}/yolo-format-dataset/labels/{subset_name}/{label_filename}', 'w'):
        for bbox in bboxes:
            with open(f'{DATASET_ROOT_DIR}/yolo-format-dataset/labels/{subset_name}/{label_filename}', 'a') as f:
                x_center, y_center, widht, height = bbox
                f.write(f'0 {x_center} {y_center} {widht} {height}\n')


def create_yolo_format_subset(df: pd.DataFrame, subset_name: str) -> None:
    with open(f'{DATASET_ROOT_DIR}/yolo-format-dataset/{subset_name}.txt', 'w') as f:
        for _, row in df.iterrows():
            img_name = row['image_name']
            if row['BoxesString'] == 'no_box':
                continue
            bboxes_gwhd = row['BoxesString'].split(';')
            bboxes_gwhd = [bbox.split(' ') for bbox in bboxes_gwhd]
            bboxes_yolo = [gwhd_coords_to_yolo_coords(*map(int, bbox)) for bbox in bboxes_gwhd]
            create_yolo_label_file(img_name, subset_name, bboxes_yolo)
            f.write(f'./images/{subset_name}/{img_name}\n')
            try:
                os.symlink(
                        f'{DATASET_ROOT_DIR}/images/{img_name}',
                        f'{DATASET_ROOT_DIR}/yolo-format-dataset/images/{subset_name}/{img_name}',
                       )
            except FileExistsError as e:
                print(f"Tried to create a duplicate symlink for {img_name}, skipping...")


def plot_bboxes(image_name: str, bboxes_yolo: list[list[float]]):
    img_width = 1024
    img_height = 1024
    img = plt.imread(f'{DATASET_ROOT_DIR}/images/{image_name}')
    fig, ax = plt.subplots(1)
    ax.imshow(img)
    for bbox in bboxes_yolo:
        x_center, y_center, width, height = bbox
        x_min = (x_center * img_width) - 0.5 * (width * img_width)
        y_min = (y_center * img_height) - 0.5 * (height * img_height)
        rect_width = width * img_width
        rect_height = height * img_height
        rect = patches.Rectangle((x_min, y_min), rect_width, rect_height, linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
    plt.show()


if __name__ == '__main__':
    #create_gwhd_yolo_dir_tree()

    #train_df = pd.read_csv(f'{DATASET_ROOT_DIR}/competition_train.csv').drop(columns=['domain'])
    #test_df = pd.read_csv(f'{DATASET_ROOT_DIR}/competition_test.csv').drop(columns=['domain'])
    #val_df = pd.read_csv(f'{DATASET_ROOT_DIR}/competition_val.csv').drop(columns=['domain'])

    #train_df.head()

    #create_yolo_format_subset(train_df, 'train')
    #create_yolo_format_subset(test_df, 'test')
    #create_yolo_format_subset(val_df, 'val')

    parser = argparse.ArgumentParser()
    parser.add_argument('--subset_name', type=str, required=True, help='Name of subset')
    parser.add_argument('--subset_descriptor_path', type=str, required=True, help='Path to descriptor file')

    cli_args = parser.parse_args()

    subset_df = pd.read_csv(cli_args.subset_descriptor_path)
    create_yolo_dir_tree(cli_args.subset_name)
    create_yolo_format_subset(subset_df, cli_args.subset_name)