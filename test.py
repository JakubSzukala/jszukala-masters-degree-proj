import os

import yaml
import argparse
from data.adapter import GwhdToYoloAdapter, load_gwhd_df
import torch

from yolov7.dataset import Yolov7Dataset
from yolov7.plotting import show_image
from yolov7.trainer import filter_eval_predictions
from yolov7.models.yolo import Yolov7Model
from yolov7 import create_yolov7_model

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, required=True, help='Path to config file')

hyperparameters_config_file = parser.parse_args().config
with open(hyperparameters_config_file) as f:
    config = yaml.safe_load(f)

model = create_yolov7_model('yolov7-tiny', num_classes=1, pretrained=False)
state_dict = torch.load('best_model.pt')
model.load_state_dict(state_dict=state_dict['model_state_dict'])
model.eval()

DATASET_ROOT_DIR = config['dataset']['path']
test_subset = os.path.join(config['dataset']['path'], config['dataset']['subsets']['test'])
test_df = load_gwhd_df(os.path.join(DATASET_ROOT_DIR, test_subset))
images_dir = os.path.join(config['dataset']['path'], config['dataset']['images_dir'])
test_adapter = GwhdToYoloAdapter(images_dir, test_df, None)
yolo_test_ds = Yolov7Dataset(test_adapter)

test_df = load_gwhd_df(os.path.join(DATASET_ROOT_DIR, test_subset))

for i in range(len(yolo_test_ds)):
    image_tensor, labels, image_id, image_size = yolo_test_ds[i]

    confidence_threshold = config['trainer_params']['filter_eval_predictions_fn_params']['confidence_threshold']
    with torch.no_grad():
        outputs = model(image_tensor[None])
        # Confidence threshold has identical effect in postprocess and filter_eval_predictions but different defaults
        preds = model.postprocess(outputs, conf_thres=confidence_threshold, multiple_labels_per_box=False)
        nms_preds = filter_eval_predictions(preds, confidence_threshold=confidence_threshold)

    boxes = nms_preds[0][:, :4]
    class_ids = nms_preds[0][:, -1]

    show_image(image_tensor.permute(1, 2, 0), boxes.tolist(), None)