import os
import PIL
import argparse
import torch

from yolov7.plotting import show_image
from yolov7.trainer import filter_eval_predictions
from yolov7 import create_yolov7_model

from torchvision.transforms import ToTensor

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, required=True, help='Yolov7 model name')
parser.add_argument('--weights', type=str, required=True, help='Path to weights .pt file')
parser.add_argument('--image', type=str, required=True, help='Path to image file')
parser.add_argument('--nms_th', type=float, required=False, default=0.45, help='NMS threshold')
parser.add_argument('--confidence_th', type=float, required=False, default=0.15, help='Confidence threshold for predictions')

model_name = parser.parse_args().model
model = create_yolov7_model(model_name, num_classes=1, pretrained=False)
state_dict = torch.load(parser.parse_args().weights)
model.load_state_dict(state_dict=state_dict['model_state_dict'])
model.eval()

transform = ToTensor()
image_tensor = transform(PIL.Image.open(parser.parse_args().image))
confidence_threshold = parser.parse_args().confidence_th
with torch.no_grad():
    outputs = model(image_tensor[None])
    preds = model.postprocess(outputs, conf_thres=confidence_threshold, multiple_labels_per_box=False)
    nms_preds = filter_eval_predictions(preds, confidence_threshold=parser.parse_args().nms_th)

boxes = nms_preds[0][:, :4]
class_ids = nms_preds[0][:, -1]

show_image(image_tensor.permute(1, 2, 0), boxes.tolist(), None)