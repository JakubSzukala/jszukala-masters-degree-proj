import os

from matplotlib import patches, pyplot as plt
import PIL
import argparse
import numpy as np
import torch

from yolov7.plotting import show_image
from yolov7.trainer import filter_eval_predictions
from yolov7 import create_yolov7_model

from model.augmentations import get_gwhd_test_augmentations

from torchvision.transforms import ToTensor


def draw_bboxes(img, bboxes, conf_scores):
    fig, ax = plt.subplots(1)
    ax.imshow(img)
    for bbox, conf_score in zip(bboxes, conf_scores):
        x_min, y_min, x_max, y_max = map(int, bbox)
        rect1 = patches.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min, linewidth=4, edgecolor='w', fill=False)
        rect2 = patches.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min, linewidth=2, edgecolor='k', fill=False)
        ax.add_patch(rect1)
        ax.add_patch(rect2)
        ax.text(x_min, y_min, round(float(conf_score.cpu()), 2), color='k', backgroundcolor='w', fontsize=8)
    plt.show()


def infer(**kwargs):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = create_yolov7_model(kwargs['model'] , num_classes=1, pretrained=False)
    state_dict = torch.load(kwargs['weights'])
    model.load_state_dict(state_dict=state_dict['model_state_dict'])
    model.to(device).eval()

    transform = get_gwhd_test_augmentations(img_height=kwargs['image_size'], img_width=kwargs['image_size'])
    image = np.array(PIL.Image.open(kwargs['image']))
    image = transform(image=image, bboxes=np.array([]), labels=np.array([]))['image']
    image_tensor = ToTensor()(image).to(device)
    confidence_threshold = kwargs['confidence_th']

    with torch.no_grad():
        outputs = model(image_tensor[None])
        preds = model.postprocess(outputs, conf_thres=confidence_threshold, multiple_labels_per_box=False)
        nms_preds = filter_eval_predictions(preds, confidence_threshold=kwargs['confidence_th'], nms_threshold=kwargs['nms_th'])

    boxes = nms_preds[0][:, :4]
    scores = nms_preds[0][:, 4]

    return image_tensor, boxes, scores


class WheatHeadDetector:
    def __init__(
        self,
        model_name,
        state_dict,
        img_size,
        device
    ):
        self.device = device
        self.model = create_yolov7_model(model_name, num_channels=1, pretrained=False)
        self.model.load_state_dict(state_dict=state_dict['model_state_dict'])
        self.model.to(self.device).eval()
        self.transform = get_gwhd_test_augmentations(img_size, img_size)

    def detect(self, image, conf_th, nms_th):
        image = self.transform(image=image, bboxes=np.array([]), labels=np.array([]))['image']
        image_tensor = ToTensor()(image).to(self.device)

        with torch.no_grad():
            outputs = self.model(image_tensor[None])
            preds = self.model.postprocess(outputs, conf_thres=conf_th, multiple_labels_per_box=False)
            nms_preds = filter_eval_predictions(preds, confidence_threshold=conf_th, nms_threshold=nms_th)

        boxes = nms_preds[0][:, :4]
        scores = nms_preds[0][:, 4]

        return boxes, scores


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, help='Yolov7 model name')
    parser.add_argument('--weights', type=str, required=True, help='Path to weights .pt file')
    parser.add_argument('--image', type=str, required=True, help='Path to image file')
    parser.add_argument('--nms_th', type=float, required=False, default=0.45, help='NMS threshold')
    parser.add_argument('--confidence_th', type=float, required=False, default=0.15, help='Confidence threshold for predictions')
    parser.add_argument('--image_size', type=int, required=False, default=640, help='Image size')

    args = parser.parse_args()
    image_tensor, boxes, scores = infer(**vars(args))
    draw_bboxes(image_tensor.permute(1, 2, 0).cpu(), boxes, scores.cpu())
    #show_image(image_tensor.permute(1, 2, 0).cpu(), boxes.tolist(), None)