from io import BytesIO
import os
from matplotlib import pyplot as plt
import matplotlib
from easyimages import EasyImage, EasyImageList
from easyimages.utils import fig2img_v2
import pandas as pd
from PIL import Image
import PIL
from pathlib import Path
import base64

DATASET_ROOT_DIR = os.environ['DATASET_ROOT_DIR']
from easyimages.utils import get_execution_context
print(get_execution_context())

grid_template = """<div class="zoom"><img style='width: {size}px; height: {size}px; margin: 1px; float: left; border: 0px solid black;'title={label} src="data:image/png;base64, {url}"/></div>"""

val_df = pd.read_csv(os.path.join(DATASET_ROOT_DIR, 'competition_val.csv'))
reduced_val_df = val_df.iloc[:2]
ez_imgs = []
templates = []
for row in reduced_val_df.iterrows():
    img_path = os.path.join(DATASET_ROOT_DIR, 'images', row[1]['image_name'])

    # Parse bboxes
    boxes = row[1]['BoxesString'].split(';')
    boxes = [box.split(' ') for box in boxes]
    new_boxes = []
    for box in boxes:
        x_min, y_min, x_max, y_max = map(int, box)
        box = [x_min, y_min, x_max, y_max]
        box.append(1)
        box.append('wheat-head')
        new_boxes.append(box)

    # Create EasyImage with bboxes
    img = EasyImage.from_file(img_path, boxes=new_boxes) # TODO: creation of EasyImage could be skipped
    fig = img.draw_boxes()
    pil_img = fig2img_v2(fig)
    buff = BytesIO()
    pil_img.convert('RGB').save(buff, format="JPEG")
    base64_img = base64.b64encode(buff.getvalue()).decode("utf-8")
    templates.append(grid_template.format(url=base64_img, label='wheat-head', size=1024))
html = ''.join(templates)
print(len(templates))
with open('test.html', 'w') as f:
    f.write(html)