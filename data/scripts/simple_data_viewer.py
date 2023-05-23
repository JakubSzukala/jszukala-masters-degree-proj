from io import BytesIO
import os
from matplotlib import pyplot as plt
import matplotlib
from easyimages import EasyImage, EasyImageList
from easyimages.utils import fig2img_v2
import pandas as pd
from PIL import Image, ImageDraw
import PIL
from pathlib import Path
import base64

DATASET_ROOT_DIR = os.environ['DATASET_ROOT_DIR']
from easyimages.utils import get_execution_context
print(get_execution_context())

grid_template = """<div class="zoom"><img style='width: {size}px; height: {size}px; margin: 1px; float: left; border: 0px solid black;'title={label} src="data:image/png;base64, {url}"/><p>{caption}</p></div>"""

val_df = pd.read_csv(os.path.join(DATASET_ROOT_DIR, 'competition_val.csv'))
reduced_val_df = val_df.iloc[:2]
ez_imgs = []
templates = []
for row in reduced_val_df.iterrows():
    img_path = os.path.join(DATASET_ROOT_DIR, 'images', row[1]['image_name'])

    # Load image
    fig, ax = plt.subplots(1)
    ax.imshow(plt.imread(img_path))

    # Draw bboxes
    boxes = row[1]['BoxesString'].split(';')
    boxes = [box.split(' ') for box in boxes]
    for box in boxes:
        x_min, y_min, x_max, y_max = map(int, box)
        width = x_max - x_min
        height = y_max - y_min
        rect = matplotlib.patches.Rectangle((x_min, y_min), width, height, fill=False, edgecolor='red', linewidth=1)
        ax.add_patch(rect)

    # Convert plt to PIL
    pil_img = fig2img_v2(fig)
    pil_img.show()
    buff = BytesIO()
    pil_img.convert('RGB').save(buff, format="JPEG", quality=100)
    base64_img = base64.b64encode(buff.getvalue()).decode("utf-8")
    templates.append(grid_template.format(url=base64_img, label='wheat-head', size=1024, caption=row[1]['image_name']))
html = ''.join(templates)
print(len(templates))
with open('test.html', 'w') as f:
    f.write(html)