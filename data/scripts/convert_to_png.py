import os
import sys
from PIL import Image
import tqdm

def convert_to_png(img_path):
    img = Image.open(img_path)
    img.save(img_path, format='PNG')

if __name__ == '__main__':
    images_dir = sys.argv[1]
    for img_name in tqdm.tqdm(os.listdir(images_dir)):
        try:
            convert_to_png(os.path.join(images_dir, img_name))
        except OSError:
            print(f"Couldn't convert {img_name}, skipping...")