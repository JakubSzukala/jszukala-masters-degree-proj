import os

import pandas as pd
from PIL import Image, ImageTk
import tkinter as tk

DATASET_ROOT_DIR = os.environ['DATASET_ROOT_DIR']

df = pd.read_csv(os.path.join(DATASET_ROOT_DIR, 'competition_train.csv'))

# https://www.google.com/search?q=wheat+development+stages&client=ubuntu&hs=CbS&channel=fs&sxsrf=APwXEdcYJctW-MYx1B7dYA0eNzi3fdZRXw:1685390467716&source=lnms&tbm=isch&sa=X&ved=2ahUKEwiU_IrhqJv_AhVUkMMKHesOAboQ_AUoAXoECAEQAw&biw=2132&bih=1108&dpr=1.2#imgrc=k_ikEsGrzQE6QM

class WheatHeadStageLabeler:
    def __init__(self, source, output, root_dir):
        self.root_dir = root_dir
        self.source = source
        self.output = output
        with open(source, 'r') as f:
            self.source_lines = f.readlines()
        with open(output, 'r') as f:
            self.output_lines = f.readlines()

        if len(self.output_lines) == 0:
            self.output_lines = [''] * len(self.source_lines)
            self.output_lines[0] = self.source_lines[0].replace('\n', ',stage\n') # Add header
            self.current_idx = 1 # Skip header
        else: # Pick up where we left off
            # Fill to the desired lenght
            self.current_idx = len(self.output_lines)
            self.output_lines += [''] * (len(self.source_lines) - len(self.output_lines))


        # Create tkinter window
        self.window = tk.Tk()
        self.window.title('Wheat Head Labeler')
        self.window.geometry('1024x1024')
        self.photo = None
        self.image = None
        self.image_label = None

        # Create buttons
        heading_option_btn = tk.Button(self.window, text='Heading stage', command=self.heading_callback)
        flowering_option_btn = tk.Button(self.window, text='Flowering / Grainfilling stage', command=self.flowering_callback)
        ripening_option_btn = tk.Button(self.window, text='Ripening stage', command=self.ripening_callback)
        heading_option_btn.pack()
        flowering_option_btn.pack()
        ripening_option_btn.pack()


    def display_image(self, image_path):
        self.image = Image.open(image_path)
        self.photo = ImageTk.PhotoImage(self.image)
        if hasattr(self, 'image_label') and self.image_label is not None:
            self.image_label.destroy()
        self.image_label = tk.Label(self.window, image=self.photo)
        self.image_label.pack()


    def start_prompt(self):
        image_name = self.source_lines[self.current_idx].split(',')[0]
        initial_image = os.path.join(self.root_dir, 'images', image_name)
        print('You are labeling image: ', initial_image)
        self.display_image(initial_image)
        self.window.mainloop()


    def heading_callback(self):
        print('You choosed heading stage')
        print(f"Annotated {self.current_idx} images out of {len(self.source_lines) - 1}")
        self.output_lines[self.current_idx] = self.source_lines[self.current_idx].replace('\n', ',heading\n')

        with open(self.output, 'w') as f:
            f.writelines(self.output_lines)

        self.current_idx += 1
        if self.current_idx < len(self.output_lines):
            next_image = os.path.join(self.root_dir, 'images', self.source_lines[self.current_idx].split(',')[0])
            print(self.current_idx)
            print('You are labeling image: ', next_image)
            self.display_image(next_image)
        else:
            self.window.quit()


    def flowering_callback(self):
        print('You choosed flowering / grainfilling stage')
        print(f"Annotated {self.current_idx} images out of {len(self.source_lines) - 1}")
        self.output_lines[self.current_idx] = self.source_lines[self.current_idx].replace('\n', ',flowering/grainfilling\n')

        with open(self.output, 'w') as f:
            f.writelines(self.output_lines)

        self.current_idx += 1
        if self.current_idx < len(self.output_lines):
            next_image = os.path.join(self.root_dir, 'images', self.source_lines[self.current_idx].split(',')[0])
            self.display_image(next_image)
        else:
            self.window.quit()


    def ripening_callback(self):
        print('You choosed ripening stage')
        print(f"Annotated {self.current_idx} images out of {len(self.source_lines) - 1}")
        self.output_lines[self.current_idx] = self.source_lines[self.current_idx].replace('\n', ',ripening\n')

        with open(self.output, 'w') as f:
            f.writelines(self.output_lines)

        self.current_idx += 1
        if self.current_idx < len(self.output_lines):
            next_image = os.path.join(self.root_dir, 'images', self.source_lines[self.current_idx].split(',')[0])
            self.display_image(next_image)
        else:
            self.window.quit()


labeler = WheatHeadStageLabeler(os.path.join(DATASET_ROOT_DIR, 'competition_train.csv'), 'competition_train_with_stages.csv', DATASET_ROOT_DIR)
labeler.start_prompt()
#for row in df.iterrows():
    #img_path = os.path.join(DATASET_ROOT_DIR, 'images', row[1]['image_name'])
    #labeler.start_prompt()
    #break