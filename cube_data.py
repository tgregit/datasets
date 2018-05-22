import cv2
import numpy as np
import os


def get_all_data(my_path):
    fls = [os.path.join(my_path, f) for f in os.listdir(my_path) if os.path.isfile(os.path.join(my_path, f))]
    return fls


files = get_all_data('/home/foo/data/blend-new/dataset_cube/images')
print(files[66])