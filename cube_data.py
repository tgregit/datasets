import cv2
import numpy as np
import os


def get_all_data(my_path, my_image_width):
    filenames = [os.path.join(my_path, f) for f in os.listdir(my_path) if os.path.isfile(os.path.join(my_path, f))]
    filenames.sort()


    total_samples = len(filenames)
    #print(total_files)

    x_input = np.zeros((total_samples, 4), dtype=np.float)
    # we have 4 inputs: x_loc, z_loc, local z_rotation, and light strength -- these are already normalized 0 - 1.0

    y_images = np.zeros((total_samples, my_image_width, my_image_width), dtype=np.float)
    for index, f in enumerate(filenames):
        img = cv2.imread(f, 0)
        img = img / 128.0
        y_images[index] = img

        f = f[0:-4]  # delete .jpg
        all = f.split('_')
        just_floats = np.array(all[2:6])
        x_input[index] = just_floats
        #print(just_floats)



    return x_input, y_images




x, y = get_all_data('/home/foo/data/blend-new/dataset_cube/images', 32)
print(y.shape)
#print(files[66])