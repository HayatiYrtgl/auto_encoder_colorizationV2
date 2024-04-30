from keras.preprocessing.image import load_img, img_to_array
import numpy as np
import glob


def process(path, target_size):
    """this method process the data"""
    x, y = [], []

    for image in glob.glob(path):
        img_color = load_img(image, target_size=target_size, color_mode="rgb")
        img_gray = load_img(image, target_size=target_size, color_mode="grayscale")

        img_color = img_to_array(img_color)
        img_gray = img_to_array(img_gray)

        x.append(img_gray/255)
        y.append(img_color/255)

    x = np.array(x)
    y = np.array(y)

    np.save("../DATASET/land/gray_x.npy", x)
    np.save("../DATASET/land/color_x.npy", y)


process(path="../DATASET/land/landscape_Images/color/*.*", target_size=(128, 128))

