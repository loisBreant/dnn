import cv2
import math
import numpy as np

image_path = '/home/lois/Documents/dnn/data/train_data/22.jpg'
gamma_value = 0.8 

def open_image(img_path) :
    img = cv2.imread(img_path)
    if img is None:
        print(f"Couldn't find image at : {img_path}")
        return None
    return img

def gamma_correction(img, gamma=1.0):
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")

    result = cv2.LUT(img, table)

    cv2.imwrite('result.png', result)


def gamma_autocorrection(img):
    cv2.imwrite("init.png", img)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv) # hue, saturation, value 

    mean_brightness = np.mean(v)

    target = 128  # mid-gray
    # want to find (0.5 = mean_brightness ^ A )
    # gamma = log(target) / log(current)  (en normalisant sur 0-1)
    gamma_val = math.log(target / 255.0) / math.log((mean_brightness + 1e-6) / 255.0)
    print(f"gamma_val: {gamma_value}")
    gamma_correction(img, gamma_val)


if __name__ == "__main__":
    img = open_image(image_path)
    gamma_autocorrection(img)
    # gamma_correction(img, gamma_value)
