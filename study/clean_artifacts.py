import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def apply_sharpening(img, strength=0.5):
    gaussian_3 = cv2.GaussianBlur(img, (0, 0), 2.0)
    unsharp_image = cv2.addWeighted(img, 1.0 + strength, gaussian_3, -strength, 0)
    return unsharp_image

def clean_artifacts(image_path, h_strength=3):
    img = cv2.imread(image_path)
    
    dst_nlm = cv2.fastNlMeansDenoisingColored(img, None, h=h_strength, hColor=h_strength, templateWindowSize=7, searchWindowSize=21)
    dst_nlm_sharp = apply_sharpening(dst_nlm, strength=0.3)

    sharp_rgb = cv2.cvtColor(dst_nlm_sharp, cv2.COLOR_BGR2RGB)
    cv2.imwrite("cleaned_img.jpg", sharp_rgb)
    
    
target_img = "result_dnn.png" 

if not os.path.exists(target_img):
    import glob
    potential_files = glob.glob("data/test_data/lol_dataset/eval15/low/*.png")
    if potential_files:
        target_img = potential_files[0]
        print(f"Test sur : {target_img}")

clean_artifacts(target_img, h_strength=3)
