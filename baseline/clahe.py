import cv2

image_path = '/home/lois/Documents/dnn/data/train_data/22.jpg'
clip_limit = 3.0
grid_size = (8, 8)

def open_image(img_path) :
    img = cv2.imread(img_path)
    if img is None:
        print(f"Couldn't find image at : {img_path}")
        return None
    return img

def apply_clahe(img):
    cv2.imwrite("init.png", img)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB) 
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size)
    l_new = clahe.apply(l)
    lab_new = cv2.merge((l_new, a, b))
    result = cv2.cvtColor(lab_new, cv2.COLOR_LAB2BGR)
    cv2.imwrite('result_clahe.png', result)

if __name__ == "__main__":
    img = open_image(image_path)
    apply_clahe(img)
