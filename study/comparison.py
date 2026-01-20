import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from skimage.measure import shannon_entropy
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from PIL import Image
import glob
import random
import pandas as pd
import math
import os

class enhance_net_nopool(nn.Module):
    def __init__(self):
        super(enhance_net_nopool, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        number_f = 32
        self.e_conv1 = nn.Conv2d(3,number_f,3,1,1,bias=True) 
        self.e_conv2 = nn.Conv2d(number_f,number_f,3,1,1,bias=True) 
        self.e_conv3 = nn.Conv2d(number_f,number_f,3,1,1,bias=True) 
        self.e_conv4 = nn.Conv2d(number_f,number_f,3,1,1,bias=True) 
        self.e_conv5 = nn.Conv2d(number_f*2,number_f,3,1,1,bias=True) 
        self.e_conv6 = nn.Conv2d(number_f*2,number_f,3,1,1,bias=True) 
        self.e_conv7 = nn.Conv2d(number_f*2,24,3,1,1,bias=True) 
        self.maxpool = nn.MaxPool2d(2, stride=2, return_indices=False, ceil_mode=False)
        self.upsample = nn.UpsamplingBilinear2d(scale_factor=2)

    def forward(self, x):
        x1 = self.relu(self.e_conv1(x))
        x2 = self.relu(self.e_conv2(x1))
        x3 = self.relu(self.e_conv3(x2))
        x4 = self.relu(self.e_conv4(x3))
        x5 = self.relu(self.e_conv5(torch.cat([x3,x4],1)))
        x6 = self.relu(self.e_conv6(torch.cat([x2,x5],1)))
        x_r = F.tanh(self.e_conv7(torch.cat([x1,x6],1)))
        r1,r2,r3,r4,r5,r6,r7,r8 = torch.split(x_r, 3, dim=1)
        x = x + r1*(torch.pow(x,2)-x)
        x = x + r2*(torch.pow(x,2)-x)
        x = x + r3*(torch.pow(x,2)-x)
        enhance_image_1 = x + r4*(torch.pow(x,2)-x)
        x = enhance_image_1 + r5*(torch.pow(enhance_image_1,2)-enhance_image_1)
        x = x + r6*(torch.pow(x,2)-x)
        x = x + r7*(torch.pow(x,2)-x)
        enhance_image = x + r8*(torch.pow(x,2)-x)
        r = torch.cat([r1,r2,r3,r4,r5,r6,r7,r8],1)
        return enhance_image_1,enhance_image,r
    
def open_image(img_path) :
    img = cv2.imread(img_path)
    if img is None:
        print(f"Couldn't find image at : {img_path}")
        return None
    return img

def apply_clahe(img_bgr):
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    l_clahe = clahe.apply(l)
    lab_clahe = cv2.merge((l_clahe, a, b))
    return cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)

def apply_autogamma(img_bgr):
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    mean_brightness = np.mean(v)
    target = 128
    
    if mean_brightness < 1e-6:
        return img_bgr
        
    gamma = math.log(target / 255.0) / math.log((mean_brightness) / 255.0)
    
    table = np.array([((i / 255.0) ** gamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(img_bgr, table)

def get_metrics(img_bgr):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    entropy = shannon_entropy(gray)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    grad_mag = np.mean(np.sqrt(sobelx**2 + sobely**2))
    return entropy, grad_mag

def run_full_comparison(model, img_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    img_org = open_image(img_path)
    if img_org is None:
        return
    img_org_rgb = cv2.cvtColor(img_org, cv2.COLOR_BGR2RGB)
    
    img_gamma_bgr = apply_autogamma(img_org)
    img_gamma_rgb = cv2.cvtColor(img_gamma_bgr, cv2.COLOR_BGR2RGB)

    img_clahe_bgr = apply_clahe(img_org)
    img_clahe_rgb = cv2.cvtColor(img_clahe_bgr, cv2.COLOR_BGR2RGB)
    
    data_lowlight = Image.open(img_path).convert('RGB')
    data_lowlight = (np.asarray(data_lowlight)/255.0)
    data_lowlight = torch.from_numpy(data_lowlight).float().permute(2,0,1).unsqueeze(0).to(device)
    
    model.eval()
    with torch.no_grad():
        _, enhanced_image, _ = model(data_lowlight)
    res_deep = enhanced_image.squeeze().cpu().permute(1, 2, 0).numpy()
    res_deep = np.clip(res_deep * 255, 0, 255).astype('uint8')
    img_deep_bgr = cv2.cvtColor(res_deep, cv2.COLOR_RGB2BGR)
    img_deep_rgb = res_deep

    ent_org, grad_org = get_metrics(img_org)
    ent_gamma, grad_gamma = get_metrics(img_gamma_bgr)
    ent_clahe, grad_clahe = get_metrics(img_clahe_bgr)
    ent_deep, grad_deep = get_metrics(img_deep_bgr)

    results = {
        "Algo": ["Original", "Gamma", "CLAHE", "Zero-DCE"],
        "Entropie": [ent_org, ent_gamma, ent_clahe, ent_deep],
        "Gradient": [grad_org, grad_gamma, grad_clahe, grad_deep]
    }
    df = pd.DataFrame(results)
    print(df.round(3).to_string(index=False))

    fig = plt.figure(figsize=(16, 10))
    
    ax1 = fig.add_subplot(2, 4, 1)
    ax1.imshow(img_org_rgb)
    ax1.set_title("Original")
    
    ax2 = fig.add_subplot(2, 4, 2)
    ax2.imshow(img_gamma_rgb)
    ax2.set_title("Gamma")

    ax3 = fig.add_subplot(2, 4, 3)
    ax3.imshow(img_clahe_rgb)
    ax3.set_title("CLAHE")

    ax4 = fig.add_subplot(2, 4, 4)
    ax4.imshow(img_deep_rgb)
    ax4.set_title("Zero-DCE")

    labels = ("Original", "Gamma", "CLAHE", "Zero-DCE")
    imgs_gray = [
        cv2.cvtColor(img_org, cv2.COLOR_BGR2GRAY),
        cv2.cvtColor(img_gamma_bgr, cv2.COLOR_BGR2GRAY),
        cv2.cvtColor(img_clahe_bgr, cv2.COLOR_BGR2GRAY),
        cv2.cvtColor(img_deep_bgr, cv2.COLOR_BGR2GRAY)
    ]
    
    ax_hist = fig.add_subplot(2, 1, 2)
    for i, img_g in enumerate(imgs_gray):
        hist = cv2.calcHist([img_g], [0], None, [256], [0, 256])
        ax_hist.plot(hist, label=labels[i])
        
    ax_hist.set_title("Histogram comparison")
    ax_hist.set_xlabel("Luminosity")
    ax_hist.legend()
    ax_hist.grid(alpha=0.3)

    plt.tight_layout()
    plt.show()
    
def process_all_images(model, image_dir):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    img_paths = glob.glob(os.path.join(image_dir, "*.png"))
    
    if not img_paths:
        print(f"No images found in {image_dir}")
        return
    
    print(f"Processing {len(img_paths)} images from {image_dir}\n")
    
    total_metrics = {
        "Original": {"entropy": 0, "gradient": 0, "psnr": 0, "ssim": 0, "mae": 0},
        "Gamma": {"entropy": 0, "gradient": 0, "psnr": 0, "ssim": 0, "mae": 0},
        "CLAHE": {"entropy": 0, "gradient": 0, "psnr": 0, "ssim": 0, "mae": 0},
        "Zero-DCE": {"entropy": 0, "gradient": 0, "psnr": 0, "ssim": 0, "mae": 0}
    }
    
    hist_accumulators = {
        "Original": np.zeros(256),
        "Gamma": np.zeros(256),
        "CLAHE": np.zeros(256),
        "Zero-DCE": np.zeros(256)
    }
    
    high_dir = image_dir.replace("/low", "/high")
    if not os.path.exists(high_dir):
        high_dir = None
    
    for idx, img_path in enumerate(sorted(img_paths), 1):
        print(f"Processing image {idx}/{len(img_paths)}: {img_path}")
        
        img_org = open_image(img_path)
        if img_org is None:
            continue
        
        img_ref = None
        if high_dir:
            ref_path = os.path.join(high_dir, os.path.basename(img_path))
            if os.path.exists(ref_path):
                img_ref = open_image(ref_path)
                img_ref_rgb = cv2.cvtColor(img_ref, cv2.COLOR_BGR2RGB) if img_ref is not None else None
        
        img_gamma_bgr = apply_autogamma(img_org)
        
        img_clahe_bgr = apply_clahe(img_org)
        
        data_lowlight = Image.open(img_path).convert('RGB')
        data_lowlight = (np.asarray(data_lowlight)/255.0)
        data_lowlight = torch.from_numpy(data_lowlight).float().permute(2,0,1).unsqueeze(0).to(device)
        
        model.eval()
        with torch.no_grad():
            _, enhanced_image, _ = model(data_lowlight)
        res_deep = enhanced_image.squeeze().cpu().permute(1, 2, 0).numpy()
        res_deep = np.clip(res_deep * 255, 0, 255).astype('uint8')
        img_deep_bgr = cv2.cvtColor(res_deep, cv2.COLOR_RGB2BGR)
        
        ent_org, grad_org = get_metrics(img_org)
        ent_gamma, grad_gamma = get_metrics(img_gamma_bgr)
        ent_clahe, grad_clahe = get_metrics(img_clahe_bgr)
        ent_deep, grad_deep = get_metrics(img_deep_bgr)
        
        total_metrics["Original"]["entropy"] += ent_org
        total_metrics["Original"]["gradient"] += grad_org
        total_metrics["Gamma"]["entropy"] += ent_gamma
        total_metrics["Gamma"]["gradient"] += grad_gamma
        total_metrics["CLAHE"]["entropy"] += ent_clahe
        total_metrics["CLAHE"]["gradient"] += grad_clahe
        total_metrics["Zero-DCE"]["entropy"] += ent_deep
        total_metrics["Zero-DCE"]["gradient"] += grad_deep
        
        gray_org = cv2.cvtColor(img_org, cv2.COLOR_BGR2GRAY)
        gray_gamma = cv2.cvtColor(img_gamma_bgr, cv2.COLOR_BGR2GRAY)
        gray_clahe = cv2.cvtColor(img_clahe_bgr, cv2.COLOR_BGR2GRAY)
        gray_deep = cv2.cvtColor(img_deep_bgr, cv2.COLOR_BGR2GRAY)
        
        hist_accumulators["Original"] += cv2.calcHist([gray_org], [0], None, [256], [0, 256]).flatten()
        hist_accumulators["Gamma"] += cv2.calcHist([gray_gamma], [0], None, [256], [0, 256]).flatten()
        hist_accumulators["CLAHE"] += cv2.calcHist([gray_clahe], [0], None, [256], [0, 256]).flatten()
        hist_accumulators["Zero-DCE"] += cv2.calcHist([gray_deep], [0], None, [256], [0, 256]).flatten()
        
        if img_ref_rgb is not None:
            img_org_rgb = cv2.cvtColor(img_org, cv2.COLOR_BGR2RGB)
            img_gamma_rgb = cv2.cvtColor(img_gamma_bgr, cv2.COLOR_BGR2RGB)
            img_clahe_rgb = cv2.cvtColor(img_clahe_bgr, cv2.COLOR_BGR2RGB)
            img_deep_rgb = res_deep
            
            total_metrics["Original"]["psnr"] += psnr(img_ref_rgb, img_org_rgb)
            total_metrics["Original"]["ssim"] += ssim(img_ref_rgb, img_org_rgb, channel_axis=2)
            total_metrics["Original"]["mae"] += np.mean(np.abs(img_ref_rgb.astype(float) - img_org_rgb.astype(float)))
            
            total_metrics["Gamma"]["psnr"] += psnr(img_ref_rgb, img_gamma_rgb)
            total_metrics["Gamma"]["ssim"] += ssim(img_ref_rgb, img_gamma_rgb, channel_axis=2)
            total_metrics["Gamma"]["mae"] += np.mean(np.abs(img_ref_rgb.astype(float) - img_gamma_rgb.astype(float)))
            
            total_metrics["CLAHE"]["psnr"] += psnr(img_ref_rgb, img_clahe_rgb)
            total_metrics["CLAHE"]["ssim"] += ssim(img_ref_rgb, img_clahe_rgb, channel_axis=2)
            total_metrics["CLAHE"]["mae"] += np.mean(np.abs(img_ref_rgb.astype(float) - img_clahe_rgb.astype(float)))
            
            total_metrics["Zero-DCE"]["psnr"] += psnr(img_ref_rgb, img_deep_rgb)
            total_metrics["Zero-DCE"]["ssim"] += ssim(img_ref_rgb, img_deep_rgb, channel_axis=2)
            total_metrics["Zero-DCE"]["mae"] += np.mean(np.abs(img_ref_rgb.astype(float) - img_deep_rgb.astype(float)))
    
    num_images = len(img_paths)
    
    results = {
        "Method": ["Original", "Gamma", "CLAHE", "Zero-DCE"],
        "Total Entropy": [
            total_metrics["Original"]["entropy"],
            total_metrics["Gamma"]["entropy"],
            total_metrics["CLAHE"]["entropy"],
            total_metrics["Zero-DCE"]["entropy"]
        ],
        "Total Gradient": [
            total_metrics["Original"]["gradient"],
            total_metrics["Gamma"]["gradient"],
            total_metrics["CLAHE"]["gradient"],
            total_metrics["Zero-DCE"]["gradient"]
        ],
        "Mean PSNR": [
            total_metrics["Original"]["psnr"] / num_images if high_dir else 0,
            total_metrics["Gamma"]["psnr"] / num_images if high_dir else 0,
            total_metrics["CLAHE"]["psnr"] / num_images if high_dir else 0,
            total_metrics["Zero-DCE"]["psnr"] / num_images if high_dir else 0
        ],
        "Mean SSIM": [
            total_metrics["Original"]["ssim"] / num_images if high_dir else 0,
            total_metrics["Gamma"]["ssim"] / num_images if high_dir else 0,
            total_metrics["CLAHE"]["ssim"] / num_images if high_dir else 0,
            total_metrics["Zero-DCE"]["ssim"] / num_images if high_dir else 0
        ],
        "Mean MAE": [
            total_metrics["Original"]["mae"] / num_images if high_dir else 0,
            total_metrics["Gamma"]["mae"] / num_images if high_dir else 0,
            total_metrics["CLAHE"]["mae"] / num_images if high_dir else 0,
            total_metrics["Zero-DCE"]["mae"] / num_images if high_dir else 0
        ]
    }
    df = pd.DataFrame(results)
    print(df.round(3).to_string(index=False))
    print()
    
    plt.figure(figsize=(12, 6))
    
    labels = ["Original", "Gamma", "CLAHE", "Zero-DCE"]
    colors = ['blue', 'orange', 'green', 'red']
    
    for label, color in zip(labels, colors):
        plt.plot(hist_accumulators[label], label=label, color=color, alpha=0.7)
    
    plt.title(f"Accumulated Histogram Comparison ({len(img_paths)} images)")
    plt.xlabel("Luminosity")
    plt.ylabel("Frequency")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    DCE_net = enhance_net_nopool().to(device)
    model_path = "snapshots/exp4/Epoch_Final.pth"
    image_dir = "data/test_data/lol_dataset/eval15/low"

    DCE_net.load_state_dict(torch.load(model_path, map_location=device))
    process_all_images(DCE_net, image_dir)