import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
import numpy as np
import cv2
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

image_path = "/home/lois/Documents/dnn/baseline/init.png"
model_path = "snapshots/Epoch_Final.pth"
output_path = "result_dnn.png"

# --- 3. CHARGEMENT ET INFERENCE ---
def run_inference(image_path, model_path, output_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = enhance_net_nopool().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    img_original = Image.open(image_path).convert('RGB')
    # Preprocessing 
    img_tensor = (np.asarray(img_original)/255.0)
    img_tensor = torch.from_numpy(img_tensor).float()
    img_tensor = img_tensor.permute(2,0,1).unsqueeze(0).to(device)
    
    with torch.no_grad():
        _, enhanced_image, _ = model(img_tensor)
    
    # Postprocessing
    result = enhanced_image.squeeze().cpu().permute(1, 2, 0).numpy()
    result = np.clip(result * 255, 0, 255).astype('uint8')
    img_enhanced = Image.fromarray(result)
    
    img_enhanced.save(output_path)
    print(f"Image saved at {output_path}")

if os.path.exists(image_path) and os.path.exists(model_path):
    run_inference(image_path, model_path, output_path)
else:
    print("not found")
