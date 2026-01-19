import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
import cv2
import os
import glob
import random

# Définition du modèle pour être autonome
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

def load_model(path, device): 
    print(f"Chargement du modèle : {path}")
    if not os.path.exists(path):
        print(f"❌ Erreur : Le fichier {path} n'existe pas.")
        return None
        
    model = enhance_net_nopool().to(device)
    # map_location est crucial pour éviter l'erreur CUDA sur CPU
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    return model

def ensemble_inference(image_path, model_a, model_b, device):
    if model_a is None or model_b is None:
        print("❌ Impossible de lancer l'inférence : un des modèles n'est pas chargé.")
        return

    if not os.path.exists(image_path):
        print(f"❌ Image introuvable : {image_path}")
        return

    print(f"Traitement de l'image : {image_path}")
    
    # Préparation
    data_lowlight = Image.open(image_path).convert('RGB')
    data_lowlight = (np.asarray(data_lowlight)/255.0)
    img_tensor = torch.from_numpy(data_lowlight).float().permute(2,0,1).unsqueeze(0).to(device)

    with torch.no_grad():
        # On demande l'avis aux 2 experts
        _, out_A, _ = model_a(img_tensor)
        _, out_B, _ = model_b(img_tensor)
        
        # LA MAGIE : On fait la moyenne des tenseurs
        out_final = (out_A + out_B ) / 2.0

    # Conversion en image
    result = out_final.squeeze().cpu().permute(1, 2, 0).numpy()
    result = np.clip(result * 255, 0, 255).astype('uint8')
    
    # Petit nettoyage final (très léger)
    result = cv2.fastNlMeansDenoisingColored(result, None, h=3, hColor=3, templateWindowSize=7, searchWindowSize=11)
    
    # Conversion RGB -> BGR pour OpenCV
    result_bgr = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
    
    output_name = "Resultat_Ensemble_Final.jpg"
    cv2.imwrite(output_name, result_bgr)
    print(f"✅ Image fusionnée sauvegardée : {output_name}")

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Utilisation du device : {device}")

    # Chemins relatifs depuis la racine du projet
    path_A = "../snapshots/exp4/Epoch_Final.pth"
    path_B = "../snapshots/exp3/Epoch_Final.pth"

    model_A = load_model(path_A, device)
    model_B = load_model(path_B, device)

    # Trouver une image de test valide
    test_img = "../data/test_data/lol_dataset/our485/low/64.png" # Chemin original
    if not os.path.exists(test_img):
        # Recherche alternative
        candidates = glob.glob("../data/test_data/**/*.png", recursive=True)
        if candidates:
            test_img = candidates[0]
            print(f"⚠️ Image par défaut introuvable, utilisation de : {test_img}")
    
    ensemble_inference(test_img, model_A, model_B, device)
