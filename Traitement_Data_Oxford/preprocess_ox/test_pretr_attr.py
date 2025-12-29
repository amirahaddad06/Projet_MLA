#Code pour vérifier si le prétraitement des attributs a été correctement fait

import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
attributes_path = '/home/ityt/Documents/Hicham/M2/MLA/Projet_MLA/Traitement_Data_Oxford/Data_preprocessed/attributes_ox.pth' #chemin vers le fichier des attributs prétraités

data = torch.load(attributes_path, weights_only=False)
img_dir = '/home/ityt/Documents/Hicham/M2/MLA/102flowers/jpg' # chemin vers le dossier des images
color_keys = list(data.keys()) # ['red', 'blue', ...]

def check_image_attributes(img_idx):
    img_name = f"image_{img_idx + 1:05d}.jpg"
    
    # on récupère la couleur de l'image
    active_colors = []
    for color in color_keys:
        if data[color][img_idx] == 1: #si c'est 1 alors c'est cette couleur
            active_colors.append(color)
    
    # Affichage
    img_path = os.path.join(img_dir, img_name)
    plt.imshow(Image.open(img_path))
    plt.title(f"{img_name}\nCouleurs détectées: {', '.join(active_colors)}")
    plt.axis('off')
    plt.show()

# Tester sur l'image 567
check_image_attributes(567 - 1)