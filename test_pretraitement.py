import pretraitement
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt

img_p1 = np.array(cv2.imread('CelebA/images/000001.jpg'))
img_p2 = np.array(cv2.imread('CelebA/images/000002.jpg'))
img_p3 = np.array(cv2.imread('CelebA/images/000003.jpg'))

plt.figure()
plt.imshow(cv2.cvtColor(img_p1, cv2.COLOR_BGR2RGB))
plt.title("test image 1")
plt.axis('off')
plt.show()

preprocess = pretraitement.Pretraitement() #créer une instance de la classe Pretraitement

img_preprocess = preprocess.preprocess_image(img_p1) #prétraite une image

plt.figure()
preprocess.visualize_image(img_preprocess) #affiche l'image prétraitée
plt.title("test image 1 prétraitée")

print("Image prétraitée shape:", img_preprocess.shape)
print("valeur max de l'image prétraitée :", torch.max(img_preprocess)) #vérifie bien que le max est 1
print("valeur min de l'image prétraitée :", torch.min(img_preprocess)) #vérifie bien que le min c'est - 1

batch_images = [img_p1, img_p2, img_p3] #crée une liste d'images
batch_preprocessed = preprocess.preprocess_batch(batch_images) #prétraite le batch d'images
print("Batch prétraité shape:", batch_preprocessed.shape)
#pour l'instant on peut pas visualiser un batch entier vu la forme du batch qui concatene les images le long de la dimension 0 (channels, height, width)
