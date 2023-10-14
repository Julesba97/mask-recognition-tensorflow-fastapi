from keras.models import load_model
import numpy as np
import cv2
import yaml
from pathlib import Path
import os
import matplotlib.pyplot  as plt


mask_recognition_model_path = Path("./mask_recognition_model.h5")

kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1,  -1]])
size = (200, 200)
target_name = {"Mask": 0, "NoMask": 1}


def extract_feature(image_path, kernel):
    """
    Extrait des caractéristiques d'une image en appliquant une convolution avec le noyau spécifié.

    Args:
        image_path (str): Le chemin de l'image à traiter.
        kernel (np.ndarray): Le noyau de convolution.

    Returns:
        np.ndarray: Un tableau NumPy contenant l'image filtrée.
    """
    image = cv2.imread(image_path, 0)
    resized_image = cv2.resize(image, size)
    filted_image = cv2.filter2D(resized_image, -1, kernel).astype(np.uint8).reshape(size[0],size[1], 1)
    return np.array([filted_image])



def mask_recognition(image_path):
    """
    Effectue la reconnaissance de masque sur une image donnée.

    Args:
        image_path (str): Le chemin de l'image à traiter.

    Returns:
        str: Le résultat de la reconnaissance ('Avec Masque' ou 'Sans Masque').
    """
    image_filtered = extract_feature(image_path, kernel)
    inv_target_name = {v: k for k, v in target_name.items()}
    mask_recognition = load_model(mask_recognition_model_path)
    y_pred = mask_recognition.predict(image_filtered).round().reshape(-1)
    return inv_target_name[int(y_pred)]


