import random
from tqdm import tqdm
import os
import cv2
import numpy as np
import yaml
from pathlib import Path

from logger import logger
from helper import load_yaml_file



def load_image(data_dir, target_name, size):
    """
    Charge les images à partir d'un répertoire spécifié et les étiquette en fonction de la cible définie.

    Args:
        data_dir (str): Le chemin du répertoire contenant les images.
        target_name (dict): Un dictionnaire indiquant les étiquettes associées aux classes. 
                            Par exemple : {"Mask": 0, "NoMask": 1}
        size (tuple): Une paire de dimensions (largeur, hauteur) pour redimensionner les images.

    Returns:
        list: Une liste de tuples où chaque tuple contient une image et son label associé.
    """
    list_images_labels = []
    logger.info(f"Chargement des images depuis : {data_dir}")
    for name in tqdm(target_name):
        logger.info(f"Début de l'enregistrement des images de : {name}...")
        path = os.path.join(data_dir, name)
        for image in os.listdir(path):
            array_image = cv2.imread(os.path.join(path,image), 0)
            resized_image = cv2.resize(array_image.astype(np.uint8), size)
            list_images_labels.append([resized_image, name])
        logger.info(f"Chargement des images de {name} terminé.")
    logger.info("Chargement des images terminé.")
    random.shuffle(list_images_labels)        
    return list_images_labels

def format_data(list_images_labels):
    """
    Prépare et formate les données d'images et de labels.

    Args:
        list_images_labels (list): Une liste de tuples où chaque tuple contient une image et son label associé.

    Returns:
        tuple: Un tuple contenant deux listes, la première contenant\
                les images formatées et la seconde les labels correspondants.
    """
    logger.info("Début de la préparation des données...")
    
    data_dict = {"images": [], "labels": []}
    for (image, label) in list_images_labels:
        data_dict["images"].append(image)
        data_dict["labels"].append(label) 
    
    images = np.array(data_dict["images"])
    labels = np.array(data_dict["labels"])
    
    logger.info("Préparation des données terminée.")
    
    return images, labels

if __name__ == "__main__":
    train_dir = Path("./mask_recognition/data/raw/train")
    val_dir = Path("./mask_recognition/data/raw/validation")
    test_dir = Path("./mask_recognition/data/raw/test")
    file_path = Path("./params.yaml")
    
    yaml_params = load_yaml_file(file_path=file_path)
    target_name = yaml_params["data_collector"]["target_name"]
    image_size = yaml_params["data_collector"]["image_size"]
    size = (image_size["width"], image_size["height"])
    
    train_data = load_image(data_dir=train_dir, target_name=target_name, size=size)
    val_data = load_image(data_dir=val_dir, target_name=target_name, size=size)
    test_data = load_image(data_dir=test_dir, target_name=target_name, size=size)
    
    
    prepared_train_data = format_data(train_data)
    prepared_val_data = format_data(val_data)
    prepared_test_data = format_data(test_data)
    np.savez("./mask_recognition/data/numpy_data/prepared_train_data.npz", image=prepared_train_data[0], label=prepared_train_data[1])
    logger.info("Les données prepared_train_data ont été enregistrées avec succès.")
    np.savez("./mask_recognition/data/numpy_data/prepared_val_data.npz", image=prepared_val_data[0], label=prepared_val_data[1])
    logger.info("Les données prepared_val_data ont été enregistrées avec succès.")
    np.savez("./mask_recognition/data/numpy_data/prepared_test_data.npz", image=prepared_test_data[0], label=prepared_test_data[1])
    logger.info("Les données prepared_test_data ont été enregistrées avec succès.")
    
    
    
