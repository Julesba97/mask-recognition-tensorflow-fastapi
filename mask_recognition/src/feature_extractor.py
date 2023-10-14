import numpy as np
import cv2
from pathlib import Path

from logger import logger
from helper import load_yaml_file

def apply_kernel_convolution(images, kernel):
    """
    Applique une convolution avec un noyau spécifié sur une série d'images.

    Args:
        images (list[numpy.ndarray]): Une liste d'images représentées sous forme de tableaux NumPy.
        kernel (numpy.ndarray): Le noyau à utiliser pour la convolution.

    Returns:
        list[numpy.ndarray]: Une liste d'images après l'application de la convolution.
    """
    size = images[0].shape[0]
    convoluted_images = []
    logger.info("Début de la convolution sur images...")
    for image in images:
        convoluted_image = cv2.filter2D(image, -1, kernel)
        convoluted_image = convoluted_image.reshape((size, size, 1))
        convoluted_images.append(convoluted_image)
    logger.info(f"La convolution a été appliquée à {len(convoluted_images)} images.")
    logger.info("Toutes les convolutions sont terminées.")
    return np.array(convoluted_images)

def encoder_label(labels):
    """
    Encode les étiquettes de classe en format numérique.

    Args:
        labels (list[str]): Une liste d'étiquettes de classe.

    Returns:
        list[int]: Une liste d'entiers représentant les étiquettes encodées.
    """
    encoded_target = np.where(labels == 'NoMask', 1, 0)
    logger.info(f"{len(labels)} étiquettes ont été encodées.")
    return encoded_target


if __name__ == "__main__":
    prepared_train_path = Path("./mask_recognition/data/numpy_data/prepared_train_data.npz")
    prepared_val_path = Path("./mask_recognition/data/numpy_data/prepared_val_data.npz")
    prepared_test_path = Path("./mask_recognition/data/numpy_data/prepared_test_data.npz")
    file_path = Path("./params.yaml")

    prepared_train_data = np.load(prepared_train_path)
    train_images = prepared_train_data["image"]
    train_labels = prepared_train_data["label"]
    
    prepared_val_data = np.load(prepared_val_path)
    val_images = prepared_val_data["image"]
    val_labels = prepared_val_data["label"]
    
    prepared_test_data = np.load(prepared_test_path)
    test_images = prepared_test_data["image"]
    test_labels = prepared_test_data["label"]
    
    yaml_params = load_yaml_file(file_path=file_path)
    kernel = np.array(yaml_params["feature_extractor"]["kernel"])
    
    processed_train_images = apply_kernel_convolution(train_images, kernel)
    encoded_train_labels  = encoder_label(train_labels)
    
    processed_val_images = apply_kernel_convolution(val_images, kernel)
    encoded_val_labels = encoder_label(val_labels)
    
    processed_test_images = apply_kernel_convolution(test_images, kernel)
    encoded_test_labels = encoder_label(test_labels)
    
    
    np.savez("./mask_recognition/data/processed/processed_train_data.npz", 
             image=processed_train_images, label=encoded_train_labels)
    logger.info("Les données processed_train_data ont été enregistrées avec succès.")
    
    np.savez("./mask_recognition/data/processed/processed_val_data.npz",
             image=processed_val_images, label=encoded_val_labels)
    logger.info("Les données processed_val_data ont été enregistrées avec succès.")
    
    np.savez("./mask_recognition/data/processed/processed_test_data.npz",
             image=processed_test_images, label=encoded_test_labels)
    logger.info("Les données processed_test_data ont été enregistrées avec succès.")
    