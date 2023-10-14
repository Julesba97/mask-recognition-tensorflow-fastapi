from typing import List
import os
import random

def get_image_filenames(directory):
    """
    Récupère la liste des noms de fichiers d'images dans le répertoire spécifié.

    Args:
        directory (str): Le chemin du répertoire contenant les images.

    Returns:
        list: Une liste contenant les noms de fichiers d'images.

    Exemple:
        >>> get_image_filenames('chemin/vers/le/repertoire/')
        ['image1.jpg', 'image2.png', 'image3.jpg']
    """
    
    return os.listdir(directory)

def select_random_image(directory):
    """
    Sélectionne aléatoirement une image à partir du répertoire spécifié.

    Args:
        directory (str): Le chemin du répertoire contenant les images.

    Returns:
        str: Le nom du fichier de l'image sélectionnée.
    """
    images = get_image_filenames(directory)
    images.remove("css")
    random_image = random.choice(images)
    image_path = os.path.join(directory, random_image)
    return image_path
