import yaml
from yaml.loader import SafeLoader
from pathlib import Path
import os

def load_yaml_file(file_path):
    """
    Charge un fichier YAML.

    Args:
        chemin (str): Le chemin du fichier YAML.

    Returns:
        dict: Le contenu du fichier YAML sous forme de dictionnaire
    """
    with open(file_path, 'r') as file:
        content = yaml.load(file, Loader=SafeLoader)
    return content


#os.chdir("./Projet-DL")
#file_path = Path("./params.yaml")
#yaml_params = load_yaml_file(file_path=file_path)
#print(os.getcwd())
