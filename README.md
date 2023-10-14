# Projet de Reconnaissance de Masque

Ce projet a pour objectif de mettre en place un système de reconnaissance de masque en utilisant des techniques de Deep Learning. 

L'application web qui en découle est construite avec FastAPI, offrant ainsi une interface utilisateur fluide et intuitive. 

Cependant, l'aspect innovant de ce projet ne s'arrête pas là : j'ai également développé une API permettant de générer aléatoirement des images de test.

## Fonctionnalités
- **Reconnaissance de Masque Avancée** : Le modèle de Deep Learning intégré permet une reconnaissance de masque.
- **Interface Utilisateur Intuitive** : L'application web, propulsée par FastAPI, offre une expérience utilisateur conviviale et agréable.
- **Génération d'Images de Test Aléatoires** : Une API spécialement conçue génère aléatoirement des images de test, offrant une multitude de scénarios pour tester l'efficacité du modèle de reconnaissance.
- **Prédiction** de la présence ou de l'absence de masque sur une image générée.

## Technologies Utilisées

- FastAPI : Un framework moderne pour le développement d'API web.
- TensorFlow : Une bibliothèque de machine learning pour créer et entraîner des modèles de Deep Learning.

## Gestion des Logs
Les logs de ce projet sont gérés en utilisant le module `logging` intégré de Python. Ils sont judicieusement placés à des points stratégiques dans le code afin d'offrir une visibilité précise sur le déroulement des opérations.

Par exemple, lors de la collecte des images, des logs sont générés pour indiquer le nombre total d'images collectées. De même, lors des phases de prétraitement des données et de construction du modèle, des informations détaillées sont enregistrées pour faciliter le suivi et le débogage du processus.
## Installation

Pour exécuter le projet localement, assurez-vous d'avoir Python installé sur votre système. Ensuite, suivez les étapes ci-dessous :

1. Clonez le dépôt :

```bash
git clone https://github.com/Julesba97/mask-recognition-tensorflow-fastapi.git
cd mask-recognition-tensorflow-fastapi
```

2. Installez les dépendances :
```bash
pip install -r requirements.txt

```
3. Lancez l'application :
```bash
python ./app/src/app.py
```

4. Ouvrez votre navigateur et accédez à http://localhost:8000 pour utiliser l'application.

## Source des Données

Les données utilisées dans ce projet proviennent du jeu de données "Reconnaissance de Masque" de Kaggle. Vous pouvez trouver plus d'informations ici.

## API Documentation
L'API offre deux points d'accès principaux : 

**POST /generate_random_image** : Génère une image de test aléatoire.

**POST /predict** : Effectue une prédiction sur l'image donnée.
