from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout, Rescaling, Resizing
from keras.optimizers import Adam
from keras.models import Sequential
from pathlib import Path
import numpy as np
import pandas as pd
import os

from helper import load_yaml_file
from logger import logger

file_path = Path("./params.yaml")
processed_train_path = Path("./mask_recognition/data/processed/processed_train_data.npz")
processed_val_path = Path("./mask_recognition/data/processed/processed_val_data.npz")

processed_train_data = np.load(processed_train_path)
train_X = processed_train_data["image"]
train_y = processed_train_data["label"]

processed_val_data = np.load(processed_val_path)
val_X = processed_val_data["image"]
val_y = processed_val_data["label"]

yaml_params = load_yaml_file(file_path=file_path)

image_size = yaml_params["deep_mask_detector"]["image_size"]
kernel_size = yaml_params["deep_mask_detector"]["kernel_size"]
strides = yaml_params["deep_mask_detector"]["strides"]
padding = yaml_params["deep_mask_detector"]["padding"]
activation = yaml_params["deep_mask_detector"]["activation"]
learning_rate = yaml_params["deep_mask_detector"]["learning_rate"]



Resize_Rescale = Sequential()
Resize_Rescale.add(Resizing(image_size, image_size))
Resize_Rescale.add(Rescaling(1./255, offset=0))

def build_cnn_model():
    """
    Construit un modèle de reconnaissance de masque avec un réseau de neurones à convolution (CNN).

    Returns:
        tensorflow.keras.Model: Le modèle CNN construit.
    """
    logger.info("Début de la construction du modèle CNN...")
    model = Sequential()
    model.add(Resize_Rescale)
    model.add(Conv2D(filters=16, kernel_size=kernel_size, 
                     strides=strides,  padding=padding, 
                     activation=activation))
    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Conv2D(filters=32, kernel_size=kernel_size,
                     strides=strides, padding=padding, 
                     activation=activation))
    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Conv2D(filters=64, kernel_size=kernel_size, 
                     strides=strides, padding=padding, activation=activation))
    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Flatten())
    model.add(Dense(units=256, activation=activation))
    model.add(Dense(units=1, activation="sigmoid"))

    input_shape = train_X.shape 
    model.build(input_shape) 
    
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss="binary_crossentropy", metrics=["accuracy"])
    
    logger.info("Modèle compilé avec succès.")
    logger.info("Construction du modèle CNN terminée.")

    return model

if __name__  == "__main__":
    model = build_cnn_model()
    #print(model.summary())
    activation = yaml_params["deep_mask_detector"]["activation"]
    batch_size = yaml_params["deep_mask_detector"]["batch_size"]
    epochs = yaml_params["deep_mask_detector"]["epochs"]
    
    logger.info("Entraînement du modèle en cours...")
    history = model.fit(train_X, train_y, batch_size=batch_size, epochs=epochs, validation_data=(val_X, val_y))
    logger.info("Modèle entraîné avec succès.")
    history_df = pd.DataFrame(history.history)
    
    logger.info("Enregistrement de l'historique des évaluations du modèle...")
    artefacts_dir = "./artefacts"
    os.makedirs(artefacts_dir, exist_ok=True)
    history_df.to_csv(f"./{artefacts_dir}/history.csv", index=False)
    logger.info("L'historique des évaluations du modèle a été enregistré.")
    
    logger.info("Enregistrement du modèle en cours...")
    model.save(f"./{artefacts_dir}/mask_recognition_model.h5")
    logger.info("Enregistrement du modèle terminé.")

    
    