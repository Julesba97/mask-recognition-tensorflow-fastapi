from fastapi import FastAPI, Request
import uvicorn
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import os
from image_random_selector import select_random_image
from recognition_mask_predictor import mask_recognition

app = FastAPI()
image_to_predict = None
image_path_predict = None
templates = Jinja2Templates(directory="templates")

app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/generate_random_image/", response_class=HTMLResponse)
def generate_random_image(request: Request) :
    """
    Génère et prédit une image à partir de la requête fournie.

    Args:
        request (Request): L'objet Request contenant les informations nécessaires pour générer l'image.

    Returns:
        TemplateResponse: Une réponse de modèle pour afficher la page d'accueil.
    """
    static_path = os.path.join(os.getcwd(), "static")
    image_path = select_random_image(static_path)
    global image_to_predict
    image_to_predict = image_path
    image_path_split = image_path.split("\\")
     
    image_path = f"/{image_path_split[-2]}/{image_path_split[-1]}"
    global image_path_predict
    image_path_predict = image_path
    context = {"request": request, "image_path": image_path}
    return templates.TemplateResponse("home.html", context)

@app.post("/generate_random_image/", response_class=HTMLResponse)
def generate_random_image(request: Request) :
    """
    Génère et prédit une image à partir de la requête fournie.

    Args:
        request (Request): L'objet Request contenant les informations nécessaires pour générer l'image.

    Returns:
        TemplateResponse: Une réponse de modèle pour afficher la page d'accueil.
    """
    context = {"request": request}
    return templates.TemplateResponse("home.html", context)

@app.post("/predict/", response_class=HTMLResponse)
def predict(request: Request) :
    """
    Effectue une prédiction à partir de la requête fournie.

    Args:
        request (Request): L'objet Request contenant les informations nécessaires pour effectuer la prédiction.

    Returns:
        TemplateResponse: Une réponse de modèle pour afficher la page de prédiction.
    """
    image_model = "."+image_path_predict
    mask_prediction = mask_recognition(image_model)
    context = {"request": request,
               "image_path_predict": image_path_predict, 
               "mask_prediction": mask_prediction}
    return templates.TemplateResponse("predict.html", context)

    

if __name__ == "__main__":
    uvicorn.run("app:app", host="localhost", port=8000, reload=True)