from fastapi import FastAPI, File, UploadFile, Body
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf
from pydantic import BaseModel
from rdflib import Graph, Namespace
import requests
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict, Any
from fastapi import Depends, HTTPException, status
import json
import cv2
import os

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


current_directory = os.path.dirname(os.path.realpath(__file__))

# Construct the path to your model relative to the current directory
model_path = os.path.join(current_directory, "vgg model")

# Load the model
MODEL = tf.keras.models.load_model(model_path)


# MODEL = tf.keras.models.load_model(
#     "D:\Project works and models\Model2\VGG16_fine_tuned_10")
CLASS_NAMES = ['Tomato_Bacterial_spot',
               'Tomato_Early_blight',
               'Tomato_Late_blight',
               'Tomato_Leaf_Mold',
               'Tomato_Septoria_leaf_spot',
               'Tomato_Spider_mites_Two_spotted_spider_mite',
               'Tomato__Black_mold',
               'Tomato__Gray_spot',
               'Tomato__Target_Spot',
               'Tomato__Tomato_YellowLeaf__Curl_Virus',
               'Tomato__Tomato_mosaic_virus',
               'Tomato__powdery_mildew',
               'Tomato_healthy']

disease_mapping = {
    "Tomato_Bacterial_spot": "bacterial_spot",
    "Tomato_Early_blight": "early_blight",
    "Tomato_Late_blight": "late_blight",
    "Tomato_Leaf_Mold": "leaf_mold",
    "Tomato_Septoria_leaf_spot": "septoria_leaf_spot",
    "Tomato_Spider_mites_Two_spotted_spider_mite": "spider_mites",
    "Tomato__Black_mold": "black_mold",
    "Tomato__Gray_spot": "gray_spot",
    "Tomato__Target_Spot": "target_spot",
    "Tomato__Tomato_YellowLeaf__Curl_Virus": "yellow_leaf_curl",
    "Tomato__Tomato_mosaic_virus": "mosaic_virus",
    "Tomato__powdery_mildew": "powdery_mildew",
    "Tomato_healthy": "healthy"
}


class ExtraSymptomModelOut(BaseModel):
    hasLeafSymptom: str = 'lesions'
    hasLeafSymptomColour: str = 'yellow'
    hasFruitSymptom: str = 'Spot_symptom'
    hasFruitSymptomColor: str = None
    hasStemSymptom: str = None
    hasStemSymptomColor: str = None
    hasLeafHalo: str = None
    hasLeafHaloColor: str = None
    hasFruitHalo: str = None
    hasFruitHaloColor: str = None
    hasFungusSymptom: str = None
    hasFungusSymptomColor: str = None


def build_sparql_query(symptoms):
    # Build SPARQL query based on symptoms
    query = """PREFIX OntoML: <https://github.com/mtbstn24/OntoMLv3#>
SELECT DISTINCT (strafter(STR(?disease), "#") AS ?diseaseName)
WHERE {
?disease rdf:type OntoML:Disease.
"""
    for symptom, value in symptoms.items():
        if value:
            query += f"?disease OntoML:{symptom} OntoML:{value}.\n"
    query += "}"
    return query


def is_blurry(image, threshold=100):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Compute the Laplacian
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    print(laplacian_var)
    return laplacian_var < threshold


@app.get("/ping")
async def ping():
    return "Hello, I am alive"


def read_file_as_image(data, target_size=(224, 224)) -> np.ndarray:
    image = Image.open(BytesIO(data))

    # Resize the image
    resized_image = tf.image.resize(np.array(image), target_size)
    print(resized_image.shape)
    return resized_image


class PredictionFacade:
    def __init__(self, model_path, ontology_url):
        self.model = tf.keras.models.load_model(model_path)
        self.class_names = ['Tomato_Bacterial_spot', 'Tomato_Early_blight', 'Tomato_Late_blight', 'Tomato_Leaf_Mold',
                            'Tomato_Septoria_leaf_spot', 'Tomato_Spider_mites_Two_spotted_spider_mite',
                            'Tomato__Black_mold', 'Tomato__Gray_spot', 'Tomato__Target_Spot',
                            'Tomato__Tomato_YellowLeaf__Curl_Virus', 'Tomato__Tomato_mosaic_virus',
                            'Tomato__powdery_mildew', 'Tomato_healthy']
        self.disease_mapping = {
            "Tomato_Bacterial_spot": "bacterial_spot",
            "Tomato_Early_blight": "early_blight",
            "Tomato_Late_blight": "late_blight",
            "Tomato_Leaf_Mold": "leaf_mold",
            "Tomato_Septoria_leaf_spot": "septoria_leaf_spot",
            "Tomato_Spider_mites_Two_spotted_spider_mite": "spider_mites",
            "Tomato__Black_mold": "black_mold",
            "Tomato__Gray_spot": "gray_spot",
            "Tomato__Target_Spot": "target_spot",
            "Tomato__Tomato_YellowLeaf__Curl_Virus": "yellow_leaf_curl",
            "Tomato__Tomato_mosaic_virus": "mosaic_virus",
            "Tomato__powdery_mildew": "powdery_mildew",
            "Tomato_healthy": "healthy"
        }
        self.ontology_url = ontology_url

    def process_image(self, file):
        image = Image.open(BytesIO(file))
        resized_image = tf.image.resize(np.array(image), (224, 224))

        image_np = np.array(resized_image)
        # if len(image_np.shape) > 2 and image_np.shape[2] == 4:
        #     # convert the image from RGBA2RGB
        #     image_np = cv2.cvtColor(image_np, cv2.COLOR_BGRA2BGR)
        # print(image_np.shape)
        image_np_uint8 = image_np.astype(np.uint8)

        if self.is_blurry(image_np_uint8):
            return {"disease": "Image is Blurry"}

        img_batch = np.expand_dims(resized_image, 0)
        predictions = self.model.predict(img_batch)
        confidence_threshold = 0.0
        class_confidence_map = dict(zip(self.class_names, predictions[0]))
        filtered_diseases = [disease for disease,
                             confidence in class_confidence_map.items() if confidence >= confidence_threshold]
        deep_model_sorted_disease = sorted(
            filtered_diseases, key=lambda x: class_confidence_map[x], reverse=True)

        ontology_satisfying_diseases = self.query_ontology()

        output_disease = self.match_disease(
            deep_model_sorted_disease, ontology_satisfying_diseases)

        return {"disease": output_disease}

    def is_blurry(self, image, threshold=50):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        return laplacian_var < threshold

    def query_ontology(self):
        response = requests.get(self.ontology_url)
        if response.status_code == 200:
            g = Graph()
            g.parse(data=response.text, format="application/rdf+xml")
            results = g.query(build_sparql_query(EXTRA_SYMPTOMS))
            ontology_satisfying_diseases = [
                row.diseaseName.value for row in results]
            return ontology_satisfying_diseases
        else:
            raise Exception("Failed to fetch ontology information")

    def match_disease(self, deep_model_sorted_disease, ontology_satisfying_diseases):
        output_disease = "Not found"
        if not deep_model_sorted_disease:
            return output_disease

        if len(ontology_satisfying_diseases) == 0:
            output_disease = deep_model_sorted_disease[0]
        elif len(ontology_satisfying_diseases) == 1:
            output_disease = ontology_satisfying_diseases[0]
        else:
            for deep_disease in deep_model_sorted_disease:
                for onto_disease in ontology_satisfying_diseases:
                    if self.disease_mapping.get(deep_disease) == onto_disease:
                        output_disease = deep_disease
                        return output_disease
        return output_disease

    def detect_ontology(self):
        response = requests.get(self.ontology_url)
        if response.status_code == 200:
            g = Graph()
            g.parse(data=response.text, format="application/rdf+xml")
            results = g.query(build_sparql_query(EXTRA_SYMPTOMS))
            ontology_satisfying_diseases = [
                row.diseaseName.value for row in results]
            return {"disease": ontology_satisfying_diseases}
        else:
            raise Exception("Failed to fetch ontology information")


# C:\\Users\\USER\\Desktop\\vgg model

@app.post("/image_upload")
async def predict(file: UploadFile = File(...)):
    print("accessed")
    facade = PredictionFacade("C:\\Users\\USER\\Desktop\\vgg model",
                              "https://raw.githubusercontent.com/Sribarathvajasarma/Plant_disease_ontology_2/main/OntoMLv3.rdf")
    result = facade.process_image(await file.read())
    return result


# https://raw.githubusercontent.com/mtbstn24/OntoML/main/OntoMLv3.rdf


@app.get("/ontology_detection")
async def predict_ontology():
    facade = PredictionFacade("C:\\Users\\USER\\Desktop\\vgg model",
                              "https://raw.githubusercontent.com/Sribarathvajasarma/Plant_disease_ontology_2/main/OntoMLv3.rdf")
    result = facade.detect_ontology()
    return result


# @app.post("/image_upload")
# async def predict(
#     file: UploadFile = File(...),
# ):

#     image = await file.read()
#     image = Image.open(BytesIO(image))
#     resized_image = tf.image.resize(np.array(image), (224, 224))

#     image_np = np.array(resized_image)
#     image_np_uint8 = image_np.astype(np.uint8)

#     # EXTRA_SYMPTOMS = {
#     #     "hasLeafSymptom": "lesions",
#     #     "hasLeafSymptomColour": "yellow",
#     # }

#     if is_blurry(image_np_uint8):
#         return {
#             "error": "Image is Blurry",
#         }

#     img_batch = np.expand_dims(resized_image, 0)
#     predictions = MODEL.predict(img_batch)
#     confidence_threshold = 0.0
#     class_confidence_map = dict(zip(CLASS_NAMES, predictions[0]))
#     filtered_diseases = [disease for disease,
#                          confidence in class_confidence_map.items() if confidence >= confidence_threshold]
#     Deep_Model_Sorted_Disease = sorted(
#         filtered_diseases, key=lambda x: class_confidence_map[x], reverse=True)

#     github_raw_uri = "https://raw.githubusercontent.com/mtbstn24/OntoML/main/OntoMLv3.rdf"

#     # Fetch the RDF file from the GitHub repository
#     response = requests.get(github_raw_uri)

#     if response.status_code == 200:
#         # Create a Graph
#         g = Graph()
#         g.parse(data=response.text, format="application/rdf+xml")

#         query = build_sparql_query(EXTRA_SYMPTOMS)

#         # Execute the SPARQL query
#         results = g.query(query)

#         ontology_satisfying_diseases = []

#         for row in results:
#             disease = row.diseaseName
#             ontology_satisfying_diseases.append(disease.value)

#         print("Ontology disease")

#         print(ontology_satisfying_diseases)

#         found = False
#         output_disease = "Not found"
#         if len(ontology_satisfying_diseases) == 0:
#             output_disease = Deep_Model_Sorted_Disease[0]
#         elif len(ontology_satisfying_diseases) == 1:
#             output_disease = ontology_satisfying_diseases[0]
#         elif len(ontology_satisfying_diseases) == 0 and len(Deep_Model_Sorted_Disease) == 0:
#             print("Couldn't find the disease with given information")
#         else:

#             for deep_disease in Deep_Model_Sorted_Disease:
#                 for onto_disease in ontology_satisfying_diseases:
#                     print(deep_disease)
#                     print(onto_disease)
#                     if disease_mapping[deep_disease] == onto_disease:
#                         output_disease = deep_disease
#                         found = True
#                         break
#                 if found:
#                     break

#     else:
#         print("Failed to fetch ontology information")

#     return {
#         "disease": output_disease,
#     }


# @app.get("/ontology_detection")
# async def predict_ontology():

#     # payload = {
#     #     "hasLeafSymptom": "lesions",
#     #     "hasLeafSymptomColour": "yellow",
#     # }

#     github_raw_uri = "https://raw.githubusercontent.com/mtbstn24/OntoML/main/OntoMLv3.rdf"

#     # Fetch the RDF file from the GitHub repository
#     response = requests.get(github_raw_uri)

#     if response.status_code == 200:
#         # Create a Graph
#         g = Graph()
#         g.parse(data=response.text, format="application/rdf+xml")

#         query = build_sparql_query(EXTRA_SYMPTOMS)

#         # Execute the SPARQL query
#         results = g.query(query)

#         ontology_satisfying_diseases = []

#         for row in results:
#             disease = row.diseaseName
#             ontology_satisfying_diseases.append(disease.value)

#         print(ontology_satisfying_diseases)

#     else:
#         print("Failed to fetch ontology information")

#     return {
#         "disease": ontology_satisfying_diseases,
#     }


@app.post("/extra_symptoms")
async def predict(
    payload: Dict[Any, Any]
):
    global EXTRA_SYMPTOMS
    EXTRA_SYMPTOMS = payload
    print(EXTRA_SYMPTOMS)
    return{
        "message": "Data received successfully"
    }


@app.post("/extra_symptomss")
async def predictttt(payload: Dict[Any, Any]):
    print(payload)
    return {"message": "Data received successfully"}


@app.post("/predict")
async def predict_image(
    file: UploadFile = File(...),
):
    image = await file.read()
    image = Image.open(BytesIO(image))
    resized_image = tf.image.resize(np.array(image), (224, 224))
    img_batch = np.expand_dims(resized_image, 0)
    predictions = MODEL.predict(img_batch)
    predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
    return{
        "disease": predicted_class,
    }


if __name__ == "__main__":
    uvicorn.run(app, host='localhost', port=8000)
