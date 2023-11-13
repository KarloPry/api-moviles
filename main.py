import os
import io
import numpy as np
import pandas as pd
from fastapi import FastAPI, UploadFile, File
from starlette.responses import JSONResponse
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.models import load_model

app = FastAPI()

# Define the path to the trained model
model_path = './food_model.hdf5'

# Load the trained model
model = load_model(model_path)

# Define the target image size (must match the size used during training)
image_size = (235, 235)

# Define class labels for your problem (e.g., the classes used during training)
class_labels = ['cheesecake', 'lasagna', 'pancakes']


def predict_image(file) -> dict:
    # Load and preprocess the image for prediction
    image = load_img(io.BytesIO(file), target_size=image_size)
    image = img_to_array(image)
    image = image / 255.0  # Normalize the pixel values to be in the range [0, 1]
    image = image.reshape((1, *image.shape))  # Reshape to match the model's input shape

    # Make predictions on the image
    predictions = model.predict(image)

    # Interpret the results
    predicted_class = class_labels[np.argmax(predictions)]
    confidence = float(np.max(predictions))

    result = {
        'Predicted Class': predicted_class,
        'Confidence': confidence
    }

    return result

@app.post("/predict")
async def predict_endpoint(file: UploadFile = File(...)):
    try:
        result = predict_image(await file.read())
        return JSONResponse(content=result)
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
