from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import numpy as np
from PIL import Image
import io
import os
import tensorflow as tf

app = FastAPI()

# Load the model
# Using a try-except block just in case the model is not found during initial startup or testing
try:
    model = tf.keras.models.load_model("potato_model.h5")
except Exception as e:
    model = None
    print("Warning: Model could not be loaded. Please ensure 'potato_model.h5' exists in the backend directory.")

# Keras ImageDataGenerator sorts class indices alphanumerically
CLASS_NAMES = ['Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy']

def preprocess_image(image: Image.Image):
    image = image.resize((256, 256))
    img_array = np.array(image)
    # Ensure image has 3 channels
    if len(img_array.shape) == 2:
        img_array = np.stack((img_array,)*3, axis=-1)
    # Rescale the image exactly as done in the notebook (rescale = 1./255)
    img_array = img_array / 255.0
    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if model is None:
        return JSONResponse(status_code=500, content={"error": "Model not loaded on the server."})
    
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        if image.mode != "RGB":
            image = image.convert("RGB")
        
        processed_img = preprocess_image(image)
        
        predictions = model.predict(processed_img)
        predicted_class_idx = np.argmax(predictions[0])
        confidence = float(predictions[0][predicted_class_idx])
        
        return {
            "class": CLASS_NAMES[predicted_class_idx],
            "confidence": confidence
        }
        
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": str(e)})

@app.get("/")
def read_root():
    return {"message": "Potato Disease Prediction API is running!"}
