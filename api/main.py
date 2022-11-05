from fastapi import FastAPI, UploadFile, File
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf
from fastapi.middleware.cors import CORSMiddleware
import os

app = FastAPI()

origins = [
    "http://localhost",
    "http://localhost:3000",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
print(os.getcwd())
MODEL = tf.keras.models.load_model("../save_models/1")

CLASS_NAMES = ["clean", "dirty"]

@app.get("/ping")
async def ping():
    val=20
    return "hello i am alive"

def read_file_as_image(data) ->np.ndarray:
    image=np.array(Image.open(BytesIO(data)))
    return image

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image = read_file_as_image(await file.read())
    img_batch = np.expand_dims(image, axis=0)
    
    prediction = MODEL.predict(img_batch)
    
    index = np.argmax(prediction[0])

    predicted_class = CLASS_NAMES[index]

    confidence = np.max(prediction[0])

    print({
        'class':predicted_class,
        'confidence': float(confidence)
    })
    return {
        'class':predicted_class,
        'confidence': float(confidence)
    }

if __name__== "__main__":
    uvicorn.run(app, host='localhost', port=8000)