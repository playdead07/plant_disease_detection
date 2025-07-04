from fastapi import FastAPI, HTTPException, File, UploadFile
from pydantic import BaseModel
import numpy as np
import tflite_runtime.interpreter as tflite
from PIL import Image
import io

app = FastAPI()

# Load TFLite model
interpreter = tflite.Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

class InputData(BaseModel):
    input: list  # assuming your model expects flat list of floats

labels = [
    "Strawberry Leaf Scorch",
    "Tomato Leaf Mold",
    "Tomato Mosaic Virus",
    "Corn Common Rust",
    "Potato Early Blight",
    "Corn Gray Leaf Spot"
]

@app.get("/")
def read_root():
    return {"message": "TFLite Model Server is Running"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Read and preprocess image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        image = image.resize((224, 224))
        input_array = np.array(image, dtype=np.uint8)
        input_array = np.expand_dims(input_array, axis=0)  # shape: (1, 224, 224, 3)

        # Apply quantization if needed
        scale, zero_point = input_details[0]['quantization']
        if scale > 0:
            input_array = (input_array / scale + zero_point).astype(np.uint8)

        # Set input tensor
        interpreter.set_tensor(input_details[0]['index'], input_array)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])

        # Get predicted label
        pred_idx = int(np.argmax(output_data))
        pred_label = labels[pred_idx]
        return {"label": pred_label}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
