from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import tflite_runtime.interpreter as tflite

app = FastAPI()

# Load TFLite model
interpreter = tflite.Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

class InputData(BaseModel):
    input: list  # assuming your model expects flat list of floats

@app.get("/")
def read_root():
    return {"message": "TFLite Model Server is Running"}

@app.post("/predict")
def predict(data: InputData):
    try:
        # Convert input list to numpy array
        input_array = np.array([data.input], dtype=np.float32)
        
        # Set input tensor
        interpreter.set_tensor(input_details[0]['index'], input_array)
        
        # Run inference
        interpreter.invoke()
        
        # Get output tensor
        output_data = interpreter.get_tensor(output_details[0]['index'])
        
        return {"prediction": output_data.tolist()}

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
