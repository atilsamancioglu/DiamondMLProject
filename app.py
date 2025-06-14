import uvicorn
from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
import pickle
import pandas as pd
from pydantic import BaseModel

app = FastAPI()

# Templates
templates = Jinja2Templates(directory="templates")

# Load model and preprocessors
with open('30-diamond_model_complete.pkl', 'rb') as f:
    saved_data = pickle.load(f)
    model = saved_data['model']
    encoders = saved_data['encoders']
    scaler = saved_data['scaler']

class DiamondFeatures(BaseModel):
    carat: float
    cut: str
    color: str
    clarity: str
    depth: float
    table: float
    x: float
    y: float
    z: float

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict")
async def predict(features: DiamondFeatures):
    # Create a DataFrame with the input features
    input_data = pd.DataFrame([features.model_dump()])
    
    # Apply label encoding using the saved encoders
    for col in ['cut', 'color', 'clarity']:
        input_data[col] = encoders[col].transform(input_data[col])
    
    # Apply standard scaling using the saved scaler
    input_scaled = scaler.transform(input_data)
    
    # Make prediction
    prediction = model.predict(input_scaled)[0]
    
    return {"predicted_price": float(prediction)}