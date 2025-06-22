from fastapi import FastAPI , HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
from predict import evaluatePrice
app = FastAPI()

# Allow all origins (you can restrict this later)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request schema
class InputFeatures(BaseModel):
    features: List[float]  # must be of length 7 (excluding bias term)


# API route
@app.post("/predict" ,status_code=201)
def predict(input_data: InputFeatures):
    try:
        result = evaluatePrice(input_data.features)
        return {"estimated_price": round(result, 2)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e) )
try:
    import uvicorn
    PORT = 3000
    uvicorn.run(app , host = "127.0.0.1" , port = PORT)
except Exception as ex :
    print(f"Error => {ex}")