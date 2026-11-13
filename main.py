from fastapi import FastAPI
from pydantic import BaseModel
import joblib


app = FastAPI()

#load the model
model = joblib.load("model/house_price_model.pkl")

class HouseInput(BaseModel):
    area: int
    bed_rooms: int
    age: int

@app.get("/")
def home():
    return {"msg":"FastAPI houseprice pediction project"}

@app.post("/ratepredict")
def predict_price(data: HouseInput):
    prediction = model.predict([[data.area,data.bed_rooms,data.age]])
    return {
            "Area":data.area,
            "Bed Rooms":data.bed_rooms,
            "Age ":data.age,
            "Price ":float(prediction[0])
        }
