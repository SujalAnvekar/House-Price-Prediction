from fastapi import FastAPI,HTTPException
import pandas as pd
from pydantic import BaseModel,Field
import joblib

app=FastAPI()

model=joblib.load("house_model.joblib")
features=joblib.load("house_features.joblib")

class HouseFeatures(BaseModel):
    MedInc:float=Field(gt=0,description="Median income of Neighbourhood")
    HouseAge:float=Field(gt=0,description="Average age of house in the block")
    AveRooms:float=Field(gt=0,description="Average number of rooms per house")
    AveBedrms:float=Field(gt=0,description="Average number of bedrooms per house")
    Population:float=Field(gt=0,description="Total population the block")
    AveOccup:float=Field(gt=0,description="Average number of people per house")
    Latitude:float=Field(ge=32,le=42,description="Latitude")
    Longitude:float=Field(ge=-125,le=-114,description="Longitide")


@app.get('/')
def home():
    return{
        "message":"California house prediction api",
        "status":"running",
        "endpoint":"send Post request to /predict"
    }


@app.get('/health')
def health():
    return{
        "status":"running",
        "model":"RandomForestRegressor",
        "features":features,
        "avg_error":"$39,000"
    }


@app.post('/predict')
def predict(house:HouseFeatures):
    try:
        input_data=pd.DataFrame([{
            "MedInc":house.MedInc,
            "HouseAge":house.HouseAge,
            "AveRooms":house.AveRooms,
            "AveBedrms":house.AveBedrms,
            "Population":house.Population,
            "AveOccup":house.AveOccup,
            "Latitude":house.Latitude,
            "Longitude":house.Longitude
        }])

        predicted=model.predict(input_data)[0]
        price_usd=predicted*100000

        return{
            "predicted_price":f"${price_usd:,.0f}",
            "predicted_price_short":f"${predicted:,.2f} hundred thousands",
            "fidence_range":f"${price_usd-39000:,.0f} to ${price_usd+39000:,.0f}"
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"prediction failed:{str(e)}"
        )