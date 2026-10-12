from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
import pandas as pd
import joblib
import io
import math

app = FastAPI()

model = joblib.load("house_model.joblib")
features = joblib.load("house_features.joblib")

required_columns = [
    "MedInc",
    "HouseAge",
    "AveRooms",
    "AveBedrms",
    "Population",
    "AveOccup",
    "Latitude",
    "Longitude"
]


class HouseFeatures(BaseModel):
    MedInc: float = Field(gt=0, description="Median income of neighbourhood")
    HouseAge: float = Field(gt=0, description="Average age of houses in the block")
    AveRooms: float = Field(gt=0, description="Average number of rooms per house")
    AveBedrms: float = Field(gt=0, description="Average number of bedrooms per house")
    Population: float = Field(gt=0, description="Total population of the block")
    AveOccup: float = Field(gt=0, description="Average number of people per house")
    Latitude: float = Field(ge=32, le=42, description="Latitude")
    Longitude: float = Field(ge=-125, le=-114, description="Longitude")


@app.get("/")
def home():
    return {
        "message": "California House Price Prediction API",
        "status": "running",
        "endpoint": "Send POST request to /predict"
    }


@app.get("/health")
def health():
    return {
        "status": "running",
        "model": "RandomForestRegressor",
        "features": features,
        "avg_error": "$39,000"
    }


@app.post("/predict")
def predict(house: HouseFeatures):
    try:
        input_data = pd.DataFrame([{
            "MedInc": house.MedInc,
            "HouseAge": house.HouseAge,
            "AveRooms": house.AveRooms,
            "AveBedrms": house.AveBedrms,
            "Population": house.Population,
            "AveOccup": house.AveOccup,
            "Latitude": house.Latitude,
            "Longitude": house.Longitude
        }])

        predicted = float(model.predict(input_data)[0])
        price_usd = predicted * 100000

        return {
            "predicted_price": f"${price_usd:,.0f}",
            "predicted_price_short": f"${predicted:.2f} hundred thousands",
            "approximate_error_range": (
                f"${price_usd - 39000:,.0f} "
                f"to ${price_usd + 39000:,.0f}"
            )
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )


@app.post("/predict-file")
async def predict_file(file: UploadFile = File(...)):

    if not file.filename or not file.filename.lower().endswith(".csv"):
        raise HTTPException(
            status_code=400,
            detail="Please upload a CSV file only"
        )

    try:
        content = await file.read()

        if not content:
            raise HTTPException(
                status_code=400,
                detail="Uploaded file is empty"
            )

        df = pd.read_csv(io.BytesIO(content))

    except HTTPException:
        raise

    except Exception:
        raise HTTPException(
            status_code=400,
            detail="Unable to read the uploaded CSV file"
        )

    if df.empty:
        raise HTTPException(
            status_code=400,
            detail="Uploaded CSV file has no data rows"
        )

    missing_columns = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing_columns:
        raise HTTPException(
            status_code=400,
            detail=f"Missing required columns: {missing_columns}"
        )

    try:
        input_data = df[required_columns].copy()

        input_data = input_data.apply(
            pd.to_numeric,
            errors="raise"
        )

        if input_data.isnull().any().any():
            raise ValueError("Input data contains missing values")

        if not input_data.map(
            lambda value: math.isfinite(float(value))
        ).all().all():
            raise ValueError("Input data contains infinite values")

        positive_columns = [
            "MedInc",
            "HouseAge",
            "AveRooms",
            "AveBedrms",
            "Population",
            "AveOccup"
        ]

        if (input_data[positive_columns] <= 0).any().any():
            raise ValueError(
                "Income, age, rooms, bedrooms, population, "
                "and occupancy must be positive"
            )

        if not input_data["Latitude"].between(32, 42).all():
            raise ValueError("Latitude must be between 32 and 42")

        if not input_data["Longitude"].between(-125, -114).all():
            raise ValueError("Longitude must be between -125 and -114")

        predictions = model.predict(input_data)

        df["predicted_price_usd"] = predictions * 100000

        df["predicted_price_usd"] = df[
            "predicted_price_usd"
        ].apply(lambda value: f"${value:,.0f}")

        output = df.to_csv(index=False)

        return StreamingResponse(
            io.BytesIO(output.encode("utf-8")),
            media_type="text/csv",
            headers={
                "Content-Disposition": "attachment; filename=predictions.csv"
            }
        )

    except ValueError as err:
        raise HTTPException(
            status_code=400,
            detail=str(err)
        )

    except Exception as err:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(err)}"
        )