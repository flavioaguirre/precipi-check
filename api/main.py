# api/main.py

from fastapi import FastAPI
from pydantic import BaseModel, Field
import joblib
import pandas as pd
import uvicorn
from typing import Literal

# ----- 1. Load model at startup -----

MODEL_PATH = "models/model_randomforest_precipicheck_26-11-2025.pkl"
model = joblib.load(MODEL_PATH)
best_model = model.best_estimator_  # pipeline: preprocessor + RF

TEMPLATE_PATH = "data/processed/weatherAUS-data-engineered.csv"
_template_df = pd.read_csv(TEMPLATE_PATH)
_template_row = _template_df.iloc[0:1].copy()
_template_row.drop(columns=["RainToday"], inplace=True, errors="ignore")

# ----- 2. Define request schema -----

class WeatherInput(BaseModel):
    """
    Minimal set of raw features needed to reconstruct the engineered feature set
    used by the model. Values are examples for Melbourne.
    """
    Location: Literal["Melbourne", "MelbourneAirport", "Watsonia"] = Field(
        ..., description="Weather station location"
    )
    Season: Literal["Summer", "Autumn", "Winter", "Spring"] = Field(
        ..., description="Season of the year"
    )

    MinTemp: float = Field(..., description="Minimum temperature (°C)")
    MaxTemp: float = Field(..., description="Maximum temperature (°C)")
    Temp9am: float = Field(..., description="Temperature at 9am (°C)")
    Temp3pm: float = Field(..., description="Temperature at 3pm (°C)")

    Rainfall: float = Field(..., description="Amount of rainfall (mm)")

    Humidity9am: float = Field(..., description="Humidity at 9am (%)")
    Humidity3pm: float = Field(..., description="Humidity at 3pm (%)")

    WindSpeed9am: float = Field(..., description="Wind speed at 9am (km/h)")
    WindSpeed3pm: float = Field(..., description="Wind speed at 3pm (km/h)")

    Pressure9am: float = Field(..., description="Pressure at 9am (hPa)")
    Pressure3pm: float = Field(..., description="Pressure at 3pm (hPa)")

    RainYesterday: Literal["Yes", "No"] = Field(
        ..., description="Whether it rained at least 1mm yesterday (Yes/No)"
    )


# ----- 3. Create FastAPI app -----

app = FastAPI(
    title="Weather Wise - Rain Prediction API",
    description=(
        "API that wraps the Random Forest model from the Weather Wise project. "
        "Given today's weather conditions in the Melbourne area, it predicts "
        "whether it will rain today and the associated probability."
    ),
    version="1.0.0",
)


# ----- 4. Helper: build model-ready DataFrame -----

def build_feature_dataframe(payload: WeatherInput) -> pd.DataFrame:
    data = payload.dict()

    df = _template_row.copy()

    cols_to_fill = [
        "Location", "Season",
        "MinTemp", "MaxTemp", "Temp9am", "Temp3pm",
        "Rainfall",
        "Humidity9am", "Humidity3pm",
        "WindSpeed9am", "WindSpeed3pm",
        "Pressure9am", "Pressure3pm",
        "RainYesterday"
    ]

    for col in cols_to_fill:
        if col in df.columns:
            df[col] = data[col]

    # Recalcular engineered features como en el notebook
    if set(["MaxTemp", "MinTemp"]).issubset(df.columns):
        df["TempDiff"] = df["MaxTemp"] - df["MinTemp"]
    if set(["Temp3pm", "Temp9am"]).issubset(df.columns):
        df["TempChange"] = df["Temp3pm"] - df["Temp9am"]

    if set(["Pressure3pm", "Pressure9am"]).issubset(df.columns):
        df["PressureDiff"] = df["Pressure3pm"] - df["Pressure9am"]

    if set(["Humidity3pm", "Humidity9am"]).issubset(df.columns):
        df["HumidityDiff"] = df["Humidity3pm"] - df["Humidity9am"]
        df["AvgHumidity"] = df[["Humidity9am", "Humidity3pm"]].mean(axis=1)

    if set(["WindSpeed3pm", "WindSpeed9am"]).issubset(df.columns):
        df["WindSpeedDiff"] = df["WindSpeed3pm"] - df["WindSpeed9am"]

    if set(["Temp9am", "Temp3pm"]).issubset(df.columns):
        df["AvgTemp"] = df[["Temp9am", "Temp3pm"]].mean(axis=1)

    if "Sunshine" in df.columns:
        df["RainfallPerSunshine"] = df["Rainfall"] / (df["Sunshine"] + 0.1)
    else:
        df["RainfallPerSunshine"] = df["Rainfall"] / 0.1

    return df


# ----- 5. Prediction endpoint -----

@app.post("/predict", summary="Predict whether it will rain today")
def predict_rain(input_data: WeatherInput):
    """
    Given weather conditions for a day in the Melbourne area, return:
    - binary prediction: 'Rain' or 'No Rain'
    - probability of rain (between 0 and 1)
    """

    # Build model-ready DataFrame
    X = build_feature_dataframe(input_data)

    # Predict with full pipeline (preprocessing + RF)
    proba = best_model.predict_proba(X)[0, 1]
    pred_label = best_model.predict(X)[0]   # probability of 'Rain' ('Yes')
    
    # Map 'Yes'/'No' to more friendly labels
    human_readable = "Rain" if pred_label == "Yes" else "No Rain"

    return {
        "prediction": human_readable,
        "prediction_raw": pred_label,
        "probability_rain": round(float(proba), 4),
        "threshold_used": 0.5,
        "interpretation": (
            "High chance of rain; consider adjusting plans."
            if proba >= 0.5
            else "Low estimated risk of rain; conditions are mostly dry."
        ),
    }


# ----- 6. Local dev entrypoint -----

if __name__ == "__main__":
    # This allows: python api/main.py
    uvicorn.run(app, host="0.0.0.0", port=8000)