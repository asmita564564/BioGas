#Multiple output code

from fastapi import FastAPI
import uvicorn
import pandas as pd
import numpy as np
import pickle
from pydantic import BaseModel
from typing import List, Optional

# Load the trained model
with open('biogas_model_3.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

# Initialize FastAPI
app = FastAPI()

# Define the data structure for incoming requests
class BiogasData(BaseModel):
    Date: Optional[float]
    WASTE_SMC: Optional[float]
    MSW_REC: Optional[float]
    DIG_FEED_B: Optional[float]
    DIG_FEED_B_1: Optional[float]
    DISPOSAL_A: Optional[float]
    DISPOSAL_B: Optional[float]
    DIG_PRESS: Optional[float]
    DIG_LEVEL: Optional[float]
    BALLOON_A: Optional[float]
    BALLOON_B: Optional[float]
    ENGINE_RUNNING_HRS: Optional[float]
    TOTAL_POWER_GENE: Optional[float]
    EXPORT_POWER: Optional[float]
    GAS_CONSUMPTION: Optional[float]
    SFC: Optional[float]
    Raw_MSW_TS: Optional[float]
    Raw_MSW_VS: Optional[float]
    Raw_MSW_MC: Optional[float]
    AC_02_TS: Optional[float]
    AC_02_VS: Optional[float]
    AC_02_MC: Optional[float]
    AC_02_Sand: Optional[float]
    AC_02_CN: Optional[float]
    AC_02_COD: Optional[float]
    Digester_Feed_TS: Optional[float]
    Digester_Feed_VS: Optional[float]
    Digester_Feed_MC: Optional[float]
    Digester_Feed_pH: Optional[float]
    Digester_Feed_VFA: Optional[float]
    Digester_Feed_ALK: Optional[float]
    Digester_Feed_VA: Optional[float]
    Digester_Feed_EC: Optional[float]
    Digester_Feed_Temp: Optional[float]
    Digester_Feed_COD: Optional[float]
    Digester_Recycle_TS: Optional[float]
    Digester_Recycle_VS: Optional[float]
    Digester_Recycle_MC: Optional[float]
    Digester_Recycle_pH: Optional[float]
    Digester_Recycle_VFA: Optional[float]
    Digester_Recycle_ALK: Optional[float]
    Digester_Recycle_VA: Optional[float]
    Digester_Recycle_EC: Optional[float]
    Digester_Recycle_Temp: Optional[float]
    Digester_Recycle_TOC: Optional[float]
    Biogas_Digester_B_Outlet_Methane_pct: Optional[float]
    Biogas_Digester_B_Outlet_H2S_ppm: Optional[float]
    Biogas_Clean_Biogas_Methane_pct: Optional[float]
    Biogas_Clean_Biogas_H2S_ppm: Optional[float]
    Biogas_Dosing_pH: Optional[float]
    Biogas_Scrubber_pH: Optional[float]
    

# API root
@app.get("/")
def read_root():
    return {"message": "Welcome to the Biogas Prediction API"}

# Prediction endpoint
@app.post("/predict")
def predict(data: BiogasData):
    # Extract data from the request
    var_list = [
        data.Date, data.WASTE_SMC, data.MSW_REC, data.DIG_FEED_B, data.DIG_FEED_B_1, data.DISPOSAL_A, data.DISPOSAL_B, 
        data.DIG_PRESS, data.DIG_LEVEL, data.BALLOON_A, data.BALLOON_B, data.ENGINE_RUNNING_HRS, data.TOTAL_POWER_GENE, 
        data.EXPORT_POWER, data.GAS_CONSUMPTION, data.SFC, data.Raw_MSW_TS, data.Raw_MSW_VS, data.Raw_MSW_MC, data.AC_02_TS, 
        data.AC_02_VS, data.AC_02_MC, data.AC_02_Sand, data.AC_02_CN, data.AC_02_COD, data.Digester_Feed_TS, data.Digester_Feed_VS, 
        data.Digester_Feed_MC, data.Digester_Feed_pH, data.Digester_Feed_VFA, data.Digester_Feed_ALK, data.Digester_Feed_VA, 
        data.Digester_Feed_EC, data.Digester_Feed_Temp, data.Digester_Feed_COD, data.Digester_Recycle_TS, data.Digester_Recycle_VS, 
        data.Digester_Recycle_MC, data.Digester_Recycle_pH, data.Digester_Recycle_VFA, data.Digester_Recycle_ALK, data.Digester_Recycle_VA, 
        data.Digester_Recycle_EC, data.Digester_Recycle_Temp, data.Digester_Recycle_TOC, data.Biogas_Digester_B_Outlet_Methane_pct, 
        data.Biogas_Digester_B_Outlet_H2S_ppm, data.Biogas_Clean_Biogas_Methane_pct, data.Biogas_Clean_Biogas_H2S_ppm, 
        data.Biogas_Dosing_pH, data.Biogas_Scrubber_pH
    ]

    # Predict using the model
    prediction = model.predict([var_list])[0]

    # Define the feature names for the 10 outputs
    output_features = [
        "Digester_Disposal_TS", "Digester_Disposal_VS", "Digester_Disposal_MC", 
        "Digester_Disposal_pH", "Digester_Disposal_VFA", "Digester_Disposal_ALK", 
        "Digester_Disposal_VA", "Digester_Disposal_EC", "Digester_Disposal_Temp", 
        "Digester_Disposal_TOC"
    ]

    # Create a dictionary mapping feature names to predicted values
    prediction_dict = {feature: value for feature, value in zip(output_features, prediction)}

    # Return the result as a dictionary
    return prediction_dict

# Run the app
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
