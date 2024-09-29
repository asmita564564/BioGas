
# from fastapi import FastAPI
# import pickle
# from pydantic import BaseModel

# # Initialize FastAPI app
# app = FastAPI()

# # Load the trained model
# pickle_in = open("classifier.pkl", "rb")
# classifier = pickle.load(pickle_in)

# # Define the input data model
# class InputData(BaseModel):
#     variance: float
#     skewness: float
#     curtosis: float
#     entropy: float

# # Define the prediction endpoint
# @app.post("/predict")
# def predict(data: InputData):
#     # Extract features from input data
#     features = [[data.variance, data.skewness, data.curtosis, data.entropy]]
#     # Make prediction
#     prediction = classifier.predict(features)
#     # Return the result
#     return {"prediction": int(prediction[0])}

# # Root endpoint
# @app.get("/")
# def read_root():
#     return {"message": "Welcome to the Random Forest Classifier API"}

# -*- coding: utf-8 -*-


# # 1. Library imports
# import uvicorn
# from fastapi import FastAPI, Depends
# from fastapi.security import HTTPBasic, HTTPBasicCredentials
# from BankNotes import BankNote
# import numpy as np
# import pickle
# import pandas as pd
# from typing import Optional
# from starlette.responses import JSONResponse
# import secrets

# # Initialize FastAPI app
# app = FastAPI()

# # Load the trained model
# pickle_in = open("classifier.pkl", "rb")
# classifier = pickle.load(pickle_in)

# # Initialize HTTPBasic for authentication
# security = HTTPBasic()

# # Hardcoded username and password (for demonstration purposes)
# USERNAME = "admin"
# PASSWORD = "password"

# # Function to verify credentials
# def authenticate(credentials: HTTPBasicCredentials = Depends(security)):
#     correct_username = secrets.compare_digest(credentials.username, USERNAME)
#     correct_password = secrets.compare_digest(credentials.password, PASSWORD)
#     if not (correct_username and correct_password):
#         return JSONResponse(
#             status_code=401,
#             content={"message": "Invalid authentication credentials"},
#         )
#     return credentials.username

# # 3. Index route, opens automatically on http://127.0.0.1:8000
# @app.get('/')
# def index(username: str = Depends(authenticate)):
#     return {'message': f'Hello, {username}'}

# # 4. Name route, opens on http://127.0.0.1:8000/{name}
# @app.get('/{name}')
# def get_name(name: str, username: str = Depends(authenticate)):
#     return {'best': f'{name}', 'authenticated_user': f'{username}'}

# # 5. Expose the prediction functionality
# @app.post('/predict')
# def predict_banknote(data: BankNote, username: str = Depends(authenticate)):
#     data = data.dict()
#     variance = data['variance']
#     skewness = data['skewness']
#     curtosis = data['curtosis']
#     entropy = data['entropy']
#     prediction = classifier.predict([[variance, skewness, curtosis, entropy]])
    
#     if prediction[0] > 0.5:
#         prediction = "Fake note"
#     else:
#         prediction = "It's a Bank note"
    
#     return {
#         'authenticated_user': username,
#         'prediction': prediction
#     }

# # 6. Run the API with uvicorn
# if __name__ == '__main__':
#     uvicorn.run(app, host='127.0.0.1', port=8000)



# # 1. Library imports
# import uvicorn
# from fastapi import FastAPI
# from BankNotes import BankNote
# import numpy as np
# import pickle
# import pandas as pd

# app = FastAPI()
# pickle_in = open("classifier.pkl","rb")
# classifier=pickle.load(pickle_in)

# # 3. Index route, opens automatically on http://127.0.0.1:8000
# @app.get('/')
# def index():
#     return {'message': 'Hello, World'}

# @app.get('/{name}')
# def get_name(name: str):
#     return {'best': f'{name}'}

# # 3. Expose the prediction functionality, make a prediction from the passed
# #    JSON data and return the predicted Bank Note with the confidence
# @app.post('/predict')
# def predict_banknote(data:BankNote):
#     data = data.dict()
#     variance=data['variance']
#     skewness=data['skewness']
#     curtosis=data['curtosis']
#     entropy=data['entropy']
#    # print(classifier.predict([[variance,skewness,curtosis,entropy]]))
#     prediction = classifier.predict([[variance,skewness,curtosis,entropy]])
#     if(prediction[0]>0.5):
#         prediction="Fake note"
#     else:
#         prediction="Its a Bank note"
#     return {
#         'prediction': prediction
#     }

# # 5. Run the API with uvicorn

# if __name__ == '__main__':
#     uvicorn.run(app, host='127.0.0.1', port=8000)
    


# from fastapi import FastAPI
# import uvicorn
# import pandas as pd
# import numpy as np
# import pickle
# from pydantic import BaseModel
# from typing import List, Optional

# # Load the trained model
# with open('biogas_model_2.pkl', 'rb') as model_file:
#     model = pickle.load(model_file)

# # Initialize FastAPI
# app = FastAPI()

# # Define the data structure for incoming requests
# class BiogasData(BaseModel):
#     Date: Optional[float]
#     WASTE_SMC: Optional[float]
#     MSW_REC: Optional[float]
#     DIG_FEED_1: Optional[float]
#     Engine_Balloon_A: Optional[float]
#     Engine_Balloon_B: Optional[float]
#     DIG_FEED1_Squared: Optional[float]
#     Engine_Balloon_A_Squared: Optional[float]
#     Exp_ENGINE_RUNNING_HRS: Optional[float]
#     biogas_digester_b_outlet_h2s_ppm: Optional[float]
#     biogas_clean_biogas_methane_pct: Optional[float]
#     biogas_clean_biogas_h2s_ppm: Optional[float]
#     biogas_clean_biogas_Dosing_pH: Optional[float]
#     biogas_clean_biogas_Scrubber_pH: Optional[float]
#     DIG_FEED1_Balloon_A_interaction: Optional[float]
#     biogas_methane_squared: Optional[float]
#     biogas_methane_to_h2s_ratio: Optional[float]
#     biogas_methane_h2s_diff: Optional[float]
#     biogas_digester_b_outlet_methane :Optional[float]
#     log_biogas_methane: Optional[float]
#     rolling_avg_methane: Optional[float]
#     TS: Optional[float]
#     VS: Optional[float]
#     MC: Optional[float]
#     pH: Optional[float]
#     VFA: Optional[float]
#     ALK: Optional[float]
#     VA: Optional[float]
#     EC: Optional[float]
#     Temp: Optional[float]
#     TOC: Optional[float]
#     DIG_FEED:Optional[float]

# # API root
# @app.get("/")
# def read_root():
#     return {"message": "Welcome to the Biogas Prediction API"}

# # Prediction endpoint
# @app.post("/predict")
# def predict(data: BiogasData):
#     Date=data.Date
#     WASTE_SMC=data.WASTE_SMC
#     MSW_REC=data.MSW_REC
#     DIG_FEED_1=data.DIG_FEED_1
#     Engine_Balloon_A=data.Engine_Balloon_A
#     Engine_Balloon_B=data.Engine_Balloon_B
#     DIG_FEED1_Squared=data.DIG_FEED1_Squared
#     Engine_Balloon_A_Squared=data.Engine_Balloon_A_Squared
#     Exp_ENGINE_RUNNING_HRS=data.Exp_ENGINE_RUNNING_HRS
#     biogas_digester_b_outlet_h2s_ppm=data.biogas_digester_b_outlet_h2s_ppm
#     biogas_clean_biogas_methane_pct=data.biogas_clean_biogas_methane_pct
#     biogas_clean_biogas_h2s_ppm=data.biogas_clean_biogas_h2s_ppm
#     biogas_clean_biogas_Dosing_pH=data.biogas_clean_biogas_Dosing_pH
#     biogas_clean_biogas_Scrubber_pH=data.biogas_clean_biogas_Scrubber_pH
#     DIG_FEED1_Balloon_A_interaction=data.DIG_FEED1_Balloon_A_interaction
#     biogas_methane_squared=data.biogas_methane_squared
#     biogas_methane_to_h2s_ratio=data.biogas_methane_to_h2s_ratio
#     biogas_methane_h2s_diff=data.biogas_methane_h2s_diff
#     biogas_digester_b_outlet_methane=data.biogas_digester_b_outlet_methane
#     log_biogas_methane=data.log_biogas_methane
#     rolling_avg_methane=data.rolling_avg_methane
#     TS=data.TS
#     VS=data.VS
#     MC=data.MC
#     pH=data.pH
#     VFA=data.VFA
#     ALK=data.ALK
#     VA=data.VA
#     EC=data.EC
#     Temp=data.Temp
#     TOC=data.TOC
#     DIG_FEED=data.DIG_FEED

#     var_list=[Date,WASTE_SMC,MSW_REC,DIG_FEED_1,Engine_Balloon_A,Engine_Balloon_B,DIG_FEED1_Squared,Engine_Balloon_A_Squared,
#              Exp_ENGINE_RUNNING_HRS,biogas_digester_b_outlet_h2s_ppm,biogas_clean_biogas_methane_pct,biogas_clean_biogas_h2s_ppm,
#                biogas_clean_biogas_Dosing_pH,biogas_clean_biogas_Scrubber_pH,DIG_FEED1_Balloon_A_interaction,biogas_methane_squared,
#                biogas_methane_to_h2s_ratio,biogas_methane_h2s_diff,biogas_digester_b_outlet_methane,log_biogas_methane,rolling_avg_methane,TS,VS,MC,pH,VFA,ALK,VA,EC,Temp,TOC,DIG_FEED]
    

#     # print(model.predict([var_list]))
#     # prediction = model.predict([var_list])
#     # Convert the incoming data to a DataFrame
#     # input_data = pd.DataFrame([data.dict()])

#     # Convert 'Date' to a Unix timestamp
#     # input_data['Date'] = pd.to_datetime(input_data['Date'], errors='coerce').astype(np.int64) // 10**9

#     #return var_list
#     # # Make a prediction
#     prediction = model.predict([var_list])
    
#     #return {"prediction": prediction[0]}
#     prediction = str(prediction[0]) 
#     # if prediction == '61':
#     #     return 'healthy {}'.format(prediction)
#     # else:
#     #     return 'unhealthy {}'.format(prediction)
#     print(prediction)
#     return prediction

# # Run the app
# if __name__ == "__main__":
#     uvicorn.run(app, host="127.0.0.1", port=8000)




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
