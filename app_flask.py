 # from flask import Flask, request, jsonify
# import pickle
# import numpy as np

# # Load the trained model
# with open('biogas_model_3.pkl', 'rb') as model_file:
#     model = pickle.load(model_file)

# # Initialize Flask app
# app = Flask(__name__)

# # Define the prediction endpoint
# @app.route('/predict', methods=['POST'])
# def predict():
#     data = request.json

#     # Extract features from the request
#     features = [
#         data.get('Date'),
#         data.get('WASTE_SMC'),
#         data.get('MSW_REC'),
#         data.get('DIG_FEED_B'),
#         data.get('DIG_FEED_B_1'),
#         data.get('DISPOSAL_A'),
#         data.get('DISPOSAL_B'),
#         data.get('DIG_PRESS'),
#         data.get('DIG_LEVEL'),
#         data.get('BALLOON_A'),
#         data.get('BALLOON_B'),
#         data.get('ENGINE_RUNNING_HRS'),
#         data.get('TOTAL_POWER_GENE'),
#         data.get('EXPORT_POWER'),
#         data.get('GAS_CONSUMPTION'),
#         data.get('SFC'),
#         data.get('Raw_MSW_TS'),
#         data.get('Raw_MSW_VS'),
#         data.get('Raw_MSW_MC'),
#         data.get('AC_02_TS'),
#         data.get('AC_02_VS'),
#         data.get('AC_02_MC'),
#         data.get('AC_02_Sand'),
#         data.get('AC_02_CN'),
#         data.get('AC_02_COD'),
#         data.get('Digester_Feed_TS'),
#         data.get('Digester_Feed_VS'),
#         data.get('Digester_Feed_MC'),
#         data.get('Digester_Feed_pH'),
#         data.get('Digester_Feed_VFA'),
#         data.get('Digester_Feed_ALK'),
#         data.get('Digester_Feed_VA'),
#         data.get('Digester_Feed_EC'),
#         data.get('Digester_Feed_Temp'),
#         data.get('Digester_Feed_COD'),
#         data.get('Digester_Recycle_TS'),
#         data.get('Digester_Recycle_VS'),
#         data.get('Digester_Recycle_MC'),
#         data.get('Digester_Recycle_pH'),
#         data.get('Digester_Recycle_VFA'),
#         data.get('Digester_Recycle_ALK'),
#         data.get('Digester_Recycle_VA'),
#         data.get('Digester_Recycle_EC'),
#         data.get('Digester_Recycle_Temp'),
#         data.get('Digester_Recycle_TOC'),
#         data.get('Biogas_Digester_B_Outlet_Methane_pct'),
#         data.get('Biogas_Digester_B_Outlet_H2S_ppm'),
#         data.get('Biogas_Clean_Biogas_Methane_pct'),
#         data.get('Biogas_Clean_Biogas_H2S_ppm'),
#         data.get('Biogas_Dosing_pH'),
#         data.get('Biogas_Scrubber_pH')
#     ]

#     # Predict using the model
#     prediction = model.predict([features])

#     # Return prediction result
#     return jsonify({"prediction": str(prediction[0])})

# # Root endpoint
# @app.route('/')
# def home():
#     return "Welcome to the Biogas Prediction API using Flask"

# # Run the app
# if __name__ == "__main__":
#     app.run(debug=True)



from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np

# Load the trained model
with open('biogas_model_3.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

# Initialize Flask app
app = Flask(__name__)

# Define the prediction endpoint
@app.route('/predict', methods=['POST'])
def predict():
    data = request.json

    # Extract features from the request
    features = [
        data.get('Date'),
        data.get('WASTE_SMC'),
        data.get('MSW_REC'),
        data.get('DIG_FEED_B'),
        data.get('DIG_FEED_B_1'),
        data.get('DISPOSAL_A'),
        data.get('DISPOSAL_B'),
        data.get('DIG_PRESS'),
        data.get('DIG_LEVEL'),
        data.get('BALLOON_A'),
        data.get('BALLOON_B'),
        data.get('ENGINE_RUNNING_HRS'),
        data.get('TOTAL_POWER_GENE'),
        data.get('EXPORT_POWER'),
        data.get('GAS_CONSUMPTION'),
        data.get('SFC'),
        data.get('Raw_MSW_TS'),
        data.get('Raw_MSW_VS'),
        data.get('Raw_MSW_MC'),
        data.get('AC_02_TS'),
        data.get('AC_02_VS'),
        data.get('AC_02_MC'),
        data.get('AC_02_Sand'),
        data.get('AC_02_CN'),
        data.get('AC_02_COD'),
        data.get('Digester_Feed_TS'),
        data.get('Digester_Feed_VS'),
        data.get('Digester_Feed_MC'),
        data.get('Digester_Feed_pH'),
        data.get('Digester_Feed_VFA'),
        data.get('Digester_Feed_ALK'),
        data.get('Digester_Feed_VA'),
        data.get('Digester_Feed_EC'),
        data.get('Digester_Feed_Temp'),
        data.get('Digester_Feed_COD'),
        data.get('Digester_Recycle_TS'),
        data.get('Digester_Recycle_VS'),
        data.get('Digester_Recycle_MC'),
        data.get('Digester_Recycle_pH'),
        data.get('Digester_Recycle_VFA'),
        data.get('Digester_Recycle_ALK'),
        data.get('Digester_Recycle_VA'),
        data.get('Digester_Recycle_EC'),
        data.get('Digester_Recycle_Temp'),
        data.get('Digester_Recycle_TOC'),
        data.get('Biogas_Digester_B_Outlet_Methane_pct'),
        data.get('Biogas_Digester_B_Outlet_H2S_ppm'),
        data.get('Biogas_Clean_Biogas_Methane_pct'),
        data.get('Biogas_Clean_Biogas_H2S_ppm'),
        data.get('Biogas_Dosing_pH'),
        data.get('Biogas_Scrubber_pH')
    ]

    # Perform prediction using the model

    prediction = model.predict([features])

    # Return prediction result
    prediction=prediction[0].tolist()
    return jsonify({"prediction": prediction})

# Root endpoint to serve the HTML file
@app.route('/')
def home():
    return render_template('index.html')

# Run the app
if __name__ == "__main__":
    app.run(debug=True)

