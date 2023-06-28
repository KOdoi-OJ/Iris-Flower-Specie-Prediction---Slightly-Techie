# ----- Load base libraries and packages
import gradio as gr

import numpy as np
import pandas as pd

import os
import pickle

import xgboost as xgb
from xgboost import XGBClassifier


# ----- Useful lists
expected_inputs = ["sepal_length", "sepal_width", "petal_length", "petal_width"]


# ----- Helper Functions
# Function to load ML toolkit
def load_ml_toolkit(file_path= r"src\Iris_App_toolkit"):
    """
    This function loads the ML items into this file. It takes the path to the ML items to load it.

    Args:
        file_path (regexp, optional): It receives the file path to the ML items, but defaults to the "src" folder in the repository. The full default relative path is r"src\Iris_App_toolkit".

    Returns:
        file: It returns the pickle file (which in this case contains the Machine Learning items.)
    """

    with open(file_path, "rb") as file:
        loaded_toolkit = pickle.load(file)
    return loaded_toolkit


# Importing the toolkit
loaded_toolkit = load_ml_toolkit()
scaler = loaded_toolkit["scaler"]

# Import the model
model = XGBClassifier()
model.load_model(r"src\xgb_model.json")


# Function to process inputs and return prediction
def process_and_predict(*args, scaler=scaler, model=model):
    """
    This function processes the inputs and returns the predicted specie of the flower
    It receives the user inputs, scaler and model. The inputs are then put through the same process as was done during modelling

    Args:
        scaler (MinMaxScaler, optional): It is the scaler (MinMaxScaler) used to scale the numeric features before training the model, and should be loaded either as part of the ML Items or as a standalone item. Defaults to scaler, which comes with the ML Items dictionary.
        model (XGBoost, optional): This is the model that was trained and is to be used for the prediction. Since XGBoost seems to have issues with Pickle, import as a standalone. It defaults to "model", as loaded.

    Returns:
        Prediction (label): Returns the label of the predicted class, i.e. the specie of the flower
    """

    # Convert inputs into a DataFrame
    input_data = pd.DataFrame([args], columns=expected_inputs)

    # Scale the numeric columns
    input_data[expected_inputs] = scaler.transform(input_data[expected_inputs])

    # Make the prediction
    model_output = model.predict_proba(input_data)
    setosa_prob = float(model_output[0][0])
    versicolor_prob = float(model_output[0][1])
    virginica_prob = 1 - (setosa_prob + versicolor_prob)
    return {"Prediction: Iris-setosa": setosa_prob, "Prediction: Iris-versicolor": versicolor_prob, "Prediction: Iris-virginica": virginica_prob}


# ----- App Interface
# Inputs
sepal_length = gr.Slider(label="Sepal length (cm)", minimum=4.3, step=0.1, maximum= 7.9, interactive=True, value=5.8)
sepal_width = gr.Slider(label="Sepal width (cm)", minimum=2, step=0.1, maximum= 4.4, interactive=True, value=3)
petal_length = gr.Slider(label="Petal length (cm)", minimum=1, step=0.05, maximum= 4.9, interactive=True, value=4.35)
petal_width = gr.Slider(label="Petal width (cm)", minimum=0.1, step=0.05, maximum= 2.5, interactive=True, value=1.3)


# Output
gr.Interface(inputs=[sepal_length, sepal_width, petal_length, petal_width], outputs=gr.Label("Awaiting Submission..."), fn=process_and_predict, title="Iris Flower Specie Prediction App",
             description="""This app uses a machine learning model to predict the specie of an Iris flower based on inputs made by you, the user. The (XGBoost) model was trained and built based on the Iris flower Dataset""").launch(inbrowser=True, show_error=True)
