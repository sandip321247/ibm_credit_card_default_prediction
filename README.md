# ibm_credit_card_default_prediction

Overview

The Credit Card Default Prediction project uses IBM Watson Machine Learning to predict whether a customer will default on their credit card payments. This helps financial institutions assess credit risk and make informed lending decisions.

Features

Utilizes IBM Watson's Machine Learning API for prediction.

Takes customer financial data as input.

Returns a probability of default.

Secure API authentication using IBM Cloud IAM.

Prerequisites

Before running the code, ensure you have:

An IBM Cloud account.

API key for authentication.

Python installed on your system.

The requests library installed (pip install requests).

Installation

Clone this repository:

git clone https://github.com/your-repo/credit-card-default-prediction.git
cd credit-card-default-prediction

Install dependencies:

pip install requests

Usage

Replace API_KEY with your IBM Watson Machine Learning API Key in the script.

Define your input data by replacing array_of_input_fields and array_of_values_to_be_scored.

Run the script:

python predict.py

The script will return a scoring response with the prediction results.

Code Explanation

The script performs the following steps:

Authenticates with IBM Watson Machine Learning using an API Key.

Generates a Bearer Token for authorization.

Sends a prediction request to the IBM Watson ML API with input data.

Receives and prints the prediction results.

Example Response

{
  "predictions": [
    {
      "values": [[0, 0.85]]
    }
  ]
}

Here, 0 indicates no default, and 0.85 represents the model's confidence.

Future Enhancements

Integrate with a web app for user-friendly access.

Improve prediction accuracy using advanced ML models.

Enable real-time risk assessment for financial institutions.

License

This project is licensed under the MIT License.

Feel free to contribute and improve this project!
