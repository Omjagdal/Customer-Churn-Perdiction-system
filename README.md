📊 Customer Churn Prediction System
A comprehensive Machine Learning-powered web application that predicts customer churn probability using advanced ML algorithms and an intuitive web interface.
🎯 Overview
This system analyzes customer data across multiple dimensions—demographics, usage patterns, subscription details, service history, and engagement metrics—to predict the likelihood of customer churn. Built with Scikit-Learn and Streamlit, it provides real-time predictions with probability scores to help businesses proactively retain customers.
✨ Key Features




Create a virtual environment (recommended)

bashpython -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

Install dependencies

bashpip install -r requirements.txt

Train the model (if not pre-trained)

bashpython train_model.py

Run the application

bashstreamlit run app.py
The app will open in your browser at http://localhost:8501
📊 Dataset Features
The model uses the following customer attributes:
Demographics

Age, Gender, Location
Account tenure

Usage Patterns

Monthly usage hours
Login frequency
Feature utilization

Subscription Details

Plan type (Basic/Standard/Premium)
Contract length
Monthly charges

Service History

Support tickets opened
Service issues reported
Resolution time

Engagement Metrics

Last interaction date
Product adoption rate
Email/notification engagement

🔧 Model Details
Algorithm: Random Forest Classifier / Gradient Boosting
Performance Metrics:

Accuracy: ~85%
Precision: ~82%
Recall: ~78%
F1-Score: ~80%
ROC-AUC: ~0.88

Preprocessing Pipeline:

Categorical encoding (OneHot/Label encoding)
Numerical feature scaling (StandardScaler)
Missing value imputation (Mean/Mode strategies)
Feature engineering and selection

💻 Usage
Single Customer Prediction

Open the web app
Fill in customer details in the sidebar form
Click "Predict Churn"
View prediction result and probability score

Batch Prediction

Navigate to "Batch Prediction" tab
Upload a CSV file with customer data
Download predictions with probability scores

API Integration (Optional)
pythonimport requests
import json

url = "http://localhost:8501/predict"
data = {
    "age": 35,
    "tenure": 12,
    "monthly_charges": 89.99,
    # ... other features
}

response = requests.post(url, json=data)
prediction = response.json()
print(f"Churn Probability: {prediction['probability']}")
📈 Model Training
To retrain the model with new data:
bashpython train_model.py --data data/new_training_data.csv --output model.pkl
Training parameters can be customized in train_model.py:

Train-test split ratio
Hyperparameters (n_estimators, max_depth, etc.)
Cross-validation folds
Feature selection methods

🛠️ Tech Stack

ML Framework: Scikit-Learn
Web Framework: Streamlit
Data Processing: Pandas, NumPy
Visualization: Matplotlib, Plotly, Seaborn
Model Serialization: Pickle/Joblib
