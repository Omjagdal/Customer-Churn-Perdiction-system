# 🎯 Customer Churn Prediction System

A machine learning-powered web application for predicting customer churn with high accuracy. Built with Streamlit and Scikit-Learn, this system helps businesses identify at-risk customers and take proactive retention measures.

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)](https://streamlit.io/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-orange.svg)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## 📋 Table of Contents

- [Features](#features)
- [Model Performance](#model-performance)
- [Installation](#installation)
- [Usage](#usage)
- [Data Requirements](#data-requirements)
- [API Integration](#api-integration)
- [Project Structure](#project-structure)
- [Technologies Used](#technologies-used)
- [Contributing](#contributing)
- [License](#license)

## ✨ Features

### Customer Analytics
- **Demographic Analysis**: Age, location, and customer profile insights
- **Subscription Tracking**: Plan types, contract lengths, and pricing analysis
- **Service History**: Support ticket tracking and resolution metrics
- **Engagement Metrics**: Interaction dates, product adoption, and communication engagement

### Prediction Capabilities
- **Single Customer Prediction**: Real-time churn risk assessment
- **Batch Processing**: Bulk predictions for entire customer databases
- **Probability Scoring**: Confidence levels for each prediction
- **Interactive Dashboard**: User-friendly web interface

### Advanced Features
- **Feature Importance Visualization**: Understand key churn drivers
- **Data Preprocessing Pipeline**: Automated data cleaning and transformation
- **Model Explainability**: Insights into prediction reasoning

## 📊 Model Performance

### Algorithm
- **Primary Model**: Random Forest Classifier / Gradient Boosting
- **Ensemble Methods**: Multiple model comparison and selection

### Performance Metrics
| Metric | Score |
|--------|-------|
| **Accuracy** | ~85% |
| **Precision** | ~82% |
| **Recall** | ~78% |
| **F1-Score** | ~80% |
| **ROC-AUC** | ~0.88 |

### Preprocessing Pipeline
- Categorical encoding (OneHot/Label encoding)
- Numerical feature scaling (StandardScaler)
- Missing value imputation (Mean/Mode strategies)
- Feature engineering and selection
- Outlier detection and handling

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Installation Steps

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/customer-churn-prediction.git
cd customer-churn-prediction
```

2. **Create a virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Run the application**
```bash
streamlit run app.py
```

The application will open in your default browser at `http://localhost:8501`

### Docker Deployment (Optional)
```bash
docker build -t churn-prediction .
docker run -p 8501:8501 churn-prediction
```

## 💻 Usage

### Single Customer Prediction

1. Open the web application
2. Fill in customer details in the sidebar form:
   - **Demographics**: Age, gender, location
   - **Login Frequency**: Daily/weekly usage patterns
   - **Feature Utilization**: Service usage metrics
   - **Subscription Details**: Plan type, contract length, monthly charges
   - **Service History**: Support tickets, issues, resolution time
   - **Engagement Metrics**: Last interaction, adoption rate, engagement score
3. Click **"Predict Churn"**
4. View prediction result and probability score

### Batch Prediction

1. Navigate to the **"Batch Prediction"** tab
2. Upload a CSV file with customer data
3. System processes all records automatically
4. Download predictions with probability scores

### Required CSV Format
```csv
customer_id,age,tenure,monthly_charges,login_frequency,plan_type,contract_length,support_tickets,last_interaction_days,engagement_score
C001,35,12,89.99,15,Premium,24,2,5,0.82
C002,42,6,49.99,8,Basic,12,5,30,0.45
```

## 📁 Data Requirements

### Input Features

| Feature Category | Features | Type |
|-----------------|----------|------|
| **Demographics** | Age, Gender, Location | Numerical/Categorical |
| **Login Activity** | Login Frequency, Session Duration | Numerical |
| **Feature Usage** | Features Used, Usage Intensity | Numerical |
| **Subscription** | Plan Type, Contract Length, Monthly Charges | Categorical/Numerical |
| **Service History** | Support Tickets, Issues Reported, Resolution Time | Numerical |
| **Engagement** | Last Interaction Date, Adoption Rate, Email Engagement | Numerical/Date |

### Sample Dataset Structure
```python
{
    'age': 35,
    'tenure': 12,
    'monthly_charges': 89.99,
    'login_frequency': 15,
    'plan_type': 'Premium',
    'contract_length': 24,
    'support_tickets': 2,
    'resolution_time': 24.5,
    'last_interaction_days': 5,
    'adoption_rate': 0.75,
    'email_engagement': 0.82
}
```

## 🔌 API Integration

### REST API Usage
```python
import requests
import json

# API endpoint
url = "http://localhost:8501/predict"

# Customer data
data = {
    "age": 35,
    "tenure": 12,
    "monthly_charges": 89.99,
    "login_frequency": 15,
    "plan_type": "Premium",
    "contract_length": 24,
    "support_tickets": 2,
    "resolution_time": 24.5,
    "last_interaction_days": 5,
    "adoption_rate": 0.75,
    "email_engagement": 0.82
}

# Make prediction request
response = requests.post(url, json=data)
result = response.json()

print(f"Churn Prediction: {result['prediction']}")
print(f"Probability: {result['probability']:.2%}")
```

### Response Format
```json
{
    "prediction": "Churn",
    "probability": 0.73,
    "risk_level": "High",
    "confidence": 0.85
}
```

## 📂 Project Structure
```
customer-churn-prediction/
│
├── app.py                      # Main Streamlit application
├── requirements.txt            # Python dependencies
├── README.md                   # Project documentation
│
├── models/
│   ├── churn_model.pkl        # Trained model
│   ├── scaler.pkl             # Feature scaler
│   └── encoder.pkl            # Categorical encoder
│
├── data/
│   ├── sample_data.csv        # Sample dataset
│   └── data_dictionary.md     # Feature descriptions
│
├── notebooks/
│   ├── eda.ipynb              # Exploratory data analysis
│   ├── model_training.ipynb   # Model development
│   └── evaluation.ipynb       # Performance evaluation
│
├── src/
│   ├── preprocessing.py       # Data preprocessing utilities
│   ├── model.py               # Model training and evaluation
│   ├── features.py            # Feature engineering
│   └── utils.py               # Helper functions
│
├── tests/
│   ├── test_preprocessing.py
│   ├── test_model.py
│   └── test_api.py
│
└── docs/
    ├── api_documentation.md
    ├── user_guide.md
    └── deployment_guide.md
```

## 🛠️ Technologies Used

| Category | Technologies |
|----------|-------------|
| **ML Framework** | Scikit-Learn |
| **Web Framework** | Streamlit |
| **Data Processing** | Pandas, NumPy |
| **Visualization** | Matplotlib, Plotly, Seaborn |
| **Model Serialization** | Pickle, Joblib |
| **API** | Requests, Flask (optional) |

## 📦 Dependencies
```txt
streamlit>=1.28.0
scikit-learn>=1.3.0
pandas>=2.0.0
numpy>=1.24.0
matplotlib>=3.7.0
seaborn>=0.12.0
plotly>=5.14.0
joblib>=1.3.0
```

## 🎯 Future Enhancements

- [ ] Real-time data pipeline integration
- [ ] A/B testing framework for retention strategies
- [ ] Advanced feature engineering with deep learning
- [ ] Customer segmentation clustering
- [ ] Automated model retraining pipeline
- [ ] Integration with CRM systems
- [ ] Mobile application support
- [ ] Multi-language support

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

Please read [CONTRIBUTING.md](CONTRIBUTING.md) for details on our code of conduct and development process.


---

⭐ If you find this project useful, please consider giving it a star!

**Made with ❤️ by [Your Name]**
