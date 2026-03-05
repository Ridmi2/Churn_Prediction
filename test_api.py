import pytest
from fastapi.testclient import TestClient
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from api import app

client = TestClient(app)

def test_home_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "message" in data

def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] == True

def test_predict_valid():
    response = client.post("/predict", json={
        "Age": 35, "Tenure": 12, "MonthlyCharges": 85.5, "TotalCharges": 1026.0,
        "Gender": "Male", "Contract": "Month-to-month",
        "InternetService": "Fiber optic", "OnlineSecurity": "No",
        "TechSupport": "No", "PaymentMethod": "Electronic check"
    })
    assert response.status_code == 200
    data = response.json()
    assert data["prediction"] in ["WILL CHURN", "WILL STAY"]
    assert 0 <= data["churn_probability"] <= 1

def test_predict_invalid():
    response = client.post("/predict", json={
        "Age": 999, "Tenure": 12, "MonthlyCharges": 85.5, "TotalCharges": 1026.0,
        "Gender": "Unknown", "Contract": "Month-to-month",
        "InternetService": "Fiber optic", "OnlineSecurity": "No",
        "TechSupport": "No", "PaymentMethod": "Electronic check"
    })
    assert response.status_code == 422

def test_predict_batch():
    response = client.post("/predict/batch", json={"customers": [
        {
            "Age": 28, "Tenure": 3, "MonthlyCharges": 95.0, "TotalCharges": 285.0,
            "Gender": "Male", "Contract": "Month-to-month",
            "InternetService": "Fiber optic", "OnlineSecurity": "No",
            "TechSupport": "No", "PaymentMethod": "Electronic check"
        },
        {
            "Age": 55, "Tenure": 60, "MonthlyCharges": 45.0, "TotalCharges": 2700.0,
            "Gender": "Female", "Contract": "Two year",
            "InternetService": "DSL", "OnlineSecurity": "Yes",
            "TechSupport": "Yes", "PaymentMethod": "Credit card"
        }
    ]})
    assert response.status_code == 200
    data = response.json()
    assert data["total_customers"] == 2
    assert len(data["predictions"]) == 2

def test_predict_batch_empty():
    response = client.post("/predict/batch", json={"customers": []})
    assert response.status_code == 422