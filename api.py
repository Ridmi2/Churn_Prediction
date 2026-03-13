from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, field_validator, model_validator
from fastapi.middleware.cors import CORSMiddleware
from typing import List
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import traceback


# Create FastAPI app
app = FastAPI(
    title="Churn Prediction API",
    description="Predict customer churn using ML",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model at startup
print("Loading ML model...")

mlflow.set_tracking_uri("./mlruns")
model_uri = "./mlruns/2/models/m-06717be09d2c488b9d4d4d235e1f7f2c/artifacts"
model = mlflow.sklearn.load_model(model_uri)

print(f"Model loaded successfully! Type:{type(model).__name__}")

#define input data structure
FEATURE_COLUMNS = [
    "Age",
    "Tenure",
    "MonthlyCharges",
    "TotalCharges",
    "Gender_Male",
    "Contract_One year",
    "Contract_Two year",
    "InternetService_Fiber optic",
    "InternetService_No",
    "OnlineSecurity_Yes",
    "TechSupport_Yes",
    "PaymentMethod_Credit card",
    "PaymentMethod_Electronic check",
    "PaymentMethod_Mailed check",
]

SCALE_PARAMS = {
    "Age":            {"mean": 48.5,   "std": 18.0},
    "Tenure":         {"mean": 35.5,   "std": 20.8},
    "MonthlyCharges": {"mean": 69.9,   "std": 28.9},
    "TotalCharges":   {"mean": 2464.0, "std": 2330.0},
}

VALID_GENDERS          = ["Male", "Female"]
VALID_CONTRACTS        = ["Month-to-month", "One year", "Two year"]
VALID_INTERNET         = ["DSL", "Fiber optic", "No"]
VALID_SECURITY         = ["Yes", "No"]
VALID_TECHSUPPORT      = ["Yes", "No"]
VALID_PAYMENT_METHODS  = ["Electronic check", "Mailed check",
                          "Bank transfer", "Credit card"]

def scale(value: float, col: str) -> float:
    return (value - SCALE_PARAMS[col]["mean"]) / SCALE_PARAMS[col]["std"]

#imput schema
#with validation

class CustomerData(BaseModel):
    Age: int
    Tenure: int
    MonthlyCharges: float
    TotalCharges: float
    Gender: str           
    Contract: str         
    InternetService: str  
    OnlineSecurity: str   
    TechSupport: str      
    PaymentMethod: str    

    # validate Age
    @field_validator("Age")
    @classmethod
    def validate_age(cls, v):
        if not (18 <= v <= 100):
            raise ValueError(f"Age must be between 18 and 100, got {v}")
        return v    

        # Validate Tenure
    @field_validator("Tenure")
    @classmethod
    def validate_tenure(cls, v):
        if not (0 <= v <= 72):
            raise ValueError(f"Tenure must be between 0 and 72 months, got {v}")
        return v

    # Validate MonthlyCharges
    @field_validator("MonthlyCharges")
    @classmethod
    def validate_monthly_charges(cls, v):
        if not (0 <= v <= 500):
            raise ValueError(f"MonthlyCharges must be between 0 and 500, got {v}")
        return v

    # Validate TotalCharges
    @field_validator("TotalCharges")
    @classmethod
    def validate_total_charges(cls, v):
        if v < 0:
            raise ValueError(f"TotalCharges cannot be negative, got {v}")
        return v

    # Validate categorical fields
    @field_validator("Gender")
    @classmethod
    def validate_gender(cls, v):
        if v not in VALID_GENDERS:
            raise ValueError(f"Gender must be one of {VALID_GENDERS}, got '{v}'")
        return v

    @field_validator("Contract")
    @classmethod
    def validate_contract(cls, v):
        if v not in VALID_CONTRACTS:
            raise ValueError(f"Contract must be one of {VALID_CONTRACTS}, got '{v}'")
        return v

    @field_validator("InternetService")
    @classmethod
    def validate_internet(cls, v):
        if v not in VALID_INTERNET:
            raise ValueError(f"InternetService must be one of {VALID_INTERNET}, got '{v}'")
        return v

    @field_validator("OnlineSecurity")
    @classmethod
    def validate_security(cls, v):
        if v not in VALID_SECURITY:
            raise ValueError(f"OnlineSecurity must be one of {VALID_SECURITY}, got '{v}'")
        return v

    @field_validator("TechSupport")
    @classmethod
    def validate_techsupport(cls, v):
        if v not in VALID_TECHSUPPORT:
            raise ValueError(f"TechSupport must be one of {VALID_TECHSUPPORT}, got '{v}'")
        return v

    @field_validator("PaymentMethod")
    @classmethod
    def validate_payment(cls, v):
        if v not in VALID_PAYMENT_METHODS:
            raise ValueError(f"PaymentMethod must be one of {VALID_PAYMENT_METHODS}, got '{v}'")
        return v

    # Cross-field validation: TotalCharges should roughly match Tenure * MonthlyCharges
    @model_validator(mode="after")
    def validate_total_vs_tenure(self):
        if self.Tenure == 0 and self.TotalCharges > 0:
            raise ValueError("TotalCharges should be 0 when Tenure is 0")
        return self
                      

    class Config:
        json_schema_extra = {
            "example": {
                "Age": 35,
                "Tenure": 12,
                "MonthlyCharges": 85.50,
                "TotalCharges": 1026.0,
                "Gender": "Male",
                "Contract": "Month-to-month",
                "InternetService": "Fiber optic",
                "OnlineSecurity": "No",
                "TechSupport": "No",
                "PaymentMethod": "Electronic check"
            }
        }

# Batch input schema

class BatchCustomerData(BaseModel):
    customers: List[CustomerData]

    @field_validator("customers")
    @classmethod
    def validate_batch_size(cls, v):
        if len(v) == 0:
            raise ValueError("Customers list cannot be empty")
        if len(v) > 100:
            raise ValueError(f"Batch size cannot exceed 100 customers, got {len(v)}")
        return v

    class Config:
        json_schema_extra = {
            "example": {
                "customers": [
                    {
                        "Age": 28,
                        "Tenure": 3,
                        "MonthlyCharges": 95.0,
                        "TotalCharges": 285.0,
                        "Gender": "Male",
                        "Contract": "Month-to-month",
                        "InternetService": "Fiber optic",
                        "OnlineSecurity": "No",
                        "TechSupport": "No",
                        "PaymentMethod": "Electronic check"
                    },
                    {
                        "Age": 55,
                        "Tenure": 60,
                        "MonthlyCharges": 45.0,
                        "TotalCharges": 2700.0,
                        "Gender": "Female",
                        "Contract": "Two year",
                        "InternetService": "DSL",
                        "OnlineSecurity": "Yes",
                        "TechSupport": "Yes",
                        "PaymentMethod": "Credit card"
                    },
                    {
                        "Age": 42,
                        "Tenure": 24,
                        "MonthlyCharges": 70.0,
                        "TotalCharges": 1680.0,
                        "Gender": "Male",
                        "Contract": "One year",
                        "InternetService": "DSL",
                        "OnlineSecurity": "No",
                        "TechSupport": "Yes",
                        "PaymentMethod": "Mailed check"
                    }
                ]
            }
        }

def preprocess(data: CustomerData) -> np.ndarray:

    # Build a dict with all feature columns defaulting to 0
    row = {col: 0 for col in FEATURE_COLUMNS}

    # Numerical (scaled) 
    row["Age"]            = scale(data.Age,            "Age")
    row["Tenure"]         = scale(data.Tenure,         "Tenure")
    row["MonthlyCharges"] = scale(data.MonthlyCharges, "MonthlyCharges")
    row["TotalCharges"]   = scale(data.TotalCharges,   "TotalCharges")

 
    # Gender: dropped = "Female", kept = "Male"
    if data.Gender == "Male":
        row["Gender_Male"] = 1

    # Contract: dropped = "Month-to-month", kept = "One year", "Two year"
    if data.Contract == "One year":
        row["Contract_One year"] = 1
    elif data.Contract == "Two year":
        row["Contract_Two year"] = 1

    # InternetService: dropped = "DSL", kept = "Fiber optic", "No"
    if data.InternetService == "Fiber optic":
        row["InternetService_Fiber optic"] = 1
    elif data.InternetService == "No":
        row["InternetService_No"] = 1

    # OnlineSecurity: dropped = "No", kept = "Yes"
    if data.OnlineSecurity == "Yes":
        row["OnlineSecurity_Yes"] = 1

    # TechSupport: dropped = "No", kept = "Yes"
    if data.TechSupport == "Yes":
        row["TechSupport_Yes"] = 1

    # PaymentMethod: dropped = "Bank transfer", kept = rest
    if data.PaymentMethod == "Credit card":
        row["PaymentMethod_Credit card"] = 1
    elif data.PaymentMethod == "Electronic check":
        row["PaymentMethod_Electronic check"] = 1
    elif data.PaymentMethod == "Mailed check":
        row["PaymentMethod_Mailed check"] = 1

    # Return as numpy array in correct column order
    return np.array([[row[col] for col in FEATURE_COLUMNS]])


@app.get("/")
def home():
    return {
        "message": "Churn Prediction API v2.0 - Now with CI/CD!",
        "model": type(model).__name__,
        "version": "2.0.0",
        "endpoints": {
            "predict": "/predict",
            "batch": "/predict/batch",
            "health": "/health",
            "docs": "/docs"
        },
        "ci_cd": "Automated with GitHub Actions"
    }

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "model_type": type(model).__name__
    }

@app.post("/predict")
def predict(data: CustomerData):

    try:
        features    = preprocess(data)
        prediction  = model.predict(features)[0]
        probability = model.predict_proba(features)[0]
        churn_prob  = float(probability[1])

        return {
            "success":           True,
            "prediction":        "WILL CHURN" if prediction == 1 else "WILL STAY",
            "churn_probability":  round(churn_prob, 4),
            "stay_probability":   round(float(probability[0]), 4),
            "risk_level": (
                "HIGH"   if churn_prob > 0.7 else
                "MEDIUM" if churn_prob > 0.4 else
                "LOW"
            ),
            "confidence": (
                "High"   if max(probability) > 0.8 else
                "Medium" if max(probability) > 0.6 else
                "Low"
            ),
            "customer_summary": {
                "tenure_months":   data.Tenure,
                "monthly_charges": data.MonthlyCharges,
                "contract":        data.Contract,
                "internet_service":data.InternetService,
                "is_new_customer": data.Tenure < 12,
            }
        }

    except HTTPException:
        raise  # re-raise validation errors as-is

    except Exception as e:
        print(f"Prediction error: {str(e)}")
        print(traceback.format_exc())
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )

@app.post("/predict/batch")
def predict_batch(data: BatchCustomerData):
    """Predict churn for multiple customers at once (max 100)."""
    try:
        results = []

        for idx, customer in enumerate(data.customers):
            features    = preprocess(customer)
            prediction  = model.predict(features)[0]
            probability = model.predict_proba(features)[0]
            churn_prob  = float(probability[1])

            results.append({
                "customer_index":    idx,
                "prediction":        "WILL CHURN" if prediction == 1 else "WILL STAY",
                "churn_probability":  round(churn_prob, 4),
                "stay_probability":   round(float(probability[0]), 4),
                "risk_level": (
                    "HIGH"   if churn_prob > 0.7 else
                    "MEDIUM" if churn_prob > 0.4 else
                    "LOW"
                ),
                "confidence": (
                    "High"   if max(probability) > 0.8 else
                    "Medium" if max(probability) > 0.6 else
                    "Low"
                )
            })

        will_churn = sum(1 for r in results if r["prediction"] == "WILL CHURN")
        will_stay  = sum(1 for r in results if r["prediction"] == "WILL STAY")
        high_risk  = sum(1 for r in results if r["risk_level"] == "HIGH")
        medium_risk= sum(1 for r in results if r["risk_level"] == "MEDIUM")
        low_risk   = sum(1 for r in results if r["risk_level"] == "LOW")

        return {
            "success":          True,
            "total_customers":  len(data.customers),
            "predictions":      results,
            "summary": {
                "will_churn":   will_churn,
                "will_stay":    will_stay,
                "churn_rate":   f"{will_churn / len(results) * 100:.1f}%",
                "high_risk":    high_risk,
                "medium_risk":  medium_risk,
                "low_risk":     low_risk,
            }
        }

    except HTTPException:
        raise
    except Exception as e:
        print(f"Batch prediction error: {str(e)}")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")
