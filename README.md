---
title: Customer Churn Prediction API
emoji: 🔮
colorFrom: blue
colorTo: purple
sdk: docker
pinned: false
---

# Customer Churn Prediction API

A FastAPI service that predicts customer churn using Logistic Regression trained with MLflow.

## Endpoints
- `GET /` - API info
- `GET /health` - Health check
- `POST /predict` - Single prediction
- `POST /predict/batch` - Batch predictions
- `GET /docs` - Interactive docs
