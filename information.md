# Cow Yield ML Model - Information

## Algorithm Used
The model utilizes a **Random Forest Regressor** from scikit-learn. It is configured with 200 estimators (`n_estimators=200`) and a fixed random state (42) for reproducibility. The regressor is part of a machine learning pipeline that includes preprocessing steps:
- **Standard Scaling** for numeric features.
- **One-Hot Encoding** for categorical features.

This setup ensures the model can handle mixed data types effectively for predicting milk yield.

## Frequently Asked Questions (FAQs)

### 1. What is this project?
This project is a machine learning application for predicting milk yield per cow. It takes various input features (e.g., breed, age, health status) and outputs an estimated milk yield using a trained Random Forest model. The project includes both a command-line interface (CLI) for predictions and a Flask-based REST API for programmatic access.

### 2. What algorithm is used in the model?
The core algorithm is a Random Forest Regressor, as detailed above. It was chosen for its robustness, ability to handle non-linear relationships, and resistance to overfitting.

### 3. What are the input features for prediction?
The model requires the following 10 features:
- Breed (e.g., "Jersey")
- Age_yrs (e.g., 6)
- Lactation_Stage (e.g., "Early")
- Feed_Quality (e.g., 8)
- Daily_Activity_hrs (e.g., 5.2)
- Body_Temp_C (e.g., 39.5)
- Heart_Rate_bpm (e.g., 72)
- Env_Temp_C (e.g., 27)
- Season (e.g., "Summer")
- Health_Status (e.g., "Healthy")

These are provided in JSON format for the API or as CLI arguments.

### 4. How do I install and set up the project?
- Install uv: `curl -LsSf https://astral.sh/uv/install.sh | sh`
- Sync dependencies: `uv sync`
- The project requires Python 3.12+ and uses uv for dependency management.

### 5. How do I train the model?
Run the training script: `uv run python -m src.models.train --data data/cattle_milk_yield_health_dummy.csv`
- This loads the dataset, preprocesses it, trains the model, and saves it to `models/rf_pipeline.pkl`.
- Outputs evaluation metrics: MAE, MSE, RMSE, and R².

### 6. How do I make predictions via CLI?
Use: `uv run python -m src.models.predict --Breed Jersey --Age_yrs 6 --Lactation_Stage Early --Feed_Quality 8 --Daily_Activity_hrs 5.2 --Body_Temp_C 39.5 --Heart_Rate_bpm 72 --Env_Temp_C 27 --Season Summer --Health_Status Healthy`
- Outputs the predicted milk yield.

### 7. How do I run the Flask API?
Start the server: `uv run python -m src.api.api`
- Runs on `http://127.0.0.1:5002`.
- Endpoints:
  - `GET /`: API information and required features.
  - `POST /predict`: Accepts JSON with features and returns prediction.

### 8. How do I test the API?
Use curl:
```
curl -X POST http://127.0.0.1:5002/predict \
  -H "Content-Type: application/json" \
  -d '{"Breed":"Jersey","Age_yrs":6,"Lactation_Stage":"Early","Feed_Quality":8,"Daily_Activity_hrs":5.2,"Body_Temp_C":39.5,"Heart_Rate_bpm":72,"Env_Temp_C":27,"Season":"Summer","Health_Status":"Healthy"}'
```
- Response: `{"predicted_milk_yield": <value>}`

### 9. What is the model's performance?
Latest metrics (on test data):
- MAE: 3.27
- MSE: 15.43
- RMSE: 3.93
- R²: 0.31
- The model explains ~31% of variance; performance may improve with more data or tuning.

### 10. What dependencies are used?
- Flask (API)
- pandas, numpy (data handling)
- scikit-learn (ML and preprocessing)
- joblib (model serialization)
- xgboost (available but not used in current model)

### 11. How is data preprocessing handled?
- Numeric features: Filled with median, then scaled.
- Categorical features: Filled with "missing", then one-hot encoded.
- Pipeline ensures consistent transformation.

### 12. Can I use custom data?
Yes, provide a CSV with the same columns as `data/cattle_milk_yield_health_dummy.csv` and update the `--data` path.

### 13. How do I handle API errors?
- 400: Missing or invalid features.
- 500: Internal errors (check logs).
- Ensure JSON is well-formed and all features are included.

### 14. Is this production-ready?
No, it's for development. For production, add security, use a WSGI server, and enhance error handling.

### 15. How can I contribute?
- Experiment with hyperparameters or algorithms.
- Add features or improve preprocessing.
- Submit issues/PRs on GitHub.

### 16. Project Structure
- `src/models/`: ML code (data_loader, predict, preprocess, train, utils).
- `src/api/`: Flask API.
- `data/`: Sample dataset.
- `models/`: Trained model.
- `notebooks/`: EDA notebook.
- `pyproject.toml`: Config and dependencies.

For more details, refer to the README.md.