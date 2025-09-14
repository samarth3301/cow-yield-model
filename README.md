# Cow Milk Yield Predictor

Project to predict milk yield per cow given features like feed quality, fodder type, exercise, weather, age and historical yield.

Quick facts
- Dataset copied to `data/`
- Trained RandomForest pipeline saved at `models/rf_pipeline.pkl`

Structure

```
cow-yield-ml-model-main/
├─ data/
│  └─ cattle_milk_yield_health_dummy.csv
├─ models/
│  └─ rf_pipeline.pkl
├─ notebooks/
│  └─ EDA_and_training.ipynb
├─ src/
│  ├─ api/
│  │  └─ api.py
│  ├─ models/
│  │  ├─ data_loader.py
│  │  ├─ predict.py
│  │  ├─ preprocess.py
│  │  ├─ train.py
│  │  └─ utils.py
│  └─ __init__.py
├─ pyproject.toml
├─ requirements.txt
└─ README.md
```

How to use
1. Install uv: `curl -LsSf https://astral.sh/uv/install.sh | sh`
2. Install dependencies: `uv sync`
3. Train: `uv run python -m src.models.train --data data/cattle_milk_yield_health_dummy.csv`
4. Predict CLI: `uv run python -m src.models.predict --Breed Jersey --Age_yrs 6 --Lactation_Stage Early --Feed_Quality 8 --Daily_Activity_hrs 5.2 --Body_Temp_C 39.5 --Heart_Rate_bpm 72 --Env_Temp_C 27 --Season Summer --Health_Status Healthy`
5. Run API: `uv run python -m src.api.api`

## API Usage

The Flask API provides endpoints to predict milk yield.

- GET `/`: API info and required features.
- POST `/predict`: Predict milk yield. Send JSON with features: Breed, Age_yrs, Lactation_Stage, Feed_Quality, Daily_Activity_hrs, Body_Temp_C, Heart_Rate_bpm, Env_Temp_C, Season, Health_Status.

Example curl:
```bash
curl -X POST http://127.0.0.1:5002/predict \
  -H "Content-Type: application/json" \
  -d '{"Breed":"Jersey","Age_yrs":6,"Lactation_Stage":"Early","Feed_Quality":8,"Daily_Activity_hrs":5.2,"Body_Temp_C":39.5,"Heart_Rate_bpm":72,"Env_Temp_C":27,"Season":"Summer","Health_Status":"Healthy"}'
```

