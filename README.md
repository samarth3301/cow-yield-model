# Cow Milk Yield Predictor

Project to predict milk yield per cow given features like feed quality, fodder type, exercise, weather, age and historical yield.

Quick facts
- Dataset copied to `data/`
- Trained RandomForest pipeline saved at `models/rf_pipeline.pkl`

Structure

```
cow_milk_yield_predictor/
├─ data/
│  └─ cattle_milk_yield_health_dummy.csv
├─ models/
│  └─ rf_pipeline.pkl
├─ notebooks/
│  └─ EDA_and_training.ipynb
├─ src/
│  ├─ data_loader.py
│  ├─ preprocess.py
│  ├─ train.py
│  ├─ predict.py
│  └─ utils.py
├─ requirements.txt
└─ README.md
```

How to use
1. Install requirements: `pip install -r requirements.txt`
2. Train: `python src/train.py --data data/cattle_milk_yield_health_dummy.csv`
3. Predict: `python src/predict.py --Breed Jersey --Age_yrs 6 --Lactation_Stage Early --Feed_Quality 8 --Daily_Activity_hrs 5.2 --Body_Temp_C 39.5 --Heart_Rate_bpm 72 --Env_Temp_C 27 --Season Summer --Health_Status Healthy`

