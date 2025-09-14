import argparse
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from .data_loader import load_data
from .predict import FEATURES
import json
import pandas as pd

def main(args):
    df = load_data(args.data)
    possible_targets = [c for c in df.columns if "milk" in c.lower() or "yield" in c.lower() or "production" in c.lower()]
    target_col = possible_targets[0] if possible_targets else df.select_dtypes(include=[float,int]).columns[-1]
    
    # Use only FEATURES for X
    X = df[FEATURES].copy()
    y = df[target_col].copy()
    
    # Determine numeric and categorical from FEATURES
    numeric_cols = [c for c in FEATURES if df[c].dtype in ['int64', 'float64']]
    categorical_cols = [c for c in FEATURES if c not in numeric_cols]
    
    # Build preprocessor
    numeric_transformer = Pipeline(steps=[("scaler", StandardScaler())])
    categorical_transformer = Pipeline(
        steps=[("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]
    )
    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_transformer, numeric_cols),
        ("cat", categorical_transformer, categorical_cols)
    ])
    
    # Fill missing
    for c in numeric_cols:
        X[c] = X[c].fillna(X[c].median())
    for c in categorical_cols:
        X[c] = X[c].fillna("missing")
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = Pipeline(steps=[("preprocessor", preprocessor),
                            ("regressor", RandomForestRegressor(n_estimators=200, random_state=42))])
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    metrics = {
        "MAE": float(mean_absolute_error(y_test, preds)),
        "MSE": float(mean_squared_error(y_test, preds)),
        "RMSE": float(mean_squared_error(y_test, preds) ** 0.5),
        "R2": float(r2_score(y_test, preds))
    }
    print("Evaluation metrics:", json.dumps(metrics, indent=2))
    joblib.dump(model, args.output_model)
    print("Saved model to", args.output_model)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="data/cattle_milk_yield_health_dummy.csv")
    parser.add_argument("--output_model", type=str, default="models/rf_pipeline.pkl")
    args = parser.parse_args()
    main(args)
