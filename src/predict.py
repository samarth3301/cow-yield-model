import argparse
import joblib
import pandas as pd
import os

MODEL_PATH = os.path.join("models", "rf_pipeline.pkl")

# Features used in training
FEATURES = [
    "Breed",
    "Age (yrs)",
    "Lactation_Stage",
    "Feed_Quality",
    "Daily_Activity (hrs)",
    "Body_Temp (°C)",
    "Heart_Rate (bpm)",
    "Env_Temp (°C)",
    "Season",
    "Health_Status",
]

def load_model(model_path=MODEL_PATH):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}, please train first.")
    return joblib.load(model_path)

def predict_single(args_dict, model):
    df = pd.DataFrame([args_dict])
    return model.predict(df)[0]

def predict_from_csv(input_csv, output_csv, model):
    df = pd.read_csv(input_csv)
    preds = model.predict(df)
    df["Predicted_Milk_Yield"] = preds
    df.to_csv(output_csv, index=False)
    print(f"Predictions saved to {output_csv}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict milk yield per cow")

    parser.add_argument("--model", type=str, default=MODEL_PATH, help="Path to trained model")
    parser.add_argument("--input", type=str, help="Input CSV file for batch prediction")
    parser.add_argument("--output", type=str, default="predictions.csv", help="Output CSV file for batch prediction")

    # CLI features
    parser.add_argument("--Breed", type=str)
    parser.add_argument("--Age_yrs", type=float, dest="Age (yrs)")
    parser.add_argument("--Lactation_Stage", type=str)
    parser.add_argument("--Feed_Quality", type=int)
    parser.add_argument("--Daily_Activity_hrs", type=float, dest="Daily_Activity (hrs)")
    parser.add_argument("--Body_Temp_C", type=float, dest="Body_Temp (°C)")
    parser.add_argument("--Heart_Rate_bpm", type=int, dest="Heart_Rate (bpm)")
    parser.add_argument("--Env_Temp_C", type=int, dest="Env_Temp (°C)")
    parser.add_argument("--Season", type=str)
    parser.add_argument("--Health_Status", type=str)

    args = parser.parse_args()
    model = load_model(args.model)

    # Build input_data using the correct keys
    input_data = {
        "Cow_ID": "TEST",
        "Date": "2025-09-13",
        "Breed": args.Breed,
        "Age (yrs)": args.__dict__["Age (yrs)"],
        "Lactation_Stage": args.Lactation_Stage,
        "Feed_Quality": args.Feed_Quality,
        "Daily_Activity (hrs)": args.__dict__["Daily_Activity (hrs)"],
        "Body_Temp (°C)": args.__dict__["Body_Temp (°C)"],
        "Heart_Rate (bpm)": args.__dict__["Heart_Rate (bpm)"],
        "Env_Temp (°C)": args.__dict__["Env_Temp (°C)"],
        "Season": args.Season,
        "Health_Status": args.Health_Status,
    }

    # Case 1: Batch mode (CSV input)
    if args.input:
        predict_from_csv(args.input, args.output, model)

    # Case 2: Single prediction via CLI args
    else:
        # Check that all values are provided
        if None in input_data.values():
            print("Error: Please provide all required features or use --input CSV.")
        else:
            pred = predict_single(input_data, model)
            print(f"Predicted Milk Yield (litres/day): {pred:.2f}")
