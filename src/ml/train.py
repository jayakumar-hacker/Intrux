import pandas as pd
import joblib
from sklearn.ensemble import IsolationForest

MODEL_PATH = "/home/jayakumar/Intrux/models/anomaly_model.pkl"
DATA_PATH = "/home/jayakumar/Intrux/src/ml/file.csv"

def train():
    df = pd.read_csv(DATA_PATH)

    # Convert dataframe to list of lists
    data = df.values

    model = IsolationForest(contamination=0.3)
    model.fit(data)

    joblib.dump(model, MODEL_PATH)
    print("Model trained with real data and saved.")

if __name__ == "__main__":
    train()