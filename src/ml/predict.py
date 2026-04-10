import joblib

MODEL_PATH = "/home/jayakumar/Intrux/models/anomaly_model.pkl"

model = joblib.load(MODEL_PATH)

def predict(features):
    result = model.predict([features])
    return 1 if result[0] == -1 else 0


if __name__ == "__main__":
    # Example real-world inputs

    normal = [110, 6, 443, 51514, 10, 1]
    attack = [12000, 6, 9999, 60000, 600, 10]
    print("Normal test:", predict(normal))
    print("Attack test:", predict(attack))