import pandas as pd
import joblib
from sklearn.ensemble import IsolationForest
df = pd.read_csv("backend/database/traffic_data.csv")
# FEATURES
X = df[[
    "vehicle_count",
    "avg_speed",
    "vehicles_in",
    "vehicles_out"
]]
model = IsolationForest(
    n_estimators=100,
    contamination=0.1,
    random_state=42
)
model.fit(X)
joblib.dump(
    model,
    "backend/ml/anomaly_model.pkl"
)
print("Anomaly Model Saved!")