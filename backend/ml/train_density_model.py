import pandas as pd
from sklearn.model_selection import (
    train_test_split
)
from sklearn.ensemble import (
    RandomForestClassifier
)
from sklearn.metrics import (
    accuracy_score
)
import joblib
df = pd.read_csv(
    "backend/database/traffic_data.csv"
)
X = df[[
    "vehicle_count",
    "avg_speed",
    "vehicles_in",
    "vehicles_out"
]]
y = df["density"]
X_train, X_test, y_train, y_test = (
    train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )
)
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)
model.fit(
    X_train,
    y_train
)
predictions = model.predict(
    X_test
)
accuracy = accuracy_score(
    y_test,
    predictions
)
print(
    f"Accuracy: {accuracy:.2f}"
)
joblib.dump(
    model,
    "backend/ml/density_model.pkl"
)
print(
    "Model Saved!"
)