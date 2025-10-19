# routing/train_routing_model.py
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
import joblib

df = pd.read_csv("routing_dataset.csv")

feature_cols = ["task_type","compute_cost","data_size_bytes",
                "pod_cpu_load","pod_battery","pod_temperature",
                "num_pods","avg_cpu_load","pod_lat","pod_lon"]
X = df[feature_cols]
y = df["label"]

preprocessor = ColumnTransformer(
    transformers=[("task_type", OneHotEncoder(handle_unknown="ignore"), ["task_type"])],
    remainder="passthrough"
)

model = Pipeline([
    ("preprocess", preprocessor),
    ("classifier", RandomForestClassifier(n_estimators=200, random_state=42))
])

model.fit(X, y)
joblib.dump(model, "routing_model.pkl")
print("[OK] routing_model.pkl trained and saved")
