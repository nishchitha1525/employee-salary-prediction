import pandas as pd
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score


# ==============================
# LOAD DATASET
# ==============================

data = pd.read_csv("ml_model/dataset.csv")


# ==============================
# FEATURES
# ==============================

X = data[
    [
        "Age",
        "Gender",
        "Education",
        "Job_Title",
        "Experience",
        "Department"
    ]
]


# ==============================
# TARGET
# ==============================

y = data["Salary"]


# ==============================
# CATEGORICAL FEATURES
# ==============================

categorical_features = [
    "Gender",
    "Education",
    "Job_Title",
    "Department"
]


# ==============================
# PREPROCESSING
# ==============================

preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features
        )
    ],
    remainder="passthrough"
)


# ==============================
# RANDOM FOREST MODEL
# ==============================

model = RandomForestRegressor(
    n_estimators=200,
    random_state=42
)


# ==============================
# PIPELINE
# ==============================

pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ]
)


# ==============================
# TRAIN AND TEST DATA
# ==============================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# ==============================
# TRAIN MODEL
# ==============================

pipeline.fit(
    X_train,
    y_train
)


# ==============================
# PREDICTION
# ==============================

predictions = pipeline.predict(
    X_test
)


# ==============================
# MODEL EVALUATION
# ==============================

mae = mean_absolute_error(
    y_test,
    predictions
)

r2 = r2_score(
    y_test,
    predictions
)


print("================================")
print("Employee Salary Prediction Model")
print("================================")

print("Model Training Completed")

print("Mean Absolute Error:", mae)

print("R2 Score:", r2)


# ==============================
# SAVE MODEL
# ==============================

joblib.dump(
    pipeline,
    "ml_model/salary_model.pkl"
)

print()
print("Model saved successfully!")
print("File: ml_model/salary_model.pkl")