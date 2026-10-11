
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import (
    train_test_split,
    cross_val_score,
)

# Find the project folder and dataset
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "student-mat.csv"
MODEL_PATH = BASE_DIR / "student_model.pkl"

# Choose the inputs and the grade we want to predict
FEATURES = ["studytime", "failures", "absences", "G1", "G2"]
TARGET = "G3"


def main():
    # Load the dataset
    df = pd.read_csv(DATA_PATH, sep=";")

    X = df[FEATURES]
    y = df[TARGET]

    print("Dataset loaded:", df.shape)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
    )

    # Train and evaluate the model
    model = RandomForestRegressor(
        n_estimators=100,
        random_state=42,
    )

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    test_mae = mean_absolute_error(y_test, predictions)
    print(f"Test MAE: {test_mae:.3f}")

    # Check performance using 5-fold cross-validation
    cv_scores = cross_val_score(
        RandomForestRegressor(
            n_estimators=100,
            random_state=42,
        ),
        X_train,
        y_train,
        cv=5,
        scoring="neg_mean_absolute_error",
    )

    cv_mae = -cv_scores
    print(f"Average cross-validation MAE: {cv_mae.mean():.3f}")
    print("MAE for each fold:", cv_mae.round(3))

    # Train the final model using all available data
    final_model = RandomForestRegressor(
        n_estimators=100,
        random_state=42,
    )
    final_model.fit(X, y)

    # Save the trained model
    joblib.dump(
        {"model": final_model, "features": FEATURES},
        MODEL_PATH,
    )

    print("Model saved successfully!")
    print("Saved to:", MODEL_PATH)


if __name__ == "__main__":
    main()
