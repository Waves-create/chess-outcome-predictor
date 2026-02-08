#For normal model training and evaluation

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

from feature_extractor import extract_features_from_moves


# Loading Dataset

df = pd.read_csv("data/games.csv")

print("Original dataset:", df.shape)

# Keeping only white/black wins
df = df[df["winner"].isin(["white", "black"])]
df = df.dropna(subset=["moves", "white_rating", "black_rating"])

print("After cleaning:", df.shape)


# Feature Engineering

data = []

for _, row in df.iterrows():

    chess_features = extract_features_from_moves(
        row["moves"],
        max_moves=40   # early game prediction
    )

    features = {
        "rating_diff": row["white_rating"] - row["black_rating"],
        "captures": chess_features["captures"],
        "checks": chess_features["checks"],
        "turns_played": chess_features["turns_played"],
        "material_balance": chess_features["material_balance"],
        "target": 1 if row["winner"] == "white" else 0
    }

    data.append(features)

ml_df = pd.DataFrame(data)

print("\nFeature dataset shape:", ml_df.shape)
print(ml_df.head())


# Training the Model

X = ml_df.drop("target", axis=1)
y = ml_df["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    random_state=42
)

model.fit(X_train, y_train)


# Evaluation

preds = model.predict(X_test)

print("\nAccuracy:", accuracy_score(y_test, preds))
# print("\nClassification Report:\n", classification_report(y_test, preds))

# Showing Sample Predictions

label_map = {
    1: "white",
    0: "black"
}

results = X_test.copy()
results["Actual"] = y_test.map(label_map).values
results["Predicted"] = pd.Series(preds).map(label_map).values

print("\nSample Predictions:")
print(results.head(30))

