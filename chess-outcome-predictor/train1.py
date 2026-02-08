#For training the model and evaluating at different stages of the game

import matplotlib.pyplot as plt
import pandas as pd

from feature_extractor import extract_features_from_moves
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Loading Dataset

df = pd.read_csv("data/games.csv")

print("Original dataset:", df.shape)

# Keep only white/black wins
df = df[df["winner"].isin(["white", "black"])]
df = df.dropna(subset=["moves", "white_rating", "black_rating"])

print("After cleaning:", df.shape)

def train_and_evaluate(df, max_moves):

    data = []

    for _, row in df.iterrows():

        chess_features = extract_features_from_moves(
            row["moves"],
            max_moves=max_moves
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

    acc = accuracy_score(y_test, preds)
    return acc


# Stage-wise Evaluation

stages = [20, 40, 80, 150]
results = []

for moves in stages:
    acc = train_and_evaluate(df, max_moves=moves)
    results.append({"max_moves": moves, "accuracy": acc})
    print(f"max_moves={moves} → accuracy={acc:.3f}")

results_df = pd.DataFrame(results)


# Plot Accuracy vs Game Progress

plt.figure(figsize=(8, 5))
plt.plot(results_df["max_moves"], results_df["accuracy"], marker="o")

plt.xlabel("Number of Moves Used (Game Progress)")
plt.ylabel("Prediction Accuracy")
plt.title("Chess Outcome Prediction Accuracy vs Game Progress")
plt.grid(True)

plt.show()
