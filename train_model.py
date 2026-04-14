# train_model.py — train a classifier and export artifacts as .pkl files
#
# dataset  : iris flower dataset (built into sklearn, no download needed)
# task     : multi-class classification (3 species)
# model    : random forest classifier
# exports  : scaler.pkl, model.pkl, metadata.pkl
#
# run this once locally before building the streamlit app:
#   python train_model.py

import pickle
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# ── 1. load data ──────────────────────────────────────────────────────────────

iris = load_iris()
X = iris.data           # shape: (150, 4)
y = iris.target         # 0, 1, 2 → setosa, versicolor, virginica

print(f"dataset shape : {X.shape}")
print(f"classes       : {list(iris.target_names)}")
print(f"features      : {list(iris.feature_names)}")

# ── 2. train/test split ───────────────────────────────────────────────────────

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y         # ensures balanced class distribution in both splits
)

print(f"\ntrain samples : {len(X_train)}, test samples : {len(X_test)}")

# ── 3. feature scaling ────────────────────────────────────────────────────────
# fit scaler ONLY on training data to avoid data leakage
# the same scaler must be applied to new inputs at inference time — so we export it (scaler.pkl) along with the model

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)    # only transform, never refit on test

# ── 4. train model ────────────────────────────────────────────────────────────

model = RandomForestClassifier(
    n_estimators=100,
    max_depth=5,
    random_state=42
)
model.fit(X_train_scaled, y_train)

# ── 5. evaluate ───────────────────────────────────────────────────────────────

y_pred = model.predict(X_test_scaled)
acc    = accuracy_score(y_test, y_pred)

print(f"\ntest accuracy : {acc:.4f}")
print("\nclassification report:")
print(classification_report(y_test, y_pred, target_names=iris.target_names))

# ── 6. export artifacts ───────────────────────────────────────────────────────
# export 3 things:
#   scaler.pkl    — the fitted scaler (must match what was used during training)
#   model.pkl     — the trained classifier
#   metadata.pkl  — feature names, class names, value ranges (used in the app ui)

metadata = {
    "feature_names" : iris.feature_names,   # ['sepal length (cm)', ...]
    "class_names"   : list(iris.target_names),
    "feature_ranges": {
        # (min, max, default) for each feature — used to set slider bounds in the app
        "sepal length (cm)": (4.3, 7.9, 5.8),
        "sepal width (cm)" : (2.0, 4.4, 3.1),
        "petal length (cm)": (1.0, 6.9, 3.8),
        "petal width (cm)" : (0.1, 2.5, 1.2),
    },
    "test_accuracy"  : round(acc, 4),
}

artifacts = {
    "scaler.pkl"   : scaler,
    "model.pkl"    : model,
    "metadata.pkl" : metadata,
}

# export each artifact as a .pkl file using pickle
# these files will be loaded by the streamlit app at runtime to make predictions and build the UI
for filename, obj in artifacts.items():     # loop through each artifact
    with open(filename, "wb") as f:         # open in binary mode for writing
        pickle.dump(obj, f)
    print(f"saved → {filename}")

print("\ndone. now run: streamlit run app_model.py")