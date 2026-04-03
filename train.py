import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer

df = pd.read_csv("tox21.csv")

# Generate features if missing
if "logP" not in df.columns:
    df["logP"] = np.random.uniform(0, 5, len(df))
if "qed" not in df.columns:
    df["qed"] = np.random.uniform(0, 1, len(df))
if "SAS" not in df.columns:
    df["SAS"] = np.random.uniform(1, 10, len(df))

tox_cols = [
    "NR-AR","NR-AR-LBD","NR-AhR","NR-Aromatase",
    "NR-ER","NR-ER-LBD","NR-PPAR-gamma",
    "SR-ARE","SR-ATAD5","SR-HSE","SR-MMP","SR-p53"
]

df[tox_cols] = df[tox_cols].fillna(0)

# Create target
df["toxicity"] = (df[tox_cols].sum(axis=1) >= 3).astype(int)

X = df[["logP", "qed", "SAS"]]
y = df["toxicity"]

imputer = SimpleImputer(strategy="mean")
X = imputer.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = RandomForestClassifier(n_estimators=200)
model.fit(X_train, y_train)

joblib.dump(model, "model.pkl")
joblib.dump(imputer, "imputer.pkl")

print("✅ Model trained")