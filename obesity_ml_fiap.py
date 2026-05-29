import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# =========================
# LOAD
# =========================
df = pd.read_csv("Obesity.csv")

df.columns = df.columns.str.strip()

# identificar target
for col in df.columns:
    if "obese" in col.lower():
        df.rename(columns={col: "Obesity_level"}, inplace=True)

# =========================
# FEATURE ENGINEERING
# =========================
df["IMC"] = df["Weight"] / (df["Height"] ** 2)

df_model = df.drop(["Weight", "Height"], axis=1)

# =========================
# ENCODING
# =========================
encoders = {}

for col in df_model.select_dtypes(include="object").columns:
    le = LabelEncoder()
    df_model[col] = le.fit_transform(df_model[col])
    encoders[col] = le

# =========================
# SPLIT
# =========================
X = df_model.drop("Obesity_level", axis=1)
y = df_model["Obesity_level"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# =========================
# MODEL
# =========================
model = RandomForestClassifier(n_estimators=200, max_depth=12)
model.fit(X_train, y_train)

# =========================
# SAVE
# =========================
pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(encoders, open("encoders.pkl", "wb"))

print("✅ Modelo treinado e salvo")
