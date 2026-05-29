import pandas as pd
import numpy as np
import pickle

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report

# =========================
# LOAD DATA
# =========================
df = pd.read_csv("Obesity.csv")

# =========================
# 🔥 LIMPAR COLUNAS
# =========================
df.columns = df.columns.str.strip()

# =========================
# 🔥 IDENTIFICAR TARGET AUTOMATICAMENTE
# =========================
target_col = None

for col in df.columns:
    if "obese" in col.lower():
        target_col = col
        break

if target_col is None:
    raise Exception(f"Coluna alvo não encontrada. Colunas disponíveis: {df.columns}")

# Padronizar nome
df.rename(columns={target_col: "Obesity_level"}, inplace=True)

# =========================
# FEATURE ENGINEERING
# =========================
df["IMC"] = df["Weight"] / (df["Height"] ** 2)

# =========================
# REMOVER LEAKAGE
# =========================
df_model = df.drop(["Weight", "Height"], axis=1)

# =========================
# ENCODING
# =========================
le_dict = {}

for col in df_model.select_dtypes(include="object").columns:
    le = LabelEncoder()
    df_model[col] = le.fit_transform(df_model[col])
    le_dict[col] = le

# =========================
# SPLIT
# =========================
X = df_model.drop("Obesity_level", axis=1)
y = df_model["Obesity_level"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# =========================
# MODEL
# =========================
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=12,
    random_state=42
)

model.fit(X_train, y_train)

# =========================
# AVALIAÇÃO
# =========================
y_pred = model.predict(X_test)

print("✅ Accuracy:", accuracy_score(y_test, y_pred))
print("\nRelatório:\n", classification_report(y_test, y_pred))

# =========================
# IMPORTÂNCIA DAS VARIÁVEIS (🔥 MUITO IMPORTANTE)
# =========================
feature_importance = pd.DataFrame({
    "feature": X.columns,
    "importance": model.feature_importances_
}).sort_values(by="importance", ascending=False)

print("\n📊 Feature Importance:")
print(feature_importance)

# =========================
# SAVE MODEL + ENCODERS
# =========================
pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(le_dict, open("encoders.pkl", "wb"))

print("✅ Modelo e encoders salvos!")
