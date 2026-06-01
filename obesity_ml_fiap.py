import pandas as pd
import pickle  
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

# =========================
# LOAD
# =========================

df = pd.read_csv("Obesity.csv")
df.columns = df.columns.str.strip()

# ✅ target fixo (igual dataset real)
TARGET = "Obesity"

# =========================
# FEATURE ENGINEERING
# =========================

df["IMC"] = df["Weight"] / (df["Height"] ** 2)
df_model = df.drop(["Weight", "Height"], axis=1)

# =========================
# ENCODING
# =========================

encoders = {}   # ✅ CORREÇÃO

for col in df_model.select_dtypes(include="object").columns:
    le = LabelEncoder()
    df_model[col] = le.fit_transform(df_model[col])
    encoders[col] = le

# =========================
# SPLIT
# =========================

X = df_model.drop(TARGET, axis=1)
y = df_model[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42  # ✅ reprodutível
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
acc = accuracy_score(y_test, y_pred)

print(f"✅ Accuracy: {round(acc*100,2)}%")

# =========================
# SAVE
# =========================

pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(encoders, open("encoders.pkl", "wb"))

print("✅ Modelo treinado e salvo")
