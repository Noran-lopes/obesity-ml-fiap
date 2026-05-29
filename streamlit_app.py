import streamlit as st
import pickle
import numpy as np

# =========================
# CONFIG
# =========================
st.set_page_config(page_title="Obesity Prediction", layout="centered")

# =========================
# LANGUAGE
# =========================
lang = st.selectbox("Idioma / Language", ["Português", "English"])

text = {
    "Português": {
        "title": "🧠 Predição de Nível de Obesidade",
        "info": "Preencha os dados abaixo:",
        "gender": "Gênero",
        "age": "Idade",
        "family": "Histórico familiar de obesidade",
        "favc": "Consome alimentos calóricos com frequência?",
        "veg": "Consumo de vegetais (1 baixa — 3 alta)",
        "meals": "Número de refeições diárias",
        "snacks": "Frequência de lanches (doces, salgados, fast food)",
        "smoke": "Fuma?",
        "water": "Consumo diário de água (litros)",
        "calories": "Controla calorias?",
        "activity": "Atividade física (dias/semana)",
        "tech": "Tempo de tecnologia (horas/dia)",
        "alcohol": "Consumo de álcool",
        "transport": "Meio de transporte",
        "weight": "Peso (kg)",
        "height": "Altura (m)",
        "predict": "🔍 Prever nível de obesidade"
    },
    "English": {
        "title": "🧠 Obesity Level Prediction",
        "info": "Fill in the data below:",
        "gender": "Gender",
        "age": "Age",
        "family": "Family history of obesity",
        "favc": "Frequent high-calorie food consumption?",
        "veg": "Vegetable consumption (1 low — 3 high)",
        "meals": "Number of daily meals",
        "snacks": "Snacking frequency (sweets, snacks, fast food)",
        "smoke": "Do you smoke?",
        "water": "Daily water intake (liters)",
        "calories": "Monitor calorie intake?",
        "activity": "Physical activity (days/week)",
        "tech": "Technology use (hours/day)",
        "alcohol": "Alcohol consumption",
        "transport": "Transport",
        "weight": "Weight (kg)",
        "height": "Height (m)",
        "predict": "🔍 Predict obesity level"
    }
}

t = text[lang]

# =========================
# LOAD MODEL
# =========================
model = pickle.load(open("model.pkl", "rb"))
encoders = pickle.load(open("encoders.pkl", "rb"))

# =========================
# UI
# =========================
st.title(t["title"])
st.write(t["info"])

gender = st.selectbox(t["gender"], ["Male", "Female"])
age = st.slider(t["age"], 10, 80)

family_history = st.selectbox(t["family"], ["yes", "no"])
favc = st.selectbox(t["favc"], ["yes", "no"])

fcvc = st.slider(t["veg"], 1, 3)
ncp = st.slider(t["meals"], 1, 5)

caec = st.selectbox(
    t["snacks"],
    ["no", "Sometimes", "Frequently", "Always"]
)

smoke = st.selectbox(t["smoke"], ["yes", "no"])

ch2o = st.slider(t["water"], 1, 5)  # ✅ até 5L
scc = st.selectbox(t["calories"], ["yes", "no"])

faf = st.slider(t["activity"], 0, 7)
tue = st.slider(t["tech"], 0, 24)  # ✅ até 24h

calc = st.selectbox(t["alcohol"], ["no", "Sometimes", "Frequently", "Always"])

mtrans = st.selectbox(
    t["transport"],
    ["Walking", "Bike", "Public_Transportation", "Automobile", "Motorbike"]
)

# =========================
# IMC
# =========================
peso = st.number_input(t["weight"], 30.0, 200.0, 70.0)
altura = st.number_input(t["height"], 1.40, 2.20, 1.70)

imc = peso / (altura ** 2)

st.write(f"📊 IMC: **{imc:.2f}**")

# =========================
# PREDICTION
# =========================
if st.button(t["predict"]):

    input_dict = {
        "Gender": gender,
        "Age": age,
        "family_history": family_history,
        "FAVC": favc,
        "FCVC": fcvc,
        "NCP": ncp,
        "CAEC": caec,
        "SMOKE": smoke,
        "CH2O": ch2o,
        "SCC": scc,
        "FAF": faf,
        "TUE": tue,
        "CALC": calc,
        "MTRANS": mtrans,
        "IMC": imc
    }

    # ENCODING
    for col in encoders:
        if col in input_dict:
            input_dict[col] = encoders[col].transform([input_dict[col]])[0]

    input_array = np.array(list(input_dict.values())).reshape(1, -1)

    prediction = int(model.predict(input_array)[0])

    # =========================
    # LABELS
    # =========================
    labels_pt = {
        0: "Abaixo do peso",
        1: "Peso normal",
        2: "Sobrepeso I",
        3: "Sobrepeso II",
        4: "Obesidade I",
        5: "Obesidade II",
        6: "Obesidade III (mórbida)"
    }

    labels_en = {
        0: "Underweight",
        1: "Normal Weight",
        2: "Overweight I",
        3: "Overweight II",
        4: "Obesity I",
        5: "Obesity II",
        6: "Obesity III (severe)"
    }

    label = labels_pt if lang == "Português" else labels_en

    st.success(f"✅ Resultado: **{label[prediction]}**")

    # =========================
    # RISK INTERPRETATION
    # =========================
    if prediction <= 1:
        st.info("🟢 Nível saudável")
    elif prediction <= 3:
        st.warning("🟡 Atenção: sobrepeso")
    else:
        st.error("🔴 Risco elevado de obesidade")

    # =========================
    # VISUAL SCALE
    # =========================
    st.subheader("📊 Posição no quadro de obesidade")
    st.progress((prediction + 1) / 7)

    # =========================
    # FOOTER
    # =========================
    st.info("💡 Resultado baseado em hábitos e características físicas.")
