import streamlit as st
import pickle
import numpy as np

# =========================
# CONFIG
# =========================
st.set_page_config(page_title="Predição de Obesidade", layout="centered")

st.title("🧠 Predição de Nível de Obesidade")
st.write("Preencha os dados abaixo para prever o nível de obesidade.")

# =========================
# LOAD MODEL
# =========================
model = pickle.load(open("model.pkl", "rb"))
encoders = pickle.load(open("encoders.pkl", "rb"))

# =========================
# INPUTS
# =========================

st.subheader("📋 Informações do Usuário")

gender = st.selectbox("Gênero", ["Male", "Female"])
age = st.slider("Idade", 10, 80)

family_history = st.selectbox("Histórico familiar de obesidade", ["yes", "no"])
favc = st.selectbox("Consome alimentos calóricos com frequência?", ["yes", "no"])

fcvc = st.slider("Consumo de vegetais (1 baixa — 3 alta)", 1.0, 3.0)
ncp = st.slider("Número de refeições diárias", 1.0, 5.0)

caec = st.selectbox("Lanches entre refeições", ["no", "Sometimes", "Frequently", "Always"])
smoke = st.selectbox("Fuma?", ["yes", "no"])

ch2o = st.slider("Consumo diário de água (litros)", 1.0, 3.0)
scc = st.selectbox("Controla calorias?", ["yes", "no"])

faf = st.slider("Frequência de atividade física (dias/semana)", 0.0, 5.0)
tue = st.slider("Tempo de uso de tecnologia (horas/dia)", 0.0, 10.0)

calc = st.selectbox("Consumo de álcool", ["no", "Sometimes", "Frequently", "Always"])

mtrans = st.selectbox(
    "Principal meio de transporte",
    ["Walking", "Bike", "Public_Transportation", "Automobile", "Motorbike"]
)

# =========================
# IMC
# =========================
st.subheader("⚖️ Medidas físicas")

peso = st.number_input("Peso (kg)", 30.0, 200.0, 70.0)
altura = st.number_input("Altura (m)", 1.40, 2.20, 1.70)

imc = peso / (altura ** 2)

st.write(f"📊 IMC calculado: **{imc:.2f}**")

# =========================
# PREVISÃO
# =========================
if st.button("🔍 Prever nível de obesidade"):

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

    # =========================
    # ENCODING
    # =========================
    for col in encoders:
        input_dict[col] = encoders[col].transform([input_dict[col]])[0]

    # Mantém ordem igual ao treino
    input_array = np.array(list(input_dict.values())).reshape(1, -1)

    # =========================
    # PREDICTION
    # =========================
    prediction = model.predict(input_array)

    st.success(f"✅ Nível previsto: **{prediction[0]}**")

    st.info("💡 Este resultado é baseado em hábitos de vida e características físicas.")
