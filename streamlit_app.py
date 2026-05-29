import streamlit as st
import pandas as pd
import pickle
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# =========================
# CONFIG
# =========================
st.set_page_config(page_title="Obesity App", layout="wide")

# =========================
# MENU
# =========================
page = st.sidebar.selectbox(
    "Navegação",
    ["📊 Análise dos Dados", "🧠 Calculadora", "💡 Recomendações"]
)

# =========================
# LOAD
# =========================
model = pickle.load(open("model.pkl", "rb"))
encoders = pickle.load(open("encoders.pkl", "rb"))

# =========================
# 1️⃣ ANÁLISE DOS DADOS
# =========================
if page == "📊 Análise dos Dados":

    st.title("📊 Análise dos Dados de Obesidade")

    df = pd.read_csv("Obesity.csv")

    st.subheader("Distribuição de Obesidade")
    fig, ax = plt.subplots()
    sns.countplot(x="Obesity", data=df, ax=ax)
    plt.xticks(rotation=45)
    st.pyplot(fig)

    st.subheader("IMC por Categoria")
    df["IMC"] = df["Weight"] / (df["Height"]**2)
    
    fig2, ax2 = plt.subplots()
    sns.boxplot(x="Obesity", y="IMC", data=df, ax=ax2)
    plt.xticks(rotation=45)
    st.pyplot(fig2)

    st.subheader("Atividade Física vs Obesidade")
    fig3, ax3 = plt.subplots()
    sns.boxplot(x="Obesity", y="FAF", data=df, ax=ax3)
    plt.xticks(rotation=45)
    st.pyplot(fig3)

    st.info("Pessoas com menor atividade física apresentam maior incidência de obesidade.")

# =========================
# 2️⃣ CALCULADORA
# =========================
elif page == "🧠 Calculadora":

    st.title("🧠 Calculadora de Obesidade")

    gender = st.selectbox("Gênero", ["Male", "Female"])
    age = st.slider("Idade", 10, 80)

    family_history = st.selectbox("Histórico familiar", ["yes", "no"])
    favc = st.selectbox("Comida calórica frequente?", ["yes", "no"])

    fcvc = st.slider("Vegetais (1-3)", 1, 3)
    ncp = st.slider("Refeições", 1, 5)

    caec = st.selectbox("Lanches (doces, fast food)", ["no", "Sometimes", "Frequently", "Always"])
    smoke = st.selectbox("Fuma?", ["yes", "no"])

    ch2o = st.slider("Água (litros)", 1, 5)
    scc = st.selectbox("Controla calorias?", ["yes", "no"])

    faf = st.slider("Atividade física", 0, 7)
    tue = st.slider("Tecnologia (horas)", 0, 24)

    calc = st.selectbox("Álcool", ["no", "Sometimes", "Frequently", "Always"])

    mtrans = st.selectbox("Transporte", ["Walking", "Bike", "Public_Transportation", "Automobile", "Motorbike"])


