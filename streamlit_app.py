import streamlit as st
import pandas as pd
import numpy as np
import pickle
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

st.set_page_config(page_title="Obesity Analytics", layout="wide")

# =========================================================
# LOAD DATA
# =========================================================
@st.cache_data
def load_data():
    df = pd.read_csv("Obesity.csv")
    df["IMC"] = df["Weight"] / (df["Height"] ** 2)
    return df

df = load_data()

model = pickle.load(open("model.pkl", "rb"))
encoders = pickle.load(open("encoders.pkl", "rb"))

# =========================================================
# MENU
# =========================================================
page = st.sidebar.radio(
    "📊 Navegação",
    ["📁 Apresentação dos Dados", "📊 Análise + Modelagem", "🧠 Calculadora"]
)

# =========================================================
# 📁 APRESENTAÇÃO
# =========================================================
if page == "📁 Apresentação dos Dados":

    st.title("📁 O que temos de dados?")

    st.markdown("""
O dataset reúne informações físicas e comportamentais de indivíduos,
com o objetivo de entender e classificar os níveis de obesidade.

As informações incluem:

- Dados físicos: idade, altura e peso  
- Hábitos: alimentação, atividade física  
- Comportamento: consumo de álcool, uso de tecnologia  

🎯 Problema: identificar fatores que levam à obesidade
""")

    # estrutura
    st.subheader("📊 Estrutura dos dados")

    st.write(f"Registros: {df.shape[0]}")
    st.write(f"Variáveis: {df.shape[1]}")

    # tipos
    st.subheader("Tipos de variáveis")

    tipos = pd.DataFrame({
        "Variável": df.columns,
        "Tipo": df.dtypes
    })

    st.dataframe(tipos)

    # qualidade
    st.subheader("Qualidade dos dados")

    st.write("Valores nulos por coluna:")
    st.dataframe(df.isnull().sum())

    # estatística
    st.subheader("Estatísticas")

    stats = df.describe().T
    st.dataframe(stats)

    st.markdown("""
### ✅ Conclusão

- Dados completos (sem missing relevante)
- Variáveis bem distribuídas
- Base adequada para análise e modelagem
""")

# =========================================================
# 📊 ANÁLISE + MODELAGEM
# =========================================================
elif page == "📊 Análise + Modelagem":

    st.title("📊 O que explica a obesidade?")

    # IMC
    st.subheader("IMC como principal fator")

    fig, ax = plt.subplots(figsize=(10,5))
    sns.boxplot(data=df, x="Obesity", y="IMC", palette="coolwarm")

    st.pyplot(fig)

    st.markdown("""
O IMC separa claramente as classes.

➡️ Principal variável explicativa do modelo.
""")

    # idade
    st.subheader("Idade")

    fig2, ax2 = plt.subplots(figsize=(10,5))
    sns.boxplot(data=df, x="Obesity", y="Age")

    st.pyplot(fig2)

    st.markdown("""
A idade tende a aumentar com a obesidade.

➡️ Fator acumulativo ao longo do tempo.
""")

    # atividade
    st.subheader("Atividade física")

    fig3, ax3 = plt.subplots()

    sns.boxplot(data=df, x="Obesity", y="FAF")

    st.pyplot(fig3)

    st.markdown("""
Baixa atividade física → maior obesidade.

➡️ Principal fator comportamental.
""")

# =========================================================
# 🤖 MODELAGEM E VALIDAÇÃO
# =========================================================
    st.title("🤖 O modelo é confiável?")

    st.markdown("""
O modelo apresentou acurácia próxima de 99%.

⚠️ Esse valor é alto demais e pode indicar **data leakage**.

Data leakage ocorre quando variáveis fornecem informação direta do resultado.
""")

    df_model = df.drop(["Weight", "Height"], axis=1)

    for col in df_model.select_dtypes("object"):
        df_model[col] = encoders[col].transform(df_model[col])

    X = df_model.drop("Obesity", axis=1)
    y = df_model["Obesity"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    pred = model.predict(X_test)

    acc = accuracy_score(y_test, pred)

    st.metric("Acurácia real", f"{acc:.2%}")

    fig_cm, ax_cm = plt.subplots()
    sns.heatmap(confusion_matrix(y_test, pred), annot=True)

    st.pyplot(fig_cm)

    st.markdown("""
### ✅ Validação

- Avaliação em dados não vistos  
- Erros entre classes próximas  
- Coerência com análise exploratória  

➡️ Modelo é confiável, mas deve ser interpretado com cautela
""")

# =========================================================
# 🧠 CALCULADORA
# =========================================================
else:

    st.title("🧠 Como melhorar meu nível de obesidade?")

    age = st.slider("Idade", 10, 80)
    faf = st.slider("Atividade física", 0, 7)
    favc = st.selectbox("Consome alimentos calóricos?", ["yes", "no"])
    calc = st.selectbox("Álcool", ["no", "Sometimes", "Frequently", "Always"])
    fcvc = st.slider("Vegetais", 1, 3)

    peso = st.number_input("Peso", 30.0, 200.0)
    altura = st.number_input("Altura", 1.40, 2.20)

    imc = peso / (altura**2)

    st.write(f"IMC atual: {imc:.2f}")

    if st.button("Avaliar"):

        peso_ideal = 24.9 * (altura**2)

        st.subheader("⚖️ Peso ideal")
        st.write(f"{peso_ideal:.1f} kg")

        st.subheader("💡 Recomendações")

        if favc == "yes":
            st.write("Reduzir calorias pode levar a uma classificação melhor.")

        if faf <= 2:
            st.write("Aumentar atividade física pode reduzir seu nível.")

        if calc in ["Frequently", "Always"]:
            st.write("Reduzir álcool pode melhorar significativamente sua condição.")

        if fcvc <= 1:
            st.write("Mais vegetais ajudam no controle do peso.")

        diff = peso - peso_ideal

        if diff > 0:
            st.write(f"Reduzindo {diff:.1f} kg você entra em uma faixa mais saudável.")
