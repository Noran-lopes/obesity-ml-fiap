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

# ordem
order_original = [
    "Insufficient_Weight",
    "Normal_Weight",
    "Overweight_Level_I",
    "Overweight_Level_II",
    "Obesity_Type_I",
    "Obesity_Type_II",
    "Obesity_Type_III"
]

labels_pt = [
    "Abaixo do peso",
    "Peso normal",
    "Sobrepeso I",
    "Sobrepeso II",
    "Obesidade I",
    "Obesidade II",
    "Obesidade III"
]

# MENU
page = st.sidebar.radio(
    "📊 Navegação",
    [
        "📁 Apresentação dos Dados",
        "📊 Análise + Modelagem",
        "🧠 Calculadora",
        "💡 Recomendações"
    ]
)

# =========================================================
# 📁 APRESENTAÇÃO
# =========================================================
if page == "📁 Apresentação dos Dados":

    st.title("📁 Entendimento dos Dados")

    st.write(f"Registros: {df.shape[0]}")
    st.write(f"Variáveis: {df.shape[1]}")

    st.subheader("📊 Estatística descritiva")

    stats = df.describe().T[["mean", "min", "max"]]
    stats.columns = ["Média", "Mínimo", "Máximo"]

    st.dataframe(stats)

# =========================================================
# 📊 ANÁLISE
# =========================================================
elif page == "📊 Análise + Modelagem":

    st.title("📊 Análise Exploratória")

    # DISTRIBUIÇÃO
    st.subheader("Distribuição dos níveis de obesidade")

    fig, ax = plt.subplots(figsize=(10,5))
    sns.countplot(data=df, x="Obesity", order=order_original, palette="viridis")

    ax.set_xticklabels(labels_pt, rotation=30)
    ax.set_yticks([])

    for p in ax.patches:
        ax.annotate(int(p.get_height()),
                    (p.get_x() + p.get_width()/2, p.get_height()),
                    ha="center")

    st.pyplot(fig)

    st.markdown("Distribuição equilibrada → modelo confiável.")

    # IMC
    st.subheader("IMC vs Obesidade")

    fig2, ax2 = plt.subplots(figsize=(10,5))
    sns.boxplot(data=df, x="Obesity", y="IMC", order=order_original, palette="coolwarm")
    ax2.set_xticklabels(labels_pt, rotation=30)

    st.pyplot(fig2)

    st.markdown("IMC aumenta proporcionalmente ao nível de obesidade.")

    # IDADE
    st.subheader("Idade vs Obesidade")

    fig3, ax3 = plt.subplots(figsize=(10,5))
    sns.boxplot(data=df, x="Obesity", y="Age", order=order_original, palette="viridis")

    ax3.set_xticklabels(labels_pt, rotation=30)
    st.pyplot(fig3)

    st.markdown("""
Observa-se aumento gradual da idade.

### Outliers
Existem valores extremos indicando indivíduos mais velhos → variabilidade real.

### Conclusão
Idade influencia, mas não é fator isolado.
""")

    # MODELAGEM
    st.subheader("🧠 Modelagem")

    st.markdown("""
Random Forest foi utilizado por:

- lidar com dados mistos
- capturar relações não lineares
- ser robusto a outliers
""")

    # AVALIAÇÃO
    df_model = df.drop(["Weight", "Height"], axis=1)

    for col in df_model.select_dtypes("object"):
        df_model[col] = encoders[col].transform(df_model[col])

    X = df_model.drop("Obesity", axis=1)
    y = df_model["Obesity"]

    X_train, X_test, y_train, y_test = train_test_split(X, y)

    pred = model.predict(X_test)

    st.metric("Acurácia", f"{accuracy_score(y_test, pred):.2%}")

    fig_cm, ax_cm = plt.subplots()
    sns.heatmap(confusion_matrix(y_test, pred), annot=True, fmt="d")

    st.pyplot(fig_cm)

# =========================================================
# 🧠 CALCULADORA (COM RECOMENDAÇÕES + PESO IDEAL)
# =========================================================
elif page == "🧠 Calculadora":

    st.title("🧠 Predição personalizada")

    # inputs
    age = st.slider("Idade", 10, 80)
    faf = st.slider("Atividade física", 0, 7)
    favc = st.selectbox("Comida calórica frequente?", ["yes", "no"])
    calc = st.selectbox("Álcool", ["no", "Sometimes", "Frequently", "Always"])

    peso = st.number_input("Peso", 30.0, 200.0)
    altura = st.number_input("Altura", 1.40, 2.20)

    imc = peso / (altura**2)

    st.write(f"IMC atual: {imc:.2f}")

    if st.button("Calcular"):

        # PESO IDEAL (IMC 24.9)
        peso_ideal = 24.9 * (altura**2)

        # PRED SIMPLES (IMC base)
        if imc < 18.5:
            nivel = "Abaixo do peso"
        elif imc < 25:
            nivel = "Peso normal"
        elif imc < 30:
            nivel = "Sobrepeso"
        else:
            nivel = "Obesidade"

        st.success(f"Nível estimado: {nivel}")

        # PESO IDEAL
        st.subheader("⚖️ Peso ideal")

        st.write(f"Peso ideal estimado: **{peso_ideal:.1f} kg**")

        diferenca = peso - peso_ideal

        if diferenca > 0:
            st.write(f"Necessário reduzir aproximadamente **{diferenca:.1f} kg**.")
        else:
            st.write("Peso dentro da faixa ideal.")

        # RECOMENDAÇÕES
        st.subheader("💡 Recomendações")

        if favc == "yes":
            st.write("🍔 Reduzir alimentos calóricos pode diminuir seu nível de obesidade.")

        if faf <= 1:
            st.write("🏃 Aumentar atividade física pode colocar você em uma faixa mais saudável.")

        if calc in ["Frequently", "Always"]:
            st.write("🍺 Reduzir álcool pode ajudar na redução de peso.")

        st.write("""
Mudanças combinadas (atividade + alimentação) podem levar a redução significativa do risco.
""")

# =========================================================
# 💡 FINAL
# =========================================================
else:
    st.title("💡 Recomendações gerais")

    st.write("Atividade física + alimentação equilibrada → principais fatores.")
