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
# 📁 APRESENTAÇÃO DOS DADOS
# =========================================================
if page == "📁 Apresentação dos Dados":

    st.title("📁 O que temos de dados?")

    st.markdown("""
Este dataset reúne informações relacionadas à obesidade, combinando:

- Dados físicos (idade, altura, peso)
- Hábitos alimentares
- Estilo de vida
- Comportamentos do dia a dia

O objetivo é analisar quais fatores influenciam o nível de obesidade.
""")

    # tamanho
    st.write(f"Registros: {df.shape[0]}")
    st.write(f"Variáveis: {df.shape[1]}")

    # tipos
    st.subheader("📊 Tipos de dados")

    tipos = df.dtypes.reset_index()
    tipos.columns = ["Variável", "Tipo"]

    st.dataframe(tipos)

    # missing
    st.subheader("🔍 Dados faltantes")

    missing = df.isnull().sum()
    st.dataframe(missing)

    # estatística
    st.subheader("📊 Estatística descritiva")

    stats = df.describe().T[["mean", "min", "max"]]
    stats.columns = ["Média", "Mínimo", "Máximo"]
    st.dataframe(stats)

    st.markdown("""
### 📌 Conclusão

- Dados completos (sem missing relevante)  
- Boa variabilidade  
- Variáveis adequadas para modelagem  

➡️ Base pronta para análise e machine learning
""")

# =========================================================
# 📊 ANÁLISE + MODELAGEM
# =========================================================
elif page == "📊 Análise + Modelagem":

    st.title("📊 Análise e Modelagem")

    # IMC
    st.subheader("IMC vs Obesidade")

    fig, ax = plt.subplots(figsize=(10,5))
    sns.boxplot(data=df, x="Obesity", y="IMC", palette="coolwarm")

    st.pyplot(fig)

    st.markdown("""
O IMC apresenta separação clara entre classes → principal fator explicativo.
""")

    # IDADE
    st.subheader("Idade vs Obesidade")

    fig2, ax2 = plt.subplots(figsize=(10,5))
    sns.boxplot(data=df, x="Obesity", y="Age")

    st.pyplot(fig2)

    st.markdown("""
A idade apresenta crescimento gradual.

Outliers indicam variabilidade real da população.
""")

    # MODELAGEM
    st.subheader("🧠 Modelagem")

    st.markdown("""
O modelo utilizou Random Forest devido a:

- capacidade de lidar com dados mistos  
- captura de relações não lineares  
- robustez a outliers  

### ⚠️ Atenção:
Acurácia de 99% pode indicar data leakage.

Para evitar isso:
- removemos Height e Weight
- usamos apenas IMC
""")

    # validação
    df_model = df.drop(["Weight", "Height"], axis=1)

    for col in df_model.select_dtypes("object"):
        df_model[col] = encoders[col].transform(df_model[col])

    X = df_model.drop("Obesity", axis=1)
    y = df_model["Obesity"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2
    )

    pred = model.predict(X_test)

    st.metric("Acurácia", f"{accuracy_score(y_test, pred):.2%}")

    # matriz
    fig_cm, ax_cm = plt.subplots()
    sns.heatmap(confusion_matrix(y_test, pred), annot=True, fmt="d")

    st.pyplot(fig_cm)

    st.markdown("""
### 📌 Provas de validade do modelo

- Avaliação em dados não vistos (teste)
- Erros concentrados entre classes próximas
- Feature importance coerente

➡️ Indica modelo consistente
""")

    # importance
    st.subheader("Importância das variáveis")

    imp = pd.DataFrame({
        "Feature": X.columns,
        "Importance": model.feature_importances_
    }).sort_values(by="Importance")

    fig_imp, ax_imp = plt.subplots()
    ax_imp.barh(imp["Feature"], imp["Importance"])

    st.pyplot(fig_imp)

# =========================================================
# 🧠 CALCULADORA ROBUSTA
# =========================================================
else:

    st.title("🧠 Avaliação personalizada")

    gender = st.selectbox("Gênero", ["Male","Female"])
    age = st.slider("Idade",10,80)

    favc = st.selectbox("Consumo de alimentos calóricos?",["yes","no"])
    fcvc = st.slider("Vegetais",1,3)

    faf = st.slider("Atividade física",0,7)
    tue = st.slider("Tempo de tecnologia",0,24)

    calc = st.selectbox("Álcool",["no","Sometimes","Frequently","Always"])

    peso = st.number_input("Peso",30.0,200.0)
    altura = st.number_input("Altura",1.40,2.20)

    imc = peso / (altura**2)

    st.write(f"IMC atual: {imc:.2f}")

    if st.button("Avaliar"):

        peso_ideal = 24.9 * (altura**2)

        st.subheader("Resultado")

        input_dict = {
            "Gender": gender,
            "Age": age,
            "FAVC": favc,
            "FCVC": fcvc,
            "FAF": faf,
            "TUE": tue,
            "CALC": calc,
            "IMC": imc
        }

        # recomendações inteligentes
        st.subheader("💡 O que pode melhorar")

        if calc in ["Frequently","Always"]:
            st.write("➡️ Reduzir álcool pode diminuir seu nível de obesidade.")

        if favc == "yes":
            st.write("➡️ Reduzir alimentos calóricos pode te levar a uma categoria mais saudável.")

        if faf <= 2:
            st.write("➡️ Mais atividade física pode reduzir seu nível de obesidade.")

        if fcvc <= 1:
            st.write("➡️ Aumentar vegetais melhora seu metabolismo.")

        st.subheader("⚖️ Peso ideal")

        diff = peso - peso_ideal

        st.write(f"Peso ideal aproximado: {peso_ideal:.1f} kg")

        if diff > 0:
            st.write(f"Você pode reduzir cerca de {diff:.1f} kg para atingir uma faixa mais saudável.")
