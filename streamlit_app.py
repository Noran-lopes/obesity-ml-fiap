import streamlit as st
import pandas as pd
import pickle
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# =========================
# CONFIG
# =========================
st.set_page_config(page_title="Obesity Analytics App", layout="wide")

# =========================
# MENU
# =========================
page = st.sidebar.radio(
    "📊 Navegação",
    [
        "📁 Apresentação dos Dados",
        "📊 Análise dos Dados",
        "🧠 Calculadora",
        "💡 Recomendações"
    ]
)

# =========================
# LOAD MODEL
# =========================
model = pickle.load(open("model.pkl", "rb"))
encoders = pickle.load(open("encoders.pkl", "rb"))

@st.cache_data
def load_data():
    df = pd.read_csv("Obesity.csv")
    df["IMC"] = df["Weight"] / (df["Height"] ** 2)
    return df

df = load_data()

# =========================================================
# 📁 APRESENTAÇÃO DOS DADOS (DATA UNDERSTANDING)
# =========================================================
if page == "📁 Apresentação dos Dados":

    st.title("📁 Apresentação do Dataset")

    st.subheader("📊 Visão geral")
    st.write(f"🔢 Registros: **{df.shape[0]}**")
    st.write(f"📊 Variáveis: **{df.shape[1]}**")

    st.dataframe(df.head())

    # Tipos
    st.subheader("📌 Tipos de dados")
    tipos = df.dtypes.reset_index()
    tipos.columns = ["Variável", "Tipo"]
    st.dataframe(tipos)

    st.markdown("""
### 📊 Classificação

**Quantitativos:**
- Age, Height, Weight, FCVC, NCP, CH2O, FAF, TUE

**Qualitativos:**
- Gender, family_history, FAVC, CAEC, SMOKE, SCC, CALC, MTRANS, Obesity
""")

    # Estatística descritiva
    st.subheader("📊 Estatística descritiva")
    st.write(df.describe())

    # Distribuição de numéricos
    st.subheader("📉 Distribuição das variáveis numéricas")

    numeric_cols = df.select_dtypes(include=np.number).columns

    for col in numeric_cols:
        fig, ax = plt.subplots()
        sns.histplot(df[col], kde=True, ax=ax)
        ax.set_title(col)
        st.pyplot(fig)

    # Frequência categóricos
    st.subheader("📋 Variáveis categóricas")

    for col in df.select_dtypes(include="object").columns:
        fig, ax = plt.subplots()
        sns.countplot(x=col, data=df, ax=ax)
        plt.xticks(rotation=45)
        st.pyplot(fig)

    st.success("""
✅ Dataset contém variáveis físicas + comportamentais  
✅ Permite análise multidimensional da obesidade  
""")

# =========================================================
# 📊 ANÁLISE DOS DADOS + PIPELINE
# =========================================================
elif page == "📊 Análise dos Dados":

    st.title("📊 Análise Estratégica e Modelagem")

    # 1. DISTRIBUIÇÃO
    st.header("📊 1. Distribuição do Target")

    fig, ax = plt.subplots()
    sns.countplot(x="Obesity", data=df, ax=ax)
    plt.xticks(rotation=45)
    st.pyplot(fig)

    st.info("Distribuição equilibrada → modelo mais robusto")

    # 2. RELAÇÕES
    st.header("⚖️ 2. Variáveis-chave")

    # IMC
    fig2, ax2 = plt.subplots()
    sns.boxplot(x="Obesity", y="IMC", data=df, ax=ax2)
    plt.xticks(rotation=45)
    st.pyplot(fig2)

    # Atividade
    fig3, ax3 = plt.subplots()
    sns.boxplot(x="Obesity", y="FAF", data=df, ax=ax3)
    plt.xticks(rotation=45)
    st.pyplot(fig3)

    # Alimentação
    fig4, ax4 = plt.subplots()
    sns.countplot(x="FAVC", hue="Obesity", data=df, ax=ax4)
    st.pyplot(fig4)

    st.success("""
🔍 Insights:
- IMC é principal driver
- Sedentarismo impacta obesidade
- Alimentação calórica influencia diretamente
""")

    # 3. PREPARAÇÃO
    st.header("🛠️ 3. Preparação dos dados")

    st.markdown("""
- Criação do IMC  
- Remoção de Weight e Height  
- Encoding das variáveis categóricas  
- Split treino/teste  

🎯 Objetivo: evitar data leakage e melhorar generalização
""")

    # 4. MODELAGEM
    st.header("🤖 4. Modelagem")

    st.markdown("""
Algoritmo: Random Forest  

✔ Captura relações complexas  
✔ Funciona bem com dados mistos  
✔ Alta precisão em classificação  
""")

    # 5. RESULTADO
    st.header("📈 5. Resultado")

    st.success("✅ Accuracy final: **97%**")

    st.markdown("""
Inicialmente: ~99% → identificado data leakage  

Após correção: ~97%  

✔ Modelo confiável  
✔ Generalização real  
✔ Aplicável em cenário real  
""")

# =========================================================
# 🧠 CALCULADORA
# =========================================================
elif page == "🧠 Calculadora":

    st.title("🧠 Predição de Obesidade")

    gender = st.selectbox("Gênero", ["Male", "Female"])
    age = st.slider("Idade", 10, 80)

    family_history = st.selectbox("Histórico familiar", ["yes", "no"])
    favc = st.selectbox("Comida calórica frequente?", ["yes", "no"])

    fcvc = st.slider("Vegetais", 1, 3)
    ncp = st.slider("Refeições", 1, 5)

    caec = st.selectbox("Lanches (doces, fast-food)", ["no", "Sometimes", "Frequently", "Always"])
    smoke = st.selectbox("Fuma?", ["yes", "no"])

    ch2o = st.slider("Água (L)", 1, 5)
    scc = st.selectbox("Controla calorias?", ["yes", "no"])

    faf = st.slider("Atividade física", 0, 7)
    tue = st.slider("Tecnologia (h)", 0, 24)

    calc = st.selectbox("Álcool", ["no", "Sometimes", "Frequently", "Always"])
    mtrans = st.selectbox("Transporte", ["Walking", "Bike", "Public_Transportation", "Automobile", "Motorbike"])

    peso = st.number_input("Peso", 30.0, 200.0)
    altura = st.number_input("Altura", 1.40, 2.20)

    imc = peso / (altura ** 2)
    st.write(f"IMC: {imc:.2f}")

    if st.button("Prever"):

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

        for col in encoders:
            if col in input_dict:
                input_dict[col] = encoders[col].transform([input_dict[col]])[0]

        input_array = np.array(list(input_dict.values())).reshape(1, -1)

        pred = int(model.predict(input_array)[0])

        labels = [
            "Abaixo do peso",
            "Normal",
            "Sobrepeso I",
            "Sobrepeso II",
            "Obesidade I",
            "Obesidade II",
            "Obesidade III"
        ]

        st.success(f"✅ Resultado: {labels[pred]}")

        st.progress((pred + 1) / 7)

# =========================================================
# 💡 RECOMENDAÇÕES
# =========================================================
else:

    st.title("💡 Recomendações")

    st.success("✅ Pratique atividade física regularmente")
    st.warning("⚠️ Reduza alimentos ultraprocessados")
    st.error("🚨 Procure acompanhamento médico se necessário")

    st.info("📚 Baseado em padrões dos dados e OMS")
