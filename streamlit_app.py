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
# LOAD
# =========================
model = pickle.load(open("model.pkl", "rb"))
encoders = pickle.load(open("encoders.pkl", "rb"))

@st.cache_data
def load_data():
    df = pd.read_csv("Obesity.csv")
    df["IMC"] = df["Weight"] / (df["Height"] ** 2)
    return df

df = load_data()

# =========================
# 0️⃣ APRESENTAÇÃO DOS DADOS
# =========================
if page == "📁 Apresentação dos Dados":

    st.title("📁 Apresentação do Dataset")

    st.subheader("📊 Visão geral")
    st.write(f"🔢 Registros: {df.shape[0]}")
    st.write(f"📊 Variáveis: {df.shape[1]}")
    st.dataframe(df.head())

    st.subheader("📌 Tipos de Dados")
    tipos = df.dtypes.reset_index()
    tipos.columns = ["Variável", "Tipo"]
    st.dataframe(tipos)

    st.subheader("📊 Classificação")

    st.markdown("""
### 🔢 Quantitativos:
- Age
- Height
- Weight
- FCVC, NCP, CH2O, FAF, TUE

### 🔤 Qualitativos:
- Gender
- family_history
- FAVC
- CAEC
- SMOKE
- SCC
- CALC
- MTRANS
- Obesity (target)
""")

    st.subheader("🧠 Descrição das Variáveis")

    st.markdown("""
- **Age / Height / Weight** → dados físicos  
- **FAVC / FCVC / CAEC** → hábitos alimentares  
- **FAF / TUE** → estilo de vida  
- **SMOKE / CALC** → hábitos adicionais  
- **MTRANS** → mobilidade  
- **Obesity** → nível de obesidade (alvo)
""")

    st.subheader("📊 Estatística Descritiva")
    st.write(df.describe())

    st.subheader("📋 Variáveis Categóricas")

    for col in df.select_dtypes(include="object").columns:
        with st.expander(f"{col}"):
            st.write(df[col].value_counts())

    st.success("""
✅ Dataset completo combinando fatores físicos e comportamentais  
✅ Ideal para análise e modelagem de obesidade  
""")

# =========================
# 1️⃣ ANÁLISE DOS DADOS
# =========================
elif page == "📊 Análise dos Dados":

    st.title("📊 Análise Estratégica da Obesidade")

    # DISTRIBUIÇÃO
    st.subheader("📊 Distribuição")
    fig, ax = plt.subplots()
    sns.countplot(x="Obesity", data=df, ax=ax)
    plt.xticks(rotation=45)
    st.pyplot(fig)

    st.info("Distribuição equilibrada permite análise robusta.")

    # IMC
    st.subheader("⚖️ IMC vs Obesidade")
    fig2, ax2 = plt.subplots()
    sns.boxplot(x="Obesity", y="IMC", data=df, ax=ax2)
    plt.xticks(rotation=45)
    st.pyplot(fig2)

    st.info("IMC é o principal fator explicativo.")

    # ATIVIDADE
    st.subheader("🏃 Atividade Física")
    fig3, ax3 = plt.subplots()
    sns.boxplot(x="Obesity", y="FAF", data=df, ax=ax3)
    plt.xticks(rotation=45)
    st.pyplot(fig3)

    st.info("Sedentarismo aumenta obesidade.")

    # ALIMENTAÇÃO
    st.subheader("🍔 Alimentação")
    fig4, ax4 = plt.subplots()
    sns.countplot(x="FAVC", hue="Obesity", data=df, ax=ax4)
    st.pyplot(fig4)

    st.info("Alimentos calóricos impactam diretamente.")

    st.success("""
🔍 Conclusão:
- IMC é determinante
- Sedentarismo é crítico
- Alimentação impacta diretamente
""")

# =========================
# 2️⃣ CALCULADORA
# =========================
elif page == "🧠 Calculadora":

    st.title("🧠 Calculadora de Obesidade")

    gender = st.selectbox("Gênero", ["Male", "Female"])
    age = st.slider("Idade", 10, 80)
    family_history = st.selectbox("Histórico familiar", ["yes", "no"])
    favc = st.selectbox("Calorias frequentes?", ["yes", "no"])

    fcvc = st.slider("Vegetais", 1, 3)
    ncp = st.slider("Refeições", 1, 5)
    caec = st.selectbox("Lanches (doces/fast food)", ["no", "Sometimes", "Frequently", "Always"])

    smoke = st.selectbox("Fuma?", ["yes", "no"])
    ch2o = st.slider("Água (L)", 1, 5)
    scc = st.selectbox("Controla calorias?", ["yes", "no"])

    faf = st.slider("Atividade física", 0, 7)
    tue = st.slider("Tecnologia", 0, 24)

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
            "Abaixo do peso", "Normal", "Sobrepeso I",
            "Sobrepeso II", "Obesidade I",
            "Obesidade II", "Obesidade III"
        ]

        st.success(f"✅ Resultado: {labels[pred]}")

        if pred <= 1:
            st.info("🟢 Saudável")
        elif pred <= 3:
            st.warning("🟡 Atenção")
        else:
            st.error("🔴 Risco")

        st.progress((pred + 1) / 7)

# =========================
# 3️⃣ RECOMENDAÇÕES
# =========================
else:

    st.title("💡 Recomendações")

    st.success("✅ Pratique atividade física regularmente")
    st.warning("⚠️ Reduza alimentos ultraprocessados")
    st.error("🚨 Procure orientação médica se necessário")

    st.info("📚 Baseado em padrões dos dados e OMS")
