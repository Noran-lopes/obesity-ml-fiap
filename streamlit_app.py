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
# MENU (AGORA PROFISSIONAL)
# =========================
page = st.sidebar.radio(
    "📊 Navegação",
    ["📊 Análise dos Dados", "🧠 Calculadora", "💡 Recomendações"]
)

# =========================
# LOAD MODEL
# =========================
model = pickle.load(open("model.pkl", "rb"))
encoders = pickle.load(open("encoders.pkl", "rb"))

# =========================
# LOAD DATA
# =========================
@st.cache_data
def load_data():
    df = pd.read_csv("Obesity.csv")
    df["IMC"] = df["Weight"] / (df["Height"] ** 2)
    return df

# =========================
# 1️⃣ ANÁLISE DOS DADOS
# =========================
if page == "📊 Análise dos Dados":

    st.title("📊 Análise Estratégica da Obesidade")

    df = load_data()

    st.markdown("## 🎯 Objetivo da análise")
    st.write("Identificar os principais fatores associados aos níveis de obesidade.")

    # -------- DISTRIBUIÇÃO
    st.markdown("---")
    st.subheader("📊 Distribuição dos níveis de obesidade")

    fig, ax = plt.subplots()
    sns.countplot(x="Obesity", data=df, ax=ax)
    plt.xticks(rotation=45)
    st.pyplot(fig)

    st.info("""
    ✅ A base possui múltiplos níveis de obesidade bem distribuídos.
    
    🎯 Isso permite treinar modelos robustos e entender o fenômeno em diferentes estágios.
    """)

    # -------- IMC
    st.markdown("---")
    st.subheader("⚖️ IMC como fator determinante")

    fig2, ax2 = plt.subplots()
    sns.boxplot(x="Obesity", y="IMC", data=df, ax=ax2)
    plt.xticks(rotation=45)
    st.pyplot(fig2)

    st.info("""
    ✅ O IMC cresce conforme o nível de obesidade aumenta.

    🎯 O IMC é o principal indicador físico para classificação da obesidade.
    """)

    # -------- ATIVIDADE FÍSICA
    st.markdown("---")
    st.subheader("🏃 Relação entre atividade física e obesidade")

    fig3, ax3 = plt.subplots()
    sns.boxplot(x="Obesity", y="FAF", data=df, ax=ax3)
    plt.xticks(rotation=45)
    st.pyplot(fig3)

    st.info("""
    ✅ Pessoas com menor atividade física apresentam maior obesidade.

    🎯 Sedentarismo é um dos principais fatores de risco.
    """)

    # -------- ALIMENTAÇÃO
    st.markdown("---")
    st.subheader("🍔 Impacto da alimentação")

    fig4, ax4 = plt.subplots()
    sns.countplot(x="FAVC", hue="Obesity", data=df, ax=ax4)
    st.pyplot(fig4)

    st.info("""
    ✅ Alta frequência de alimentos calóricos está associada a maior obesidade.

    🎯 Alimentação é um fator altamente influenciável e estratégico.
    """)

    # -------- CONCLUSÃO
    st.markdown("---")
    st.subheader("📊 Conclusão geral")

    st.success("""
    🔍 Principais fatores identificados:
    - IMC (principal)
    - Baixa atividade física
    - Alimentação calórica

    🎯 Aplicação prática:
    - Prevenção
    - Recomendação personalizada
    - Apoio à saúde preventiva
    """)

# =========================
# 2️⃣ CALCULADORA
# =========================
elif page == "🧠 Calculadora":

    st.title("🧠 Calculadora de Nível de Obesidade")

    # INPUTS
    gender = st.selectbox("Gênero", ["Male", "Female"])
    age = st.slider("Idade", 10, 80)

    family_history = st.selectbox("Histórico familiar", ["yes", "no"])
    favc = st.selectbox("Alimentos calóricos frequentes?", ["yes", "no"])

    fcvc = st.slider("Vegetais (1–3)", 1, 3)
    ncp = st.slider("Refeições", 1, 5)

    caec = st.selectbox(
        "Lanches (doces, salgadinhos, fast food)",
        ["no", "Sometimes", "Frequently", "Always"]
    )

    smoke = st.selectbox("Fuma?", ["yes", "no"])

    ch2o = st.slider("Água (litros)", 1, 5)
    scc = st.selectbox("Controla calorias?", ["yes", "no"])

    faf = st.slider("Atividade física (dias)", 0, 7)
    tue = st.slider("Tecnologia (horas)", 0, 24)

    calc = st.selectbox("Álcool", ["no", "Sometimes", "Frequently", "Always"])

    mtrans = st.selectbox(
        "Transporte",
        ["Walking", "Bike", "Public_Transportation", "Automobile", "Motorbike"]
    )

    # IMC
    peso = st.number_input("Peso", 30.0, 200.0)
    altura = st.number_input("Altura", 1.40, 2.20)

    imc = peso / (altura ** 2)
    st.write(f"IMC: {imc:.2f}")

    if st.button("🔍 Prever"):

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

        # ENCODE
        for col in encoders:
            if col in input_dict:
                input_dict[col] = encoders[col].transform([input_dict[col]])[0]

        input_array = np.array(list(input_dict.values())).reshape(1, -1)

        pred = int(model.predict(input_array)[0])

        labels = [
            "Abaixo do peso",
            "Peso normal",
            "Sobrepeso I",
            "Sobrepeso II",
            "Obesidade I",
            "Obesidade II",
            "Obesidade III"
        ]

        st.success(f"✅ Resultado: {labels[pred]}")

        if pred <= 1:
            st.info("🟢 Nível saudável")
        elif pred <= 3:
            st.warning("🟡 Atenção: sobrepeso")
        else:
            st.error("🔴 Risco elevado")

        st.progress((pred + 1) / 7)

# =========================
# 3️⃣ RECOMENDAÇÕES
# =========================
else:

    st.title("💡 Recomendações de Saúde")

    st.markdown("## 🧠 Com base na análise dos dados:")

    st.success("""
    ✅ Manter boa alimentação e atividade física reduz risco de obesidade.
    """)

    st.warning("""
    ⚠️ Reduzir alimentos ultraprocessados pode melhorar significativamente o quadro.
    """)

    st.error("""
    🚨 Em casos mais graves, é recomendado acompanhamento médico especializado.
    """)

    st.info("""
    📚 Baseado em padrões identificados nos dados e diretrizes da OMS.
    """)
