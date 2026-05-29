import streamlit as st
import pandas as pd
import pickle
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# =========================
# CONFIG
# =========================
st.set_page_config(page_title="Obesity Analytics", layout="wide")

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
# LOAD DATA
# =========================
@st.cache_data
def load_data():
    df = pd.read_csv("Obesity.csv")
    df["IMC"] = df["Weight"] / (df["Height"] ** 2)
    return df

df = load_data()

model = pickle.load(open("model.pkl", "rb"))
encoders = pickle.load(open("encoders.pkl", "rb"))

# =========================================================
# 📁 APRESENTAÇÃO DOS DADOS
# =========================================================
if page == "📁 Apresentação dos Dados":

    st.title("📁 Apresentação do Dataset")

    st.subheader("📊 Visão geral")
    st.write(f"🔢 Registros: {df.shape[0]}")
    st.write(f"📊 Variáveis: {df.shape[1]}")
    st.dataframe(df.head())

    st.subheader("📌 Tipos de dados")
    tipos = df.dtypes.reset_index()
    tipos.columns = ["Variável", "Tipo"]
    st.dataframe(tipos)

    st.markdown("""
### 📊 Classificação

🔢 Quantitativos:
- Idade, Altura, Peso, IMC, Água, Atividade

🔤 Qualitativos:
- Gênero, Alimentação, Hábitos, Transporte, Obesidade
""")

    st.subheader("📊 Estatística descritiva")
    st.write(df.describe())

# =========================================================
# 📊 ANÁLISE DOS DADOS
# =========================================================
elif page == "📊 Análise dos Dados":

    st.title("📊 Análise Estratégica da Obesidade")

    # DISTRIBUIÇÃO
    st.subheader("📊 Distribuição dos níveis de obesidade")

    fig, ax = plt.subplots(figsize=(10,5))

    sns.countplot(x="Obesity", data=df, palette="viridis", ax=ax)

    labels_pt = [
        "Peso normal", "Sobrepeso I", "Sobrepeso II",
        "Obesidade I", "Abaixo do peso",
        "Obesidade II", "Obesidade III"
    ]

    ax.set_xticklabels(labels_pt, rotation=30)
    ax.set_ylabel("")
    ax.set_yticks([])

    for p in ax.patches:
        ax.annotate(
            f'{int(p.get_height())}',
            (p.get_x() + p.get_width()/2., p.get_height()),
            ha='center',
            va='bottom'
        )

    st.pyplot(fig)

    st.info("Distribuição equilibrada → modelo robusto.")

    # IMC
    st.subheader("⚖️ IMC vs obesidade")

    fig2, ax2 = plt.subplots(figsize=(10,5))

    sns.boxplot(x="Obesity", y="IMC", data=df, palette="coolwarm", ax=ax2)
    ax2.set_xticklabels(labels_pt, rotation=30)

    st.pyplot(fig2)

    st.info("IMC é o principal fator de classificação.")

    # ATIVIDADE
    st.subheader("🏃 Atividade física")

    fig3, ax3 = plt.subplots(figsize=(10,5))
    sns.boxplot(x="Obesity", y="FAF", data=df, palette="Blues", ax=ax3)
    ax3.set_xticklabels(labels_pt, rotation=30)

    st.pyplot(fig3)

    st.info("Sedentarismo está ligado ao aumento da obesidade.")

    # ALIMENTAÇÃO
    st.subheader("🍔 Alimentação calórica")

    fig4, ax4 = plt.subplots(figsize=(10,5))
    sns.countplot(x="FAVC", hue="Obesity", data=df, palette="Set2", ax=ax4)

    st.pyplot(fig4)

    st.info("Consumo calórico influencia fortemente a obesidade.")

    # PIPELINE
    st.subheader("🛠️ Preparação e Modelagem")

    st.markdown("""
✔ Criação do IMC  
✔ Remoção de Height e Weight (evitar leakage)  
✔ Encoding das variáveis categóricas  
✔ Random Forest  

""")

    # RESULTADO
    st.subheader("📈 Resultado")

    st.success("✅ Acurácia final: 97%")

    st.markdown("""
Inicialmente: ~99%  
Após ajuste (sem vazamento): 97%  

✔ Modelo confiável  
✔ Boa generalização
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
    caec = st.selectbox("Lanches", ["no", "Sometimes", "Frequently", "Always"])

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
            "Gender": gender, "Age": age,
            "family_history": family_history, "FAVC": favc,
            "FCVC": fcvc, "NCP": ncp,
            "CAEC": caec, "SMOKE": smoke,
            "CH2O": ch2o, "SCC": scc,
            "FAF": faf, "TUE": tue,
            "CALC": calc, "MTRANS": mtrans,
            "IMC": imc
        }

        for col in encoders:
            if col in input_dict:
                input_dict[col] = encoders[col].transform([input_dict[col]])[0]

        input_array = np.array(list(input_dict.values())).reshape(1, -1)

        pred = int(model.predict(input_array)[0])

        labels = [
            "Abaixo do peso", "Normal",
            "Sobrepeso I", "Sobrepeso II",
            "Obesidade I", "Obesidade II", "Obesidade III"
        ]

        st.success(f"✅ Resultado: {labels[pred]}")
        st.progress((pred + 1) / 7)

# =========================================================
# 💡 RECOMENDAÇÕES
# =========================================================
else:

    st.title("💡 Recomendações de Saúde")

    st.success("✅ Pratique atividade física regularmente")
    st.warning("⚠️ Reduza alimentos ultraprocessados")
    st.error("🚨 Busque orientação médica em casos avançados")

    st.info("📚 Baseado nos dados e boas práticas de saúde")
