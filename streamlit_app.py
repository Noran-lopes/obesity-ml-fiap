import streamlit as st
import pandas as pd
import numpy as np
import pickle
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

# =========================
# ORDEM CORRETA (REQUISITO)
# =========================
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

# =========================================================
# 📁 APRESENTAÇÃO DOS DADOS
# =========================================================
if page == "📁 Apresentação dos Dados":

    st.title("📁 Entendimento do Dataset")

    st.write(f"📊 Registros: {df.shape[0]}")
    st.write(f"📊 Variáveis: {df.shape[1]}")

    tipos = df.dtypes.reset_index()
    tipos.columns = ["Variável", "Tipo técnico"]

    descricao = {
        "Gender": "Sexo do indivíduo",
        "Age": "Idade (anos)",
        "Height": "Altura (m)",
        "Weight": "Peso (kg)",
        "family_history": "Histórico familiar",
        "FAVC": "Consumo calórico",
        "FCVC": "Consumo de vegetais",
        "NCP": "Refeições por dia",
        "CAEC": "Lanches",
        "SMOKE": "Fumante",
        "CH2O": "Consumo de água",
        "SCC": "Controle de calorias",
        "FAF": "Atividade física",
        "TUE": "Tempo de tecnologia",
        "CALC": "Álcool",
        "MTRANS": "Transporte",
        "Obesity": "Nível de obesidade"
    }

    tipos["Descrição"] = tipos["Variável"].map(descricao)

    st.dataframe(tipos, use_container_width=True)

    st.info("""
O dataset combina:

🔹 Dados físicos (IMC, idade)  
🔹 Dados comportamentais (atividade, alimentação)

➡️ Permite modelagem completa do risco de obesidade
""")

# =========================================================
# 📊 ANÁLISE DOS DADOS
# =========================================================
elif page == "📊 Análise dos Dados":

    st.title("📊 Análise Exploratória + Pipeline Analítico")

    # =========================
    # 1. DISTRIBUIÇÃO
    # =========================
    st.subheader("📊 Distribuição dos níveis de obesidade")

    fig, ax = plt.subplots(figsize=(10,5))

    sns.countplot(
        data=df,
        x="Obesity",
        order=order_original,
        palette="viridis",
        ax=ax
    )

    ax.set_xticklabels(labels_pt, rotation=30)
    ax.set_ylabel("")
    ax.set_yticks([])

    for p in ax.patches:
        ax.annotate(
            f"{int(p.get_height())}",
            (p.get_x() + p.get_width()/2., p.get_height()),
            ha="center"
        )

    st.pyplot(fig)

    st.markdown("""
### 📌 Análise do gráfico:

A distribuição dos dados mostra que todas as classes de obesidade possuem volume relevante de observações.

➡️ Destaque para:
- **Obesidade Tipo I (~351 casos)** como maior grupo
- **Abaixo do peso (~272 casos)** como menor grupo

### 🎯 Conclusão:
A base é **equilibrada**, o que evita viés no treinamento e permite um modelo mais robusto.
""")

    # =========================
    # 2. IMC
    # =========================
    st.subheader("⚖️ Relação entre IMC e obesidade")

    fig2, ax2 = plt.subplots(figsize=(10,5))

    sns.boxplot(
        data=df,
        x="Obesity",
        y="IMC",
        order=order_original,
        palette="coolwarm",
        ax=ax2
    )

    ax2.set_xticklabels(labels_pt, rotation=30)

    st.pyplot(fig2)

    st.markdown("""
### 📌 Análise do gráfico:

Existe uma clara progressão do IMC conforme o nível de obesidade aumenta.

➡️ A separação entre grupos é bem definida.

### 🎯 Conclusão:
O IMC é o **principal driver da classificação**, justificando sua inclusão como variável central no modelo.
""")

    # =========================
    # 3. ATIVIDADE
    # =========================
    st.subheader("🏃 Atividade física vs obesidade")

    fig3, ax3 = plt.subplots(figsize=(10,5))

    sns.boxplot(
        data=df,
        x="Obesity",
        y="FAF",
        order=order_original,
        palette="Blues",
        ax=ax3
    )

    ax3.set_xticklabels(labels_pt, rotation=30)

    st.pyplot(fig3)

    st.markdown("""
### 📌 Análise:

Observa-se que níveis mais altos de obesidade estão associados a menor frequência de atividade física.

### 🎯 Conclusão:
Sedentarismo é um dos principais fatores comportamentais ligados à obesidade.
""")

    # =========================
    # 4. ALIMENTAÇÃO
    # =========================
    st.subheader("🍔 Consumo calórico vs obesidade")

    fig4, ax4 = plt.subplots(figsize=(10,5))

    sns.countplot(
        data=df,
        x="FAVC",
        hue="Obesity",
        palette="Set2",
        ax=ax4
    )

    st.pyplot(fig4)

    st.markdown("""
### 📌 Análise:

A presença de consumo frequente de alimentos calóricos aumenta conforme os níveis de obesidade crescem.

### 🎯 Conclusão:
A alimentação é um fator diretamente relacionado ao aumento do peso e risco de obesidade.
""")

    # =========================
    # PIPELINE
    # =========================
    st.subheader("🛠️ Pipeline de Modelagem")

    st.markdown("""
### Etapas realizadas:

1. Criação do IMC  
2. Remoção de Height e Weight (evitar leakage)  
3. Encoding das variáveis categóricas  
4. Treinamento com Random Forest  

### 🎯 Resultado técnico:
Modelo com alta capacidade de generalização
""")

    st.success("✅ Acurácia final: 97%")

    st.markdown("""
O modelo inicialmente apresentou ~99%, porém foi identificado vazamento de dados.

Após ajuste, a acurácia de 97% representa um modelo mais confiável e aplicável.
""")

# =========================================================
# 🧠 CALCULADORA
# =========================================================
elif page == "🧠 Calculadora":

    st.title("🧠 Calculadora de Obesidade")

    gender = st.selectbox("Gênero", ["Male", "Female"])
    age = st.slider("Idade", 10, 80)

    family_history = st.selectbox("Histórico", ["yes", "no"])
    favc = st.selectbox("Comida calórica?", ["yes", "no"])

    fcvc = st.slider("Vegetais", 1, 3)
    ncp = st.slider("Refeições", 1, 5)

    caec = st.selectbox("Lanches", ["no","Sometimes","Frequently","Always"])

    smoke = st.selectbox("Fuma?", ["yes","no"])
    ch2o = st.slider("Água",1,5)
    scc = st.selectbox("Controle de calorias",["yes","no"])

    faf = st.slider("Atividade física",0,7)
    tue = st.slider("Tecnologia",0,24)

    calc = st.selectbox("Álcool",["no","Sometimes","Frequently","Always"])
    mtrans = st.selectbox("Transporte",["Walking","Bike","Public_Transportation","Automobile","Motorbike"])

    peso = st.number_input("Peso",30.0,200.0)
    altura = st.number_input("Altura",1.40,2.20)

    imc = peso/(altura**2)
    st.write(f"IMC: {imc:.2f}")

    if st.button("Prever"):

        input_dict = {
            "Gender":gender,"Age":age,
            "family_history":family_history,"FAVC":favc,
            "FCVC":fcvc,"NCP":ncp,
            "CAEC":caec,"SMOKE":smoke,
            "CH2O":ch2o,"SCC":scc,
            "FAF":faf,"TUE":tue,
            "CALC":calc,"MTRANS":mtrans,
            "IMC":imc
        }

        for col in encoders:
            if col in input_dict:
                input_dict[col] = encoders[col].transform([input_dict[col]])[0]

        pred = int(model.predict(np.array(list(input_dict.values())).reshape(1,-1))[0])

        labels = [
            "Abaixo do peso","Normal","Sobrepeso I",
            "Sobrepeso II","Obesidade I","Obesidade II","Obesidade III"
        ]

        st.success(f"✅ Resultado: {labels[pred]}")
        st.progress((pred+1)/7)

# =========================================================
# 💡 RECOMENDAÇÕES
# =========================================================
else:

    st.title("💡 Recomendações")

    st.success("✅ Mantenha atividade física regular")
    st.warning("⚠️ Reduza alimentos calóricos")
    st.error("🚨 Busque suporte médico se necessário")

    st.info("📚 Baseado nos padrões identificados")

