# =========================================================
# 📦 IMPORTS
# =========================================================
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import seaborn as sns
import matplotlib.pyplot as plt

# =========================================================
# ⚙️ CONFIG
# =========================================================
st.set_page_config(page_title="Obesity Analytics", layout="wide")

# =========================================================
# 📊 MENU (ORDEM CORRIGIDA)
# =========================================================
st.sidebar.markdown("""
### 🧠 Fluxo Analítico

1. Exploração dos dados  
2. Análise e modelagem  
3. Dashboard executivo  
4. Simulação  
5. Recomendações  
""")

page = st.sidebar.radio(
    "📊 Navegação",
    [
        "📁 Exploração dos Dados",
        "📊 Análise + Modelagem",
        "📊 Dashboard Executivo",
        "🧠 Calculadora Inteligente",
        "💡 Recomendações Estratégicas"
    ]
)

# =========================================================
# 📁 LOAD DATA (SUPER ROBUSTO)
# =========================================================
@st.cache_data
def load_data():
    df = pd.read_csv("Obesity.csv")

    # limpar nomes
    df.columns = df.columns.str.strip()

    # detectar coluna target automaticamente
    for col in df.columns:
        if col.lower() in ["nobeyesdad", "obesity_level"]:
            df.rename(columns={col: "Obesity_level"}, inplace=True)

    # criar IMC
    if "Weight" in df.columns and "Height" in df.columns:
        df["IMC"] = df["Weight"] / (df["Height"] ** 2)

    return df

df = load_data()

# =========================================================
# 🤖 LOAD MODEL (opcional)
# =========================================================
try:
    model = pickle.load(open("model.pkl", "rb"))
    encoders = pickle.load(open("encoders.pkl", "rb"))
    model_loaded = True
except:
    model_loaded = False

# =========================================================
# 📁 EXPLORAÇÃO DOS DADOS (PRIMEIRO)
# =========================================================
if page == "📁 Exploração dos Dados":

    st.title("📁 Exploração dos Dados")

    st.subheader("Visão inicial")
    st.write(df.head())

    st.subheader("Estrutura das variáveis")
    st.write(df.dtypes)

    st.subheader("Estatísticas descritivas")
    st.write(df.describe())

    st.markdown("""
    ### 🧠 Interpretação

    A base contém variáveis:
    - Físicas (IMC, peso, altura)
    - Demográficas (idade, gênero)
    - Comportamentais (alimentação, atividade física, hábitos)

    Esses dados permitem identificar padrões associados ao risco de obesidade.
    """)

# =========================================================
# 📊 ANÁLISE + MODELAGEM
# =========================================================
elif page == "📊 Análise + Modelagem":

    st.title("📊 Análise Prescritiva + Modelagem")

    if "Obesity_level" in df.columns and "IMC" in df.columns:

        st.subheader("Relação Idade x IMC")

        fig, ax = plt.subplots()
        sns.scatterplot(data=df, x="Age", y="IMC", hue="Obesity_level", ax=ax)
        st.pyplot(fig)

        st.markdown("""
        ### 🔎 Principais Insights

        - IMC é o principal indicador de obesidade  
        - Atividade física reduz risco  
        - Alimentação impacta diretamente  

        ### 🎯 Visão Prescritiva

        - Incentivar exercícios reduz obesidade  
        - Melhorar dieta é essencial  
        - Mudança de hábitos = maior impacto  
        """)

    else:
        st.error("Coluna Obesity_level não encontrada")

    st.subheader("🤖 Modelagem")

    st.markdown("""
    **Pipeline:**
    - Criação de IMC  
    - Remoção de variáveis redundantes  
    - Encoding  
    - Train/Test 80/20  
    - Random Forest  

    **Resultado:**
    - Acurácia ~97%

    **Limitação:**
    Forte dependência do IMC  
    """)

# =========================================================
# 📊 DASHBOARD EXECUTIVO (DEPOIS DA ANÁLISE)
# =========================================================
elif page == "📊 Dashboard Executivo":

    st.title("📊 Dashboard Executivo de Obesidade")

    col1, col2, col3 = st.columns(3)

    col1.metric("Total de Registros", len(df))
    col2.metric("Idade Média", round(df["Age"].mean(), 1))
    col3.metric("IMC Médio", round(df["IMC"].mean(), 1))

    st.divider()

    if "Obesity_level" in df.columns:

        c1, c2 = st.columns(2)

        with c1:
            st.subheader("Distribuição dos Níveis de Obesidade")
            fig, ax = plt.subplots()
            df["Obesity_level"].value_counts().plot(kind="bar", ax=ax)
            plt.xticks(rotation=45)
            st.pyplot(fig)

        with c2:
            st.subheader("IMC por Classe")
            fig, ax = plt.subplots()
            sns.boxplot(data=df, x="Obesity_level", y="IMC", ax=ax)
            plt.xticks(rotation=45)
            st.pyplot(fig)

    else:
        st.error("Coluna Obesity_level não encontrada")

    st.markdown("""
    ### 🧠 Insight Executivo

    - IMC explica bem os níveis de obesidade  
    - Há forte impacto comportamental  
    - Classes intermediárias são mais difíceis  
    """)

# =========================================================
# 🧠 CALCULADORA INTELIGENTE
# =========================================================
elif page == "🧠 Calculadora Inteligente":

    st.title("🧠 Simulador de Obesidade")

    idade = st.slider("Idade", 10, 80, 25)
    altura = st.number_input("Altura (m)", 1.4, 2.2, 1.70)
    peso = st.number_input("Peso (kg)", 40, 200, 70)

    atividade = st.slider("Atividade Física", 0, 3, 1)
    vegetais = st.slider("Consumo de Vegetais", 1, 3, 2)
    agua = st.slider("Consumo de Água", 1, 3, 2)

    imc = peso / (altura ** 2)

    st.metric("IMC", round(imc, 2))

    if st.button("Analisar"):

        if imc < 18.5:
            nivel = "Abaixo do peso"
        elif imc < 25:
            nivel = "Peso normal"
        elif imc < 30:
            nivel = "Sobrepeso"
        else:
            nivel = "Obesidade"

        st.subheader(f"Classificação: {nivel}")

        recomendacoes = []

        if atividade < 1:
            recomendacoes.append("Aumentar atividade física")

        if vegetais < 2:
            recomendacoes.append("Melhorar alimentação")

        if agua < 2:
            recomendacoes.append("Aumentar ingestão de água")

        if imc > 25:
            recomendacoes.append("Reduzir consumo calórico")

        st.subheader("Plano de Evolução")

        for r in recomendacoes:
            st.write(f"✅ {r}")

# =========================================================
# 💡 RECOMENDAÇÕES
# =========================================================
else:

    st.title("💡 Recomendações Estratégicas")

    st.markdown("""
    ### 🎯 Principais Fatores

    - Sedentarismo  
    - Alimentação inadequada  
    - Baixo consumo de água  

    ### 📊 Ações

    1. Incentivo à atividade física  
    2. Educação nutricional  
    3. Monitoramento contínuo  

    ### 🧠 Conclusão

    A obesidade pode ser prevista e reduzida com intervenções comportamentais.
    """)
