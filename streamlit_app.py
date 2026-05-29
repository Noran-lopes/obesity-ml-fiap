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
# 📊 MENU
# =========================================================
page = st.sidebar.radio(
    "📊 Navegação",
    [
        "📊 Dashboard Executivo",
        "📁 Exploração dos Dados",
        "📊 Análise + Modelagem",
        "🧠 Calculadora Inteligente",
        "💡 Recomendações Estratégicas"
    ]
)

# =========================================================
# 📁 LOAD DATA
# =========================================================
@st.cache_data
def load_data():
    df = pd.read_csv("Obesity.csv")
    df["IMC"] = df["Weight"] / (df["Height"] ** 2)
    return df

df = load_data()

# =========================================================
# 🤖 LOAD MODEL (se existir)
# =========================================================
try:
    model = pickle.load(open("model.pkl", "rb"))
    encoders = pickle.load(open("encoders.pkl", "rb"))
    model_loaded = True
except:
    model_loaded = False

# =========================================================
# 📊 DASHBOARD EXECUTIVO
# =========================================================
if page == "📊 Dashboard Executivo":

    st.title("📊 Dashboard Executivo de Obesidade")

    col1, col2, col3 = st.columns(3)
    col1.metric("Total de Registros", len(df))
    col2.metric("Idade Média", round(df["Age"].mean(), 1))
    col3.metric("IMC Médio", round(df["IMC"].mean(), 1))

    st.divider()

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
        sns.boxplot(x=df["Obesity_level"], y=df["IMC"], ax=ax)
        plt.xticks(rotation=45)
        st.pyplot(fig)

    st.markdown("""
    ### 🧠 Insight Executivo
    - O IMC é o principal fator de separação entre os níveis de obesidade  
    - Classes intermediárias apresentam maior sobreposição  
    - Há forte influência de comportamento nos níveis mais altos
    """)

# =========================================================
# 📁 EXPLORAÇÃO DOS DADOS
# =========================================================
elif page == "📁 Exploração dos Dados":

    st.title("📁 Entendimento dos Dados")

    st.subheader("📌 Visão Geral")
    st.write(df.head())

    st.subheader("📊 Tipos de Variáveis")
    st.write(df.dtypes)

    st.subheader("📊 Estatísticas Descritivas")
    st.write(df.describe())

    st.markdown("""
    ### 🧠 Interpretação

    A base contém:

    - Variáveis físicas: IMC, peso, altura  
    - Demográficas: idade e gênero  
    - Comportamentais: alimentação, atividade física e hábitos  

    Esses fatores permitem identificar padrões relacionados ao risco de obesidade.
    """)

# =========================================================
# 📊 ANÁLISE + MODELAGEM
# =========================================================
elif page == "📊 Análise + Modelagem":

    st.title("📊 Análise Prescritiva + Modelagem")

    st.subheader("Relação Idade x IMC")

    fig, ax = plt.subplots()
    sns.scatterplot(data=df, x="Age", y="IMC", hue="Obesity_level", ax=ax)
    st.pyplot(fig)

    st.markdown("""
    ### 🔎 Principais Insights

    - O IMC é o principal indicador de obesidade  
    - Baixa atividade física aumenta o risco  
    - Dietas mais saudáveis reduzem progressão  

    ### 🎯 Interpretação Prescritiva

    - Aumentar atividade física reduz risco diretamente  
    - Alimentação equilibrada é fator crítico  
    - Mudanças comportamentais são decisivas  
    """)

    st.subheader("🤖 Modelagem")

    st.markdown("""
    ### 📌 Pipeline

    - Criação da variável IMC  
    - Remoção de Height e Weight (evitar leakage)  
    - Encoding de variáveis categóricas  
    - Divisão treino/teste (80/20)  
    - Modelo: Random Forest  

    ### ✅ Resultado

    - Acurácia aproximada: **97%**  

    ### 🧠 Interpretação

    O modelo apresenta alta capacidade preditiva, com maior dificuldade na distinção entre classes intermediárias.

    ⚠️ Limitação:
    Forte dependência do IMC.
    """)

# =========================================================
# 🧠 CALCULADORA INTELIGENTE
# =========================================================
elif page == "🧠 Calculadora Inteligente":

    st.title("🧠 Simulação Inteligente de Obesidade")

    idade = st.slider("Idade", 10, 80, 25)
    altura = st.number_input("Altura (m)", 1.4, 2.2, 1.70)
    peso = st.number_input("Peso (kg)", 40, 200, 70)

    atividade = st.slider("Atividade Física (0-3)", 0, 3, 1)
    vegetais = st.slider("Consumo de Vegetais (1-3)", 1, 3, 2)
    agua = st.slider("Consumo de Água (1-3)", 1, 3, 2)

    imc = peso / (altura ** 2)

    st.metric("Seu IMC", round(imc, 2))

    if st.button("🔍 Analisar"):

        # Classificação simples (fallback)
        if imc < 18.5:
            nivel = "Abaixo do peso"
        elif imc < 25:
            nivel = "Peso Normal"
        elif imc < 30:
            nivel = "Sobrepeso"
        else:
            nivel = "Obesidade"

        st.subheader(f"📊 Classificação: {nivel}")

        recomendacoes = []

        if atividade < 1:
            recomendacoes.append("Aumentar frequência de atividade física")

        if vegetais < 2:
            recomendacoes.append("Melhorar consumo de vegetais")

        if agua < 2:
            recomendacoes.append("Aumentar ingestão de água")

        if imc > 25:
            recomendacoes.append("Reduzir ingestão calórica")

        st.subheader("📈 Plano de Evolução")

        for r in recomendacoes:
            st.write(f"✅ {r}")

# =========================================================
# 💡 RECOMENDAÇÕES
# =========================================================
else:

    st.title("💡 Recomendações Estratégicas")

    st.markdown("""
    ### 🎯 Principais Fatores de Risco

    - Sedentarismo  
    - Má alimentação  
    - Baixo consumo de água  
    - Alta ingestão calórica  

    ### 📊 Recomendações

    1. Incentivar atividade física regular  
    2. Programas de educação nutricional  
    3. Monitoramento de pacientes  
    4. Acompanhamento preventivo  

    ### 🧠 Conclusão

    A obesidade é altamente influenciada por comportamento e pode ser prevenida através de intervenções direcionadas.
    """)
