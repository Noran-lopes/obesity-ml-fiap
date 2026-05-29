# =========================================================
# IMPORTS
# =========================================================
import streamlit as st
import pandas as pd
import pickle
import seaborn as sns
import matplotlib.pyplot as plt

# =========================================================
# CONFIG
# =========================================================
st.set_page_config(page_title="Obesity Analytics", layout="wide")

# =========================================================
# MENU
# =========================================================
st.sidebar.markdown("""
### 🧠 Fluxo Analítico
1. Exploração  
2. Análise + Modelagem  
3. Dashboard  
4. Simulação  
5. Recomendações  
""")

page = st.sidebar.radio(
    "Navegação",
    ["Exploração", "Análise + Modelagem", "Dashboard", "Calculadora", "Recomendações"]
)

# =========================================================
# LOAD DATA (AJUSTE DEFINITIVO ✅)
# =========================================================
@st.cache_data
def load_data():
    df = pd.read_csv("Obesity.csv")

    # limpar colunas
    df.columns = df.columns.str.strip()

    # ✅ CORREÇÃO PRINCIPAL (SEU CASO REAL)
    if "Obesity" in df.columns:
        df.rename(columns={"Obesity": "Obesity_level"}, inplace=True)

    # criar IMC
    df["IMC"] = df["Weight"] / (df["Height"] ** 2)

    return df

df = load_data()

# =========================================================
# LOAD MODEL
# =========================================================
model = pickle.load(open("model.pkl", "rb"))
encoders = pickle.load(open("encoders.pkl", "rb"))


# =========================================================
# EXPLORAÇÃO
# =========================================================
if page == "Exploração":

    st.title("📁 Exploração dos Dados")

    # =========================
    # 📌 DESCRIÇÃO DO DATASET
    # =========================
    st.markdown("""
    ### 📊 Descrição dos Dados

    Este dataset contém informações voltadas à análise de obesidade, reunindo dados físicos, demográficos e comportamentais dos indivíduos.

    O objetivo é identificar padrões e fatores que influenciam a classificação de obesidade, permitindo análises preditivas e prescritivas para apoiar decisões médicas.
    """)

    # =========================
    # 📐 DIMENSÃO DO DATASET
    # =========================
    linhas, colunas = df.shape

    st.markdown(f"""
    ### 📐 Dimensão da Base
    - 🔹 Número de linhas: **{linhas}**
    - 🔹 Número de colunas: **{colunas}**

    Esses números indicam o volume de dados disponível para análise e modelagem.
    """)

    # =========================
    # 🧩 TIPOS DE VARIÁVEIS
    # =========================
    st.markdown("""
    ### 🧩 Tipos de Variáveis

    O dataset é composto por diferentes tipos de variáveis:

    - **Físicas:** Weight, Height, IMC  
    - **Demográficas:** Age, Gender  
    - **Comportamentais:** FAF, FCVC, CH2O, CALC, entre outras  
    - **Variável alvo:** Obesity_level  

    Essas variáveis permitem analisar tanto características biológicas quanto hábitos de vida.
    """)

    # =========================
    # 📋 VISUALIZAÇÃO DOS DADOS
    # =========================
    st.subheader("🔍 Amostra dos Dados")
    st.write(df.head())

    # =========================
    # 📈 ESTATÍSTICAS
    # =========================
    st.subheader("📊 Estatísticas Descritivas")
    st.write(df.describe())

    # =========================
    # 🧠 INTERPRETAÇÃO
    # =========================
    st.markdown("""
    ### 🧠 Interpretação

    A presença de variáveis comportamentais (como atividade física e alimentação) combinadas com dados físicos permite entender que a obesidade não depende apenas de um fator isolado, mas do conjunto de hábitos do indivíduo.

    Essa base é adequada para construção de modelos que vão além da previsão, permitindo recomendações práticas de melhoria.
    """)


# =========================================================
# ANÁLISE + MODELAGEM
# =========================================================
elif page == "Análise + Modelagem":

    st.title("📊 Análise + Modelagem")

    # =========================================================
    # 🔥 ORDEM CORRETA DAS CLASSES
    # =========================================================
    ordem = [
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

    mapa_labels = dict(zip(ordem, labels_pt))

    if "Obesity_level" in df.columns:

        df_plot = df.copy()

        # garantir ordem correta
        df_plot["Obesity_level"] = pd.Categorical(
            df_plot["Obesity_level"],
            categories=ordem,
            ordered=True
        )

        # =========================================================
        # 📊 ANÁLISE DESCRITIVA
        # =========================================================
        st.subheader("1️⃣ Distribuição de IMC por Nível de Obesidade")

        fig, ax = plt.subplots()
        sns.boxplot(
            data=df_plot,
            x="Obesity_level",
            y="IMC",
            order=ordem,
            ax=ax
        )

        ax.set_title("Distribuição do IMC por Classe de Obesidade")
        ax.set_xlabel("Classificação de Obesidade")
        ax.set_ylabel("IMC")

        ax.set_xticklabels(labels_pt, rotation=45)

        st.pyplot(fig)

        st.markdown("""
        ✅ **Interpretação:**
        - Observa-se crescimento consistente do IMC entre as classes  
        - Há separação clara entre níveis normais e obesidade  
        - Classes intermediárias apresentam maior sobreposição  

        🎯 **Conclusão:**
        O IMC é o principal indicador para classificação de obesidade.
        """)

        st.divider()

        # =========================================================
        # 🔍 ANÁLISE DIAGNÓSTICA
        # =========================================================
        st.subheader("2️⃣ Relação entre Idade e IMC")

        fig, ax = plt.subplots()

        sns.scatterplot(
            data=df_plot,
            x="Age",
            y="IMC",
            hue="Obesity_level",
            hue_order=ordem,
            ax=ax
        )

        ax.set_title("Relação entre Idade e IMC por Classe")
        ax.set_xlabel("Idade")
        ax.set_ylabel("IMC")

        st.pyplot(fig)

        st.markdown("""
        ✅ **Insight:**
        - A idade apresenta leve tendência de aumento do IMC  
        - Entretanto, não explica sozinha a obesidade  

        🎯 **Conclusão:**
        A obesidade é multifatorial e depende mais de hábitos do que idade isoladamente.
        """)

        st.divider()

        # =========================================================
        # 🤖 ANÁLISE PREDITIVA
        # =========================================================
        st.subheader("3️⃣ Modelagem Preditiva")

        st.markdown("""
        O modelo foi desenvolvido utilizando Random Forest.

        ✅ **Motivos da escolha:**
        - Captura relações não lineares  
        - Robusto a outliers  
        - Reduz overfitting  

        ✅ **Pipeline aplicada:**
        - Criação do IMC  
        - Remoção de Height e Weight (evita leakage)  
        - Encoding de variáveis categóricas  
        - Split treino/teste (80/20)  

        ✅ **Performance:**
        - Acurácia aproximada de 97%  

        🎯 **Interpretação:**
        O modelo apresenta alta capacidade de generalização.
        """)

        st.divider()

        # =========================================================
        # 💡 ANÁLISE PRESCRITIVA
        # =========================================================
        st.subheader("4️⃣ Análise Prescritiva")

        st.markdown("""
        ### 🔍 Fatores principais identificados:

        - IMC elevado  
        - Baixa atividade física (FAF)  
        - Baixo consumo de vegetais (FCVC)  
        - Consumo frequente de alimentos calóricos (FAVC)  

        ### ✅ Recomendações:

        - Aumentar frequência de atividade física  
        - Melhorar qualidade alimentar  
        - Reduzir alimentos altamente calóricos  
        - Aumentar consumo de água  

        🎯 **Conclusão final:**

        A obesidade pode ser significativamente reduzida através de mudanças comportamentais mensuráveis, sendo possível atuar preventivamente com base nos dados.
        """)

    else:
        st.error(f"Coluna não encontrada: {df.columns}")
# =========================================================
# DASHBOARD
# =========================================================
elif page == "Dashboard":

    st.title("📊 Dashboard Executivo")

    # =========================
    # KPIs
    # =========================
    col1, col2, col3 = st.columns(3)

    col1.metric("Total de Registros", len(df))
    col2.metric("IMC Médio", round(df["IMC"].mean(), 2))
    col3.metric("Idade Média", round(df["Age"].mean(), 1))

    st.divider()

    # =========================
    # 🔥 ORDEM DAS CLASSES
    # =========================
    ordem = [
        "Insufficient_Weight",
        "Normal_Weight",
        "Overweight_Level_I",
        "Overweight_Level_II",
        "Obesity_Type_I",
        "Obesity_Type_II",
        "Obesity_Type_III"
    ]

    labels_pt = [
        "Abaixo peso",
        "Normal",
        "Sobrepeso I",
        "Sobrepeso II",
        "Obesidade I",
        "Obesidade II",
        "Obesidade III"
    ]

    # =========================
    # LINHA 1
    # =========================
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Distribuição de Obesidade")

        if "Obesity_level" in df.columns:
            df_plot = df.copy()

            df_plot["Obesity_level"] = pd.Categorical(
                df_plot["Obesity_level"],
                categories=ordem,
                ordered=True
            )

            contagem = df_plot["Obesity_level"].value_counts().sort_index()

            fig, ax = plt.subplots()
            contagem.plot(kind="bar", ax=ax)

            ax.set_xticklabels(labels_pt, rotation=45)
            ax.set_ylabel("Quantidade")

            st.pyplot(fig)

    with col2:
        st.subheader("IMC por Classe")

        fig, ax = plt.subplots()
        sns.boxplot(data=df_plot, x="Obesity_level", y="IMC", order=ordem, ax=ax)

        ax.set_xticklabels(labels_pt, rotation=45)
        st.pyplot(fig)

    st.divider()

    # =========================
    # LINHA 2
    # =========================
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Distribuição de Idade")

        fig, ax = plt.subplots()
        df["Age"].hist(bins=20, ax=ax)
        ax.set_xlabel("Idade")
        ax.set_ylabel("Quantidade")
        st.pyplot(fig)

    with col2:
        st.subheader("Distribuição do IMC")

        fig, ax = plt.subplots()
        df["IMC"].hist(bins=20, ax=ax)
        ax.set_xlabel("IMC")
        st.pyplot(fig)

    st.divider()

    # =========================
    # 🔥 LINHA 3 (CORRIGIDA)
    # =========================
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Atividade Física (FAF)")

        ordem_faf = sorted(df["FAF"].unique())

        fig, ax = plt.subplots()
        df["FAF"].value_counts().sort_index().plot(kind="bar", ax=ax)

        ax.set_xlabel("Nível de Atividade (0 = baixo | 3 = alto)")
        ax.set_ylabel("Quantidade")

        st.pyplot(fig)

    with col2:
        st.subheader("Consumo de Vegetais (FCVC)")

        ordem_fcvc = sorted(df["FCVC"].unique())

        fig, ax = plt.subplots()
        df["FCVC"].value_counts().sort_index().plot(kind="bar", ax=ax)

        ax.set_xlabel("Frequência (1 = baixo | 3 = alto)")
        ax.set_ylabel("Quantidade")

        st.pyplot(fig)

    st.divider()

    # =========================
    # LINHA 4
    # =========================
    st.subheader("Correlação entre Variáveis")

    fig, ax = plt.subplots()

    corr = df[["Age", "IMC", "FAF", "FCVC", "CH2O"]].corr()

    sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)

    st.pyplot(fig)
# =========================================================
# CALCULADORA COMPLETA
# =========================================================
elif page == "Calculadora":

    st.title("🧠 Simulador Inteligente de Obesidade")

    # =========================
    # INPUTS
    # =========================
    col1, col2 = st.columns(2)

    with col1:
        idade = st.slider("Idade", 10, 80, 25)
        altura = st.number_input("Altura (m)", 1.4, 2.2, 1.70)
        peso = st.number_input("Peso (kg)", 40, 200, 70)
        genero = st.selectbox("Gênero", ["Male", "Female"])

    with col2:
        faf = st.slider("Atividade Física (0=baixa | 3=alta)", 0, 3, 1)
        fcvc = st.slider("Consumo de Vegetais (1=baixo | 3=alto)", 1, 3, 2)
        ch2o = st.slider("Consumo de Água (1=baixo | 3=alto)", 1, 3, 2)
        favc = st.selectbox("Consome alimentos calóricos?", ["yes", "no"])

    # valores padrão (compatíveis com modelo)
    ncp = 3
    caec = "Sometimes"
    smoke = "no"
    scc = "no"
    tue = 1
    calc = "Sometimes"
    mtrans = "Public_Transportation"

    # =========================
    # IMC
    # =========================
    imc = peso / (altura ** 2)
    st.metric("📊 Seu IMC", round(imc, 2))

    if st.button("🔍 Analisar"):

        # =========================
        # INPUT MODELO
        # =========================
        input_dict = {
            "Gender": genero,
            "Age": idade,
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

        input_df = pd.DataFrame([input_dict])

        # aplicar encoders
        for col, enc in encoders.items():
            if col in input_df.columns:
                input_df[col] = enc.transform(input_df[col])

        input_df = input_df.reindex(columns=model.feature_names_in_, fill_value=0)

        pred = model.predict(input_df)[0]

        # =========================
        # TRADUÇÃO RESULTADO
        # =========================
        labels_map = {
            "Insufficient_Weight": "Abaixo do peso",
            "Normal_Weight": "Peso normal",
            "Overweight_Level_I": "Sobrepeso I",
            "Overweight_Level_II": "Sobrepeso II",
            "Obesity_Type_I": "Obesidade I",
            "Obesity_Type_II": "Obesidade II",
            "Obesity_Type_III": "Obesidade III",
        }

        resultado = labels_map.get(pred, pred)

        st.subheader(f"🏥 Classificação: {resultado}")

        st.divider()

        # =========================
        # 🧠 DIAGNÓSTICO
        # =========================
        st.subheader("🧠 Diagnóstico do Perfil")

        riscos = []

        if imc > 25:
            riscos.append("IMC elevado")

        if faf < 1:
            riscos.append("Baixa atividade física")

        if fcvc < 2:
            riscos.append("Baixo consumo de vegetais")

        if ch2o < 2:
            riscos.append("Baixo consumo de água")

        if favc == "yes":
            riscos.append("Alto consumo de alimentos calóricos")

        if riscos:
            for r in riscos:
                st.write(f"⚠️ {r}")
        else:
            st.success("✅ Perfil saudável identificado")

        st.divider()

        # =========================
        # 📊 ESTADO ATUAL (IMC)
        # =========================
        st.subheader("📊 Estado Atual")

        if imc < 18.5:
            status_imc = "Abaixo do peso"
        elif imc < 25:
            status_imc = "Saudável"
        elif imc < 30:
            status_imc = "Sobrepeso"
        else:
            status_imc = "Obesidade"

        st.write(f"Classificação IMC: **{status_imc}**")

        # =========================
        # 🎯 PESO IDEAL
        # =========================
        st.subheader("🎯 Faixa de Peso Ideal")

        peso_min = 18.5 * (altura ** 2)
        peso_max = 24.9 * (altura ** 2)

        st.write(f"Peso ideal mínimo: **{round(peso_min,1)} kg**")
        st.write(f"Peso ideal máximo: **{round(peso_max,1)} kg**")

        # =========================
        # 📉 AJUSTE NECESSÁRIO
        # =========================
        st.subheader("📉 Ajuste Necessário")

        if peso > peso_max:
            excesso = peso - peso_max
            st.warning(f"Você precisa perder cerca de **{round(excesso,1)} kg**")

        elif peso < peso_min:
            falta = peso_min - peso
            st.warning(f"Você precisa ganhar cerca de **{round(falta,1)} kg**")

        else:
            st.success("Você está dentro da faixa ideal ✅")

        st.divider()

        # =========================
        # 📈 PLANO DE EVOLUÇÃO
        # =========================
        st.subheader("📈 Plano de Evolução")

        if peso > peso_max:
            st.write("🎯 Redução gradual de peso")
            st.write("📌 Meta saudável: 0.5 a 1 kg por semana")

        elif peso < peso_min:
            st.write("🎯 Ganho de peso controlado")
            st.write("📌 Ajustar ingestão calórica")

        else:
            st.write("🎯 Manutenção do peso atual")

        if faf < 1:
            st.write("✅ Aumentar atividade física")

        if fcvc < 2:
            st.write("✅ Melhorar alimentação")

        if ch2o < 2:
            st.write("✅ Beber mais água")

        if favc == "yes":
            st.write("✅ Reduzir alimentos calóricos")

        st.markdown("""
        ---
        🎯 **Resumo:**  
        Ajustes no comportamento e hábitos são fundamentais para evolução da saúde.
        """)
# =========================================================
# RECOMENDAÇÕES
# =========================================================
else:

    st.title("💡 Recomendações Estratégicas")

    st.markdown("""
    ✅ Prática regular de atividade física  
    ✅ Alimentação balanceada  
    ✅ Redução de calorias  
    ✅ Consumo adequado de água  

    📊 A obesidade pode ser prevenida com mudanças no comportamento.
    """)
