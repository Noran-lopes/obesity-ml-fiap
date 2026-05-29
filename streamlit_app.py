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
# ANÁLISE + MODELAGEM FINAL COMPLETA (SEM ERRO)
# =========================================================
elif page == "Análise + Modelagem":

    import numpy as np
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import confusion_matrix, accuracy_score

    st.title("📊 Análise + Modelagem")

    # =========================================================
    # 🔥 ORDEM DAS CLASSES
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

    df_plot = df.copy()

    if "Obesity_level" in df_plot.columns:
        df_plot["Obesity_level"] = pd.Categorical(
            df_plot["Obesity_level"],
            categories=ordem,
            ordered=True
        )

    # =========================================================
    # 1️⃣ DESCRITIVA
    # =========================================================
    st.subheader("1️⃣ Análise Descritiva")

    fig, ax = plt.subplots()
    sns.boxplot(data=df_plot, x="Obesity_level", y="IMC", order=ordem, ax=ax)
    ax.set_xticklabels(labels_pt, rotation=45)

    st.pyplot(fig)

    st.markdown("""
    O IMC cresce progressivamente entre os níveis de obesidade, indicando forte capacidade de separação das classes.
    """)

    st.divider()

    # =========================================================
    # 2️⃣ DIAGNÓSTICA
    # =========================================================
    st.subheader("2️⃣ Análise Diagnóstica")

    fig, ax = plt.subplots()
    sns.scatterplot(data=df_plot, x="Age", y="IMC", hue="Obesity_level", ax=ax)

    st.pyplot(fig)

    st.markdown("""
    A idade apresenta influência moderada, mas não é determinante isolada — reforçando caráter multifatorial da obesidade.
    """)

    st.divider()

    # =========================================================
    # 3️⃣ MODELAGEM
    # =========================================================
    st.subheader("3️⃣ Modelagem Preditiva")

    st.markdown("""
    Modelo utilizado: **Random Forest**

    - Captura relações não lineares  
    - Alta robustez  
    - Excelente desempenho em classificação  
    """)

    # =========================================================
    # 🔥 PIPELINE CORRETA
    # =========================================================
    df_model = df.copy()

    df_model = df_model.drop(["Weight", "Height"], axis=1)

    # encoding completo
    for col, enc in encoders.items():
        if col in df_model.columns:
            df_model[col] = enc.transform(df_model[col])

    # separar X e y
    X = df_model.drop("Obesity_level", axis=1)
    y = df_model["Obesity_level"]

    # ✅ CORREÇÃO CRÍTICA (evita erro da matriz!)
    target_encoder = encoders["Obesity_level"]
    y = target_encoder.transform(y)

    # split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # =========================================================
    # ✅ ACURÁCIA
    # =========================================================
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)

    st.metric("✅ Acurácia do Modelo", f"{round(acc*100,2)}%")

    # =========================================================
    # 📊 MATRIZ DE CONFUSÃO
    # =========================================================
    st.subheader("📊 Matriz de Confusão")

    cm = confusion_matrix(y_test, y_pred)

    fig, ax = plt.subplots()

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        ax=ax
    )

    ax.set_xlabel("Previsto")
    ax.set_ylabel("Real")

    st.pyplot(fig)

    st.markdown("""
    - Alta concentração de acertos na diagonal  
    - Erros entre classes próximas  
    - Excelente desempenho geral  

    ✅ Modelo confiável para classificação.
    """)

    st.divider()

    # =========================================================
    # 📈 FEATURE IMPORTANCE
    # =========================================================
    st.subheader("📈 Importância das Variáveis")

    importances = pd.DataFrame({
        "Variável": X.columns,
        "Importância": model.feature_importances_
    }).sort_values(by="Importância", ascending=False)

    fig, ax = plt.subplots()

    sns.barplot(
        data=importances.head(10),
        x="Importância",
        y="Variável",
        ax=ax
    )

    st.pyplot(fig)

    st.markdown("""
    - IMC é o fator mais relevante  
    - Variáveis comportamentais impactam fortemente  
    - Modelo analisa múltiplas dimensões  

    ✅ A obesidade é multifatorial.
    """)

    st.divider()

    # =========================================================
    # 💡 PRESCRITIVA
    # =========================================================
    st.subheader("4️⃣ Análise Prescritiva")

    st.markdown("""
    🔎 Fatores críticos:

    - Baixa atividade física  
    - Má alimentação  
    - Alto consumo calórico  

    ✅ Recomendações:

    - Aumentar atividade física  
    - Melhorar alimentação  
    - Reduzir calorias  

    🎯 A obesidade pode ser reduzida com mudanças comportamentais.
    """)

# =========================================================
# DASHBOARD EXECUTIVO FINAL
# =========================================================
elif page == "Dashboard":

    st.title("📊 Dashboard Executivo de Obesidade")

    # =========================
    # 📊 KPIs
    # =========================
    col1, col2, col3 = st.columns(3)

    col1.metric("Total de Registros", len(df))
    col2.metric("Idade Média", round(df["Age"].mean(), 1))
    col3.metric("IMC Médio", round(df["IMC"].mean(), 2))

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

    df_plot = df.copy()

    if "Obesity_level" in df_plot.columns:
        df_plot["Obesity_level"] = pd.Categorical(
            df_plot["Obesity_level"],
            categories=ordem,
            ordered=True
        )

    # =========================================================
    # 📊 LINHA 1 — DISTRIBUIÇÃO + IMC
    # =========================================================
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Distribuição dos Níveis de Obesidade")

        contagem = df_plot["Obesity_level"].value_counts().sort_index()

        fig, ax = plt.subplots()
        contagem.plot(kind="bar", ax=ax)

        ax.set_xticklabels(labels_pt, rotation=45)
        ax.set_ylabel("Quantidade")
        ax.set_xlabel("Classificação")

        st.pyplot(fig)

    with col2:
        st.subheader("IMC por Nível de Obesidade")

        fig, ax = plt.subplots()
        sns.boxplot(data=df_plot, x="Obesity_level", y="IMC", order=ordem, ax=ax)

        ax.set_xticklabels(labels_pt, rotation=45)
        ax.set_ylabel("IMC")

        st.pyplot(fig)

    st.markdown("""
    ### 🧠 Interpretação

    A distribuição dos níveis de obesidade evidencia a concentração da população em diferentes faixas de risco.

    🔎 **Insights:**
    - Há maior concentração em níveis intermediários  
    - O IMC cresce de forma consistente entre as categorias  
    - Existe separação clara entre níveis saudáveis e obesidade  

    🎯 **Conclusão:**
    O IMC é um excelente indicador para segmentação dos níveis de obesidade.
    """)

    st.divider()

    # =========================================================
    # 📊 LINHA 2 — DISTRIBUIÇÕES
    # =========================================================
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
        ax.set_ylabel("Quantidade")

        st.pyplot(fig)

    st.markdown("""
    ### 🧠 Interpretação

    As distribuições permitem entender o perfil da população analisada.

    🔎 **Insights:**
    - O IMC apresenta variação significativa entre indivíduos  
    - A idade se concentra em determinadas faixas  

    🎯 **Conclusão:**
    A variabilidade reforça a necessidade de análise individualizada no diagnóstico de obesidade.
    """)

    st.divider()

    # =========================================================
    # 📊 LINHA 3 — COMPORTAMENTO (CORRIGIDO)
    # =========================================================
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Atividade Física (FAF)")

        df_plot["FAF"] = df_plot["FAF"].round(0).astype(int)
        faf_counts = df_plot["FAF"].value_counts().sort_index()

        fig, ax = plt.subplots()
        faf_counts.plot(kind="bar", ax=ax)

        ax.set_xticks(range(len(faf_counts)))
        ax.set_xticklabels([
            "Muito baixa (0)",
            "Baixa (1)",
            "Moderada (2)",
            "Alta (3)"
        ])

        ax.set_ylabel("Quantidade")
        ax.set_xlabel("Nível de atividade")

        st.pyplot(fig)

    with col2:
        st.subheader("Consumo de Vegetais (FCVC)")

        df_plot["FCVC"] = df_plot["FCVC"].round(0).astype(int)
        fcvc_counts = df_plot["FCVC"].value_counts().sort_index()

        fig, ax = plt.subplots()
        fcvc_counts.plot(kind="bar", ax=ax)

        ax.set_xticks(range(len(fcvc_counts)))
        ax.set_xticklabels([
            "Baixo (1)",
            "Médio (2)",
            "Alto (3)"
        ])

        ax.set_ylabel("Quantidade")
        ax.set_xlabel("Frequência de consumo")

        st.pyplot(fig)

    st.markdown("""
    ### 🧠 Interpretação

    Os hábitos comportamentais mostram padrões importantes relacionados à saúde.

    🔎 **Insights:**
    - Grande parte da população apresenta níveis baixos ou moderados de atividade física  
    - O consumo de vegetais não está predominante em níveis elevados  

    🎯 **Conclusão:**
    O comportamento (atividade física e alimentação) é um dos principais fatores associados ao risco de obesidade.
    """)

    st.divider()

    # =========================================================
    # 📊 LINHA 4 — CORRELAÇÃO
    # =========================================================
    st.subheader("Correlação entre Variáveis")

    fig, ax = plt.subplots()

    corr = df[["Age", "IMC", "FAF", "FCVC", "CH2O"]].corr()
    sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)

    st.pyplot(fig)

    st.markdown("""
    ### 🧠 Interpretação

    A matriz de correlação evidencia as relações entre variáveis numéricas.

    🔎 **Insights:**
    - IMC possui correlação positiva com idade  
    - Atividade física apresenta relação negativa com IMC  
    - As correlações são moderadas  

    🎯 **Conclusão:**
    A obesidade é um fenômeno multifatorial, não sendo explicada por uma única variável isolada.
    """)
# =========================================================
# CALCULADORA UX/UI MELHORADA
# =========================================================
elif page == "Calculadora":

    st.title("🧠 Simulador Inteligente de Saúde")

    st.markdown("Preencha seus dados para receber uma análise completa do seu estado atual e recomendações personalizadas.")

    # =========================
    # 🧍 DADOS DO USUÁRIO
    # =========================
    st.subheader("🧍 Dados Pessoais")

    col1, col2 = st.columns(2)

    with col1:
        idade = st.slider("Idade", 10, 80, 25)
        genero = st.selectbox("Gênero", ["Male", "Female"])

    with col2:
        altura = st.number_input("Altura (m)", 1.4, 2.2, 1.70)
        peso = st.number_input("Peso (kg)", 40, 200, 70)

    # =========================
    # 🏃 HÁBITOS
    # =========================
    st.subheader("🏃 Hábitos")

    col1, col2 = st.columns(2)

    with col1:
        faf = st.slider("Atividade Física", 0, 3, 1)
        fcvc = st.slider("Consumo de Vegetais", 1, 3, 2)

    with col2:
        ch2o = st.slider("Consumo de Água", 1, 3, 2)
        favc = st.selectbox("Consumo de alimentos calóricos", ["yes", "no"])

    # =========================
    # 📊 IMC (VISUAL)
    # =========================
    imc = peso / (altura ** 2)

    if imc < 18.5:
        st.info(f"📊 IMC: {round(imc,2)} (Abaixo do peso)")
    elif imc < 25:
        st.success(f"📊 IMC: {round(imc,2)} (Saudável)")
    elif imc < 30:
        st.warning(f"📊 IMC: {round(imc,2)} (Sobrepeso)")
    else:
        st.error(f"📊 IMC: {round(imc,2)} (Obesidade)")

    # valores padrão modelo
    ncp, caec, smoke = 3, "Sometimes", "no"
    scc, tue, calc = "no", 1, "Sometimes"
    mtrans = "Public_Transportation"

    st.divider()

    # =========================
    # 🔍 BOTÃO
    # =========================
    if st.button("🔍 Gerar Análise Completa"):

        # =========================
        # MODELO
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

        for col, enc in encoders.items():
            if col in input_df.columns:
                input_df[col] = enc.transform(input_df[col])

        input_df = input_df.reindex(columns=model.feature_names_in_, fill_value=0)

        pred = model.predict(input_df)[0]

        labels_map = {
            "Insufficient_Weight": "Abaixo do peso",
            "Normal_Weight": "Peso normal",
            "Overweight_Level_I": "Sobrepeso I",
            "Overweight_Level_II": "Sobrepeso II",
            "Obesity_Type_I": "Obesidade I",
            "Obesity_Type_II": "Obesidade II",
            "Obesity_Type_III": "Obesidade III"
        }

        resultado = labels_map.get(pred, pred)

        # =========================
        # RESULTADO PRINCIPAL
        # =========================
        st.success(f"🏥 Classificação: {resultado}")

        st.divider()

        # =========================
        # 🎯 PESO IDEAL
        # =========================
        st.subheader("🎯 Meta de Peso Saudável")

        peso_min = 18.5 * (altura ** 2)
        peso_max = 24.9 * (altura ** 2)

        col1, col2 = st.columns(2)

        col1.metric("Peso mínimo ideal", f"{round(peso_min,1)} kg")
        col2.metric("Peso máximo ideal", f"{round(peso_max,1)} kg")

        # =========================
        # 📉 AJUSTE
        # =========================
        st.subheader("📉 Ajuste Necessário")

        if peso > peso_max:
            excesso = peso - peso_max
            st.error(f"Você precisa reduzir aproximadamente **{round(excesso,1)} kg**")

        elif peso < peso_min:
            falta = peso_min - peso
            st.warning(f"Você precisa ganhar aproximadamente **{round(falta,1)} kg**")

        else:
            st.success("✅ Você está dentro da faixa ideal")

        st.divider()

        # =========================
        # 🧠 DIAGNÓSTICO
        # =========================
        st.subheader("🧠 Fatores de Atenção")

        riscos = []

        if faf < 1:
            riscos.append("Baixa atividade física")

        if fcvc < 2:
            riscos.append("Baixa ingestão de vegetais")

        if ch2o < 2:
            riscos.append("Baixa ingestão de água")

        if favc == "yes":
            riscos.append("Alto consumo calórico")

        if riscos:
            for r in riscos:
                st.warning(r)
        else:
            st.success("✅ Bons hábitos identificados")

        st.divider()

        # =========================
        # 📈 PLANO
        # =========================
        st.subheader("📈 Plano de Evolução")

        if peso > peso_max:
            st.write("🎯 Reduzir peso gradualmente")

        if faf < 1:
            st.write("✅ Aumentar atividade física")

        if fcvc < 2:
            st.write("✅ Melhorar alimentação")

        if ch2o < 2:
            st.write("✅ Aumentar consumo de água")

        if favc == "yes":
            st.write("✅ Reduzir alimentos calóricos")

        st.divider()

        # =========================
        # ✅ RESUMO FINAL
        # =========================
        st.subheader("✅ Resumo")

        st.info(f"""
        Você está classificado como: **{resultado}**

        Para melhorar sua condição:
        - Ajuste seus hábitos
        - Aproxime-se da faixa de peso ideal
        - Mantenha consistência no tempo
        """)
# =========================================================
# RECOMENDAÇÕES (BLOG)
# =========================================================
else:

    st.title("📚 Conteúdos sobre Bem-estar e Saúde")

    st.markdown("""
    Explore conteúdos educativos sobre qualidade de vida, alimentação e hábitos saudáveis.
    """)

    # =========================
    # MENU DE CATEGORIAS
    # =========================
    categoria = st.selectbox(
        "Escolha um tema:",
        ["Alimentação", "Exercícios", "Hidratação", "Hábitos Saudáveis"]
    )

    st.divider()

    # =========================================================
    # 🥗 ALIMENTAÇÃO
    # =========================================================
    if categoria == "Alimentação":

        st.subheader("🥗 Alimentação Saudável")

        with st.expander("✅ Como melhorar sua alimentação"):
            st.write("""
            - Inclua vegetais diariamente  
            - Evite alimentos ultraprocessados  
            - Prefira alimentos naturais  

            Uma alimentação equilibrada é um dos principais fatores na prevenção da obesidade.
            """)

        with st.expander("🚫 Evitar alimentos calóricos"):
            st.write("""
            - Reduza refrigerantes e doces  
            - Evite fast food frequente  
            - Controle porções  

            O consumo frequente de alimentos calóricos está relacionado ao aumento do IMC.
            """)

    # =========================================================
    # 🏃 EXERCÍCIOS
    # =========================================================
    elif categoria == "Exercícios":

        st.subheader("🏃 Atividade Física")

        with st.expander("✅ Benefícios da atividade física"):
            st.write("""
            - Redução de peso  
            - Melhora da saúde cardiovascular  
            - Aumento da qualidade de vida  
            """)

        with st.expander("📅 Quantidade recomendada"):
            st.write("""
            - 150 minutos por semana  
            - Pode dividir em pequenas sessões diárias  
            """)

    # =========================================================
    # 💧 HIDRATAÇÃO
    # =========================================================
    elif categoria == "Hidratação":

        st.subheader("💧 Hidratação")

        with st.expander("✅ Importância da água"):
            st.write("""
            - Regula o metabolismo  
            - Auxilia na digestão  
            - Ajuda no controle de peso  
            """)

        with st.expander("📊 Quanto devo beber?"):
            st.write("""
            - Cerca de 2 litros por dia  
            - Pode variar por peso e atividade física  
            """)

    # =========================================================
    # 🧠 HÁBITOS
    # =========================================================
    elif categoria == "Hábitos Saudáveis":

        st.subheader("🧠 Hábitos de Vida")

        with st.expander("✅ Sono de qualidade"):
            st.write("""
            - Dormir bem regula hormônios  
            - Ajuda no controle de peso  
            """)

        with st.expander("📉 Reduzir sedentarismo"):
            st.write("""
            - Evite ficar longos períodos sentado  
            - Movimente-se ao longo do dia  
            """)
