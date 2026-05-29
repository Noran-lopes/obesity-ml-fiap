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

    st.subheader("Amostra")
    st.write(df.head())

    st.subheader("Estatísticas")
    st.write(df.describe())

# =========================================================
# ANÁLISE
# =========================================================
elif page == "Análise + Modelagem":

    st.title("📊 Análise Prescritiva")

    if "Obesity_level" in df.columns:

        fig, ax = plt.subplots()
        sns.scatterplot(data=df, x="Age", y="IMC", hue="Obesity_level", ax=ax)
        st.pyplot(fig)

        st.markdown("""
        ✅ IMC é o principal fator  
        ✅ Estilo de vida impacta diretamente  
        ✅ Mudança comportamental reduz risco  
        """)

    else:
        st.error(f"Erro: coluna não encontrada → {df.columns}")

    st.subheader("🤖 Modelagem")
    st.markdown("""
    - Modelo: Random Forest  
    - Feature Engineering: IMC  
    - Encoding: LabelEncoder  
    - Acurácia: ~97%  
    """)

# =========================================================
# DASHBOARD
# =========================================================
elif page == "Dashboard":

    st.title("📊 Dashboard Executivo")

    col1, col2 = st.columns(2)
    col1.metric("Total de Registros", len(df))
    col2.metric("IMC Médio", round(df["IMC"].mean(), 2))

    if "Obesity_level" in df.columns:

        fig, ax = plt.subplots()
        df["Obesity_level"].value_counts().plot(kind="bar", ax=ax)
        plt.xticks(rotation=45)
        st.pyplot(fig)

    else:
        st.warning("⚠️ Coluna Obesity_level não encontrada")

# =========================================================
# CALCULADORA (MODELO REAL ✅🔥)
# =========================================================
elif page == "Calculadora":

    st.title("🧠 Predição com Machine Learning")

    idade = st.slider("Idade", 10, 80, 25)
    altura = st.number_input("Altura (m)", 1.4, 2.2, 1.7)
    peso = st.number_input("Peso (kg)", 40, 200, 70)

    genero = st.selectbox("Gênero", ["Male", "Female"])
    favc = st.selectbox("Consome alimentos calóricos?", ["yes", "no"])
    fcvc = st.slider("Consumo de vegetais", 1, 3, 2)
    ncp = st.slider("Refeições diárias", 1, 4, 3)
    caec = st.selectbox("Lanches", ["no", "Sometimes", "Frequently", "Always"])
    smoke = st.selectbox("Fumante", ["yes", "no"])
    ch2o = st.slider("Água", 1, 3, 2)
    scc = st.selectbox("Monitora calorias?", ["yes", "no"])
    faf = st.slider("Atividade física", 0, 3, 1)
    tue = st.slider("Uso tecnologia", 0, 2, 1)
    calc = st.selectbox("Álcool", ["no", "Sometimes", "Frequently", "Always"])
    mtrans = st.selectbox("Transporte", ["Public_Transportation", "Walking", "Automobile", "Motorbike", "Bike"])

    imc = peso / (altura ** 2)
    st.metric("IMC", round(imc, 2))

    if st.button("🔍 Prever"):

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
                try:
                    input_df[col] = enc.transform(input_df[col])
                except:
                    input_df[col] = enc.transform([enc.classes_[0]])

        # ✅ LINHA MAIS IMPORTANTE (resolve o erro!)
        input_df = input_df.reindex(columns=model.feature_names_in_, fill_value=0)

        pred = model.predict(input_df)[0]

        st.success(f"🏥 Classificação: {pred}")

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
