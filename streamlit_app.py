import streamlit as st
import pandas as pd
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
    [
        "Exploração",
        "Análise + Modelagem",
        "Dashboard",
        "Calculadora",
        "Recomendações"
    ]
)

# =========================
# LOAD DATA
# =========================
@st.cache_data
def load_data():
    df = pd.read_csv("Obesity.csv")

    df.columns = df.columns.str.strip()

    for col in df.columns:
        if "obese" in col.lower():
            df.rename(columns={col: "Obesity_level"}, inplace=True)

    df["IMC"] = df["Weight"] / (df["Height"] ** 2)

    return df

df = load_data()

# =========================
# LOAD MODEL
# =========================
model = pickle.load(open("model.pkl", "rb"))
encoders = pickle.load(open("encoders.pkl", "rb"))

# =========================================================
# EXPLORAÇÃO
# =========================================================
if page == "Exploração":

    st.title("📁 Exploração dos Dados")

    st.write(df.head())
    st.write(df.describe())

# =========================================================
# ANÁLISE
# =========================================================
elif page == "Análise + Modelagem":

    st.title("📊 Análise")

    fig, ax = plt.subplots()
    sns.scatterplot(data=df, x="Age", y="IMC", hue="Obesity_level", ax=ax)
    st.pyplot(fig)

    st.markdown("""
    - IMC principal driver  
    - Hábitos impactam diretamente  
    """)

# =========================================================
# DASHBOARD
# =========================================================
elif page == "Dashboard":

    st.title("📊 Dashboard Executivo")

    st.metric("Total", len(df))
    st.metric("IMC Médio", round(df["IMC"].mean(), 2))

    fig, ax = plt.subplots()
    df["Obesity_level"].value_counts().plot(kind="bar", ax=ax)
    st.pyplot(fig)

# =========================================================
# CALCULADORA COM MODELO
# =========================================================
elif page == "Calculadora":

    st.title("🧠 Predição com Machine Learning")

    idade = st.slider("Idade", 10, 80)
    altura = st.number_input("Altura", 1.4, 2.2, 1.7)
    peso = st.number_input("Peso", 40, 150, 70)

    genero = st.selectbox("Gênero", ["Male", "Female"])
    favc = st.selectbox("Calórico", ["yes", "no"])
    fcvc = st.slider("Vegetais", 1, 3, 2)
    faf = st.slider("Atividade física", 0, 3, 1)

    imc = peso / (altura ** 2)
    st.metric("IMC", round(imc, 2))

    if st.button("Prever"):

        input_dict = {
            "Gender": genero,
            "Age": idade,
            "FAVC": favc,
            "FCVC": fcvc,
            "FAF": faf,
            "IMC": imc
        }

        input_df = pd.DataFrame([input_dict])

        for col, enc in encoders.items():
            if col in input_df:
                input_df[col] = enc.transform(input_df[col])

        pred = model.predict(input_df)[0]

        st.success(f"Resultado: {pred}")

# =========================================================
# RECOMENDAÇÕES
# =========================================================
else:

    st.title("💡 Recomendações")

    st.markdown("""
    - Aumentar atividade física  
    - Melhorar alimentação  
    - Monitorar hábitos  
    """)
