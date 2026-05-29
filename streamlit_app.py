import streamlit as st
import pandas as pd
import numpy as np
import pickle
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

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
        "📊 Análise + Modelagem",
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

# ORDEM CORRETA
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

    st.write(f"Registros: {df.shape[0]}")
    st.write(f"Variáveis: {df.shape[1]}")

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
        "CH2O": "Água (litros)",
        "SCC": "Controle de calorias",
        "FAF": "Atividade física",
        "TUE": "Tempo de tela",
        "CALC": "Álcool",
        "MTRANS": "Transporte",
        "Obesity": "Nível de obesidade"
    }

    tipos["Descrição"] = tipos["Variável"].map(descricao)

    st.dataframe(tipos, use_container_width=True)

    st.info("O dataset combina dados físicos e comportamentais, permitindo análise abrangente do risco de obesidade.")

# =========================================================
# 📊 ANÁLISE + MODELO
# =========================================================
elif page == "📊 Análise + Modelagem":

    st.title("📊 Análise e Modelagem")

    # DISTRIBUIÇÃO
    st.subheader("Distribuição dos níveis de obesidade")

    fig, ax = plt.subplots(figsize=(10,5))
    sns.countplot(data=df, x="Obesity", order=order_original, palette="viridis", ax=ax)

    ax.set_xticklabels(labels_pt, rotation=30)
    ax.set_ylabel("")
    ax.set_yticks([])

    for p in ax.patches:
        ax.annotate(f"{int(p.get_height())}", (p.get_x()+p.get_width()/2., p.get_height()), ha="center")

    st.pyplot(fig)

    st.markdown("""
A distribuição mostra boa representatividade entre as classes, com maior concentração em obesidade tipo I e menor em abaixo do peso.  
Isso contribui para treinamento equilibrado e reduz viés no modelo.
""")

    # IMC
    st.subheader("IMC vs obesidade")

    fig2, ax2 = plt.subplots(figsize=(10,5))
    sns.boxplot(data=df, x="Obesity", y="IMC", order=order_original, palette="coolwarm", ax=ax2)

    ax2.set_xticklabels(labels_pt, rotation=30)
    st.pyplot(fig2)

    st.markdown("""
Observa-se progressão clara do IMC entre as categorias, indicando que o indicador é consistente e discriminatório.  
Esse comportamento valida o IMC como principal variável explicativa.
""")

    # PIPELINE
    st.subheader("Pipeline de modelagem")

    st.markdown("""
- Criação do IMC  
- Remoção de peso e altura (evitar vazamento)  
- Encoding de variáveis categóricas  
- Treinamento com Random Forest  

O modelo foi projetado para capturar relações não lineares presentes nos dados.
""")

    # =================================================
    # AVALIAÇÃO
    # =================================================
    st.subheader("Avaliação do modelo")

    df_model = df.drop(["Weight", "Height"], axis=1)

    for col in df_model.select_dtypes(include="object").columns:
        df_model[col] = encoders[col].transform(df_model[col])

    X = df_model.drop("Obesity", axis=1)
    y = df_model["Obesity"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)

    st.metric("Acurácia", f"{acc:.2%}")

    st.markdown("""
O modelo foi avaliado em dados não utilizados no treinamento.  
A acurácia indica alta capacidade de generalização, sem evidências de overfitting.
""")

    # MATRIZ
    fig_cm, ax_cm = plt.subplots()
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d", cmap="Blues")
    st.pyplot(fig_cm)

    st.markdown("""
A matriz de confusão mostra predominância de acertos na diagonal principal.  
Erros ocorrem principalmente entre categorias próximas.
""")

    # IMPORTÂNCIA
    st.subheader("Importância das variáveis")

    importances = model.feature_importances_

    importance_df = pd.DataFrame({
        "Variável": X.columns,
        "Importância": importances
    }).sort_values(by="Importância", ascending=False)

    st.bar_chart(importance_df.set_index("Variável"))

    st.markdown("""
Variáveis como IMC, atividade física e alimentação destacam-se como determinantes,  
alinhando o modelo com o entendimento do problema.
""")

    # REPORT
    st.subheader("Métricas por classe")

    report = classification_report(y_test, y_pred, output_dict=True)
    st.dataframe(pd.DataFrame(report).transpose())

# =========================================================
# 🧠 CALCULADORA
# =========================================================
elif page == "🧠 Calculadora":

    st.title("Calculadora de Obesidade")

    gender = st.selectbox("Gênero", ["Male","Female"])
    age = st.slider("Idade",10,80)

    family_history = st.selectbox("Histórico",["yes","no"])
    favc = st.selectbox("Calorias frequentes?",["yes","no"])

    fcvc = st.slider("Vegetais",1,3)
    ncp = st.slider("Refeições",1,5)

    caec = st.selectbox("Lanches",["no","Sometimes","Frequently","Always"])
    smoke = st.selectbox("Fuma?",["yes","no"])

    ch2o = st.slider("Água",1,5)
    scc = st.selectbox("Controle calórico",["yes","no"])

    faf = st.slider("Atividade física",0,7)
    tue = st.slider("Tecnologia",0,24)

    calc = st.selectbox("Álcool",["no","Sometimes","Frequently","Always"])
    mtrans = st.selectbox("Transporte",["Walking","Bike","Public_Transportation","Automobile","Motorbike"])

    peso = st.number_input("Peso",30.0,200.0)
    altura = st.number_input("Altura",1.40,2.20)

    imc = peso/(altura**2)

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

        labels = labels_pt

        st.success(f"Resultado: {labels[pred]}")
        st.progress((pred+1)/7)

# =========================================================
# 💡 RECOMENDAÇÕES
# =========================================================
else:

    st.title("Recomendações")

    st.write("Manter atividade física regular contribui para redução de risco.")
    st.write("Alimentação equilibrada é fator determinante.")
    st.write("Casos avançados requerem acompanhamento profissional.")
