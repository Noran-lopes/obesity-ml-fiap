# =========================================================
# 📦 IMPORTAÇÕES
# =========================================================
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# =========================================================
# ⚙️ CONFIGURAÇÃO DA APLICAÇÃO
# =========================================================
st.set_page_config(
    page_title="Obesity Analytics App",
    layout="wide"
)

# =========================================================
# 📊 MENU DE NAVEGAÇÃO
# =========================================================
page = st.sidebar.radio(
    "📊 Navegação",
    [
        "📁 Apresentação dos Dados",
        "📊 Análise + Modelagem",
        "🧠 Calculadora",
        "💡 Recomendações"
    ]
)

# =========================================================
# 📥 CARREGAMENTO DE DADOS (COM CACHE)
# =========================================================
@st.cache_data
def load_data():
    df = pd.read_csv("Obesity.csv")
    
    # Feature Engineering: criação do IMC
    df["IMC"] = df["Weight"] / (df["Height"] ** 2)
    
    return df

df = load_data()

# =========================================================
# 🤖 CARREGAMENTO DO MODELO
# =========================================================
model = pickle.load(open("model.pkl", "rb"))
encoders = pickle.load(open("encoders.pkl", "rb"))

# =========================================================
# 📌 ORDEM CORRETA DAS CLASSES
# =========================================================
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
# 📁 1. APRESENTAÇÃO DOS DADOS
# =========================================================
if page == "📁 Apresentação dos Dados":

    st.title("📁 Entendimento do Dataset")

    st.write(f"🔢 Registros: {df.shape[0]}")
    st.write(f"📊 Variáveis: {df.shape[1]}")

    # ---------------------------
    # TIPOS + DESCRIÇÃO
    # ---------------------------
    st.subheader("📊 Tipos de dados e descrição")

    tipos = df.dtypes.reset_index()
    tipos.columns = ["Variável", "Tipo técnico"]

    descricao = {
        "Gender": "Sexo do indivíduo",
        "Age": "Idade em anos",
        "Height": "Altura (m)",
        "Weight": "Peso (kg)",
        "family_history": "Histórico familiar de obesidade",
        "FAVC": "Consumo de alimentos calóricos",
        "FCVC": "Consumo de vegetais",
        "NCP": "Número de refeições por dia",
        "CAEC": "Frequência de lanches",
        "SMOKE": "Indica se fuma",
        "CH2O": "Consumo de água (litros)",
        "SCC": "Controle de calorias",
        "FAF": "Atividade física",
        "TUE": "Tempo de uso de tecnologia",
        "CALC": "Consumo de álcool",
        "MTRANS": "Transporte",
        "Obesity": "Classificação da obesidade",
        "IMC": "Índice de Massa Corporal"
    }

    tipos["Descrição"] = tipos["Variável"].map(descricao)

    st.dataframe(tipos, use_container_width=True)

    # ---------------------------
    # ESTATÍSTICA DESCRITIVA
    # ---------------------------
    st.subheader("📊 Estatísticas descritivas")

    stats = df.describe().T[["mean", "min", "max"]]
    stats.columns = ["Média", "Mínimo", "Máximo"]

    st.dataframe(stats)

    st.info("""
A análise descritiva permite compreender a distribuição e amplitude dos dados, 
sendo fundamental para orientar a modelagem e interpretação.
""")

# =========================================================
# 📊 2. ANÁLISE + MODELAGEM
# =========================================================
elif page == "📊 Análise + Modelagem":

    st.title("📊 Análise e Modelagem")

    # ---------------------------
    # DISTRIBUIÇÃO
    # ---------------------------
    st.subheader("Distribuição dos níveis de obesidade")

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

    # Colocar valores nas barras
    for p in ax.patches:
        ax.annotate(
            f"{int(p.get_height())}",
            (p.get_x()+p.get_width()/2., p.get_height()),
            ha="center"
        )

    st.pyplot(fig)

    st.markdown("""
A distribuição apresenta equilíbrio entre as classes, com maior concentração em obesidade tipo I.  
Isso reduz o risco de viés no modelo e contribui para melhor generalização.
""")

    # ---------------------------
    # IMC
    # ---------------------------
    st.subheader("IMC vs obesidade")

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
Observa-se progressão clara do IMC conforme o nível de obesidade aumenta.  
Esse comportamento confirma o IMC como principal variável explicativa.
""")

    # ---------------------------
    # IDADE
    # ---------------------------
    st.subheader("Idade vs obesidade")

    fig3, ax3 = plt.subplots()

    sns.boxplot(data=df, x="Obesity", y="Age", order=order_original)

    ax3.set_xticklabels(labels_pt, rotation=30)

    st.pyplot(fig3)

    st.markdown("""
Há tendência de aumento da idade em níveis mais altos de obesidade, 
indicando influência temporal e comportamental no risco.
""")

    # ---------------------------
    # MODELAGEM
    # ---------------------------
    st.subheader("Metodologia de modelagem")

    st.markdown("""
- Criação do IMC (feature engineering)  
- Remoção de Height e Weight (evitar vazamento de dados)  
- Codificação de variáveis categóricas  
- Modelo: Random Forest (melhor para dados não lineares)  
""")

    # ---------------------------
    # AVALIAÇÃO
    # ---------------------------
    st.subheader("Avaliação do modelo")

    df_model = df.drop(["Weight", "Height"], axis=1)

    for col in df_model.select_dtypes(include="object").columns:
        df_model[col] = encoders[col].transform(df_model[col])

    X = df_model.drop("Obesity", axis=1)
    y = df_model["Obesity"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)

    st.metric("Acurácia", f"{acc:.2%}")

    # ---------------------------
    # MATRIZ DE CONFUSÃO
    # ---------------------------
    fig_cm, ax_cm = plt.subplots()

    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d", cmap="Blues")

    st.pyplot(fig_cm)

    # ---------------------------
    # IMPORTÂNCIA
    # ---------------------------
    st.subheader("Importância das variáveis")

    importance_df = pd.DataFrame({
        "Variável": X.columns,
        "Importância": model.feature_importances_
    }).sort_values(by="Importância", ascending=False)

    st.bar_chart(importance_df.set_index("Variável"))

# =========================================================
# 🧠 3. CALCULADORA
# =========================================================
elif page == "🧠 Calculadora":

    st.title("🧠 Predição de Obesidade")

    gender = st.selectbox("Gênero", ["Male", "Female"])
    age = st.slider("Idade", 10, 80)

    faf = st.slider("Atividade física", 0, 7)

    peso = st.number_input("Peso", 30.0, 200.0)
    altura = st.number_input("Altura", 1.40, 2.20)

    imc = peso / (altura ** 2)
    st.write(f"IMC: {imc:.2f}")

# =========================================================
# 💡 4. RECOMENDAÇÕES
# =========================================================
else:

    st.title("💡 Recomendações")

    st.markdown("""
- Praticar atividade física regularmente  
- Reduzir alimentos ultraprocessados  
- Monitorar o IMC  
- Reduzir tempo sedentário  

Essas recomendações são consistentes com padrões observados nos dados.
""")
