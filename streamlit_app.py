# =========================================================
# 📦 IMPORTS
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
# ⚙️ CONFIGURAÇÃO
# =========================================================
st.set_page_config(page_title="Obesity Analytics", layout="wide")

# =========================================================
# 📊 MENU
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
# 📥 LOAD DATA (CACHEADO PARA PERFORMANCE)
# =========================================================
@st.cache_data
def load_data():
    df = pd.read_csv("Obesity.csv")

    # Feature Engineering: criação do IMC
    df["IMC"] = df["Weight"] / (df["Height"] ** 2)

    return df

df = load_data()

# =========================================================
# 🤖 LOAD MODEL
# =========================================================
model = pickle.load(open("model.pkl", "rb"))
encoders = pickle.load(open("encoders.pkl", "rb"))

# =========================================================
# 🔢 ORDEM DAS CLASSES (IMPORTANTE)
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
# 📁 APRESENTAÇÃO DOS DADOS
# =========================================================
if page == "📁 Apresentação dos Dados":

    st.title("📁 Entendimento do Dataset")

    st.write(f"🔢 Registros: {df.shape[0]}")
    st.write(f"📊 Variáveis: {df.shape[1]}")

    # Tipos + descrição
    st.subheader("📊 Tipos de dados")

    tipos = df.dtypes.reset_index()
    tipos.columns = ["Variável", "Tipo técnico"]

    descricao = {
        "Age": "Idade (anos)",
        "Height": "Altura (m)",
        "Weight": "Peso (kg)",
        "FAF": "Atividade física",
        "TUE": "Tempo de tecnologia",
        "IMC": "Índice de Massa Corporal"
    }

    tipos["Descrição"] = tipos["Variável"].map(descricao)

    st.dataframe(tipos, use_container_width=True)

    # Estatística descritiva
    st.subheader("📊 Estatísticas (Média, Mínimo e Máximo)")
    stats = df.describe().T[["mean", "min", "max"]]
    stats.columns = ["Média", "Mínimo", "Máximo"]

    st.dataframe(stats)

# =========================================================
# 📊 ANÁLISE + MODELAGEM
# =========================================================
elif page == "📊 Análise + Modelagem":

    st.title("📊 Análise e Modelagem")

    # ---------------------------
    # DISTRIBUIÇÃO
    # ---------------------------
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
A distribuição apresenta equilíbrio entre os níveis, com maior concentração em obesidade tipo I.
Isso contribui para um modelo mais estável e menos enviesado.
""")

    # ---------------------------
    # IDADE VS OBESIDADE
    # ---------------------------
    st.subheader("Idade vs obesidade")

    fig2, ax2 = plt.subplots(figsize=(10,5))
    sns.boxplot(data=df, x="Obesity", y="Age", order=order_original, palette="viridis")

    ax2.set_xticklabels(labels_pt, rotation=30)

    st.pyplot(fig2)

    st.markdown("""
Observa-se aumento gradual da idade ao longo dos níveis de obesidade.

### Outliers
Há presença de valores extremos (pontos fora da caixa), indicando indivíduos significativamente mais velhos dentro de determinadas classes.

Esses outliers representam variações reais da população e indicam que a obesidade pode ocorrer também em faixas etárias mais elevadas.

### Conclusão
A idade influencia o risco, mas não é fator isolado — atuando em conjunto com estilo de vida e hábitos.
""")

    # ---------------------------
    # IMC
    # ---------------------------
    st.subheader("IMC vs obesidade")

    fig3, ax3 = plt.subplots(figsize=(10,5))
    sns.boxplot(data=df, x="Obesity", y="IMC", order=order_original, palette="coolwarm")

    ax3.set_xticklabels(labels_pt, rotation=30)

    st.pyplot(fig3)

    st.markdown("""
O IMC cresce de forma consistente com o nível de obesidade,
confirmando seu papel como principal variável explicativa.
""")

    # ======================================================
    # 🤖 MODELAGEM
    # ======================================================
    st.subheader("🧠 Metodologia de Modelagem")

    st.markdown("""
### Preparação dos dados
- Criação do IMC  
- Remoção de altura e peso para evitar vazamento de dados  
- Codificação de variáveis categóricas  

### Modelo
Foi utilizado o algoritmo Random Forest.

### Justificativa
O Random Forest foi escolhido pois:
- Lida bem com dados mistos  
- Captura relações não lineares  
- É robusto a outliers  
- Reduz overfitting por meio de múltiplas árvores  

### Estratégia
- Divisão treino/teste (80/20)
- Avaliação fora da base de treino

### Resultado
Acurácia aproximada de 97%, indicando boa capacidade de generalização.
""")

    # ======================================================
    # 📊 AVALIAÇÃO
    # ======================================================
    st.subheader("📊 Avaliação do modelo")

    df_model = df.drop(["Weight", "Height"], axis=1)

    for col in df_model.select_dtypes(include="object").columns:
        df_model[col] = encoders[col].transform(df_model[col])

    X = df_model.drop("Obesity", axis=1)
    y = df_model["Obesity"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)

    st.metric("Acurácia", f"{acc:.2%}")

    # matriz
    fig_cm, ax_cm = plt.subplots()
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d", cmap="Blues")
    st.pyplot(fig_cm)

    st.markdown("""
O modelo apresenta alto nível de acerto, com erros concentrados entre categorias próximas,
o que é esperado para variáveis contínuas como o IMC.
""")

# =========================================================
# 🧠 CALCULADORA
# =========================================================
elif page == "🧠 Calculadora":

    st.title("🧠 Predição de Obesidade")

    age = st.slider("Idade", 10, 80)
    faf = st.slider("Atividade física (dias)", 0, 7)

    peso = st.number_input("Peso", 30.0, 200.0)
    altura = st.number_input("Altura", 1.40, 2.20)

    imc = peso / (altura**2)

    st.write(f"IMC: {imc:.2f}")

    st.info("A predição considera hábitos e características físicas.")

# =========================================================
# 💡 RECOMENDAÇÕES
# =========================================================
else:

    st.title("💡 Recomendações")

    st.markdown("""
- A prática regular de atividade física reduz o risco de obesidade  
- A alimentação equilibrada é essencial para controle do peso  
- Redução do sedentarismo (tempo de tela) é importante  
- Monitorar o IMC permite identificar riscos precocemente  

As recomendações são baseadas nos padrões observados na análise dos dados.
""")
