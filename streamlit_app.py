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
# 📥 LOAD DATA
# =========================================================
@st.cache_data
def load_data():
    df = pd.read_csv("Obesity.csv")
    df["IMC"] = df["Weight"] / (df["Height"] ** 2)
    return df

df = load_data()

# =========================================================
# 🤖 LOAD MODEL
# =========================================================
model = pickle.load(open("model.pkl", "rb"))
encoders = pickle.load(open("encoders.pkl", "rb"))

# =========================================================
# ORDEM DAS CLASSES
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

    st.write(f"Registros: {df.shape[0]}")
    st.write(f"Variáveis: {df.shape[1]}")

    # Tipos + descrição
    tipos = df.dtypes.reset_index()
    tipos.columns = ["Variável", "Tipo técnico"]

    st.subheader("Tipos de dados")
    st.dataframe(tipos, use_container_width=True)

    # Estatísticas
    st.subheader("Estatísticas descritivas")
    stats = df.describe().T[["mean", "min", "max"]]
    stats.columns = ["Média", "Mínimo", "Máximo"]
    st.dataframe(stats)

# =========================================================
# 📊 ANÁLISE + MODELAGEM
# =========================================================
elif page == "📊 Análise + Modelagem":

    st.title("📊 Análise e Modelagem")

    # =========================
    # IDADE VS OBESIDADE
    # =========================
    st.subheader("Idade vs obesidade")

    fig, ax = plt.subplots(figsize=(10,5))
    sns.boxplot(
        data=df,
        x="Obesity",
        y="Age",
        order=order_original,
        palette="viridis"
    )

    ax.set_xticklabels(labels_pt, rotation=30)

    st.pyplot(fig)

    st.markdown("""
Observa-se aumento gradual da idade conforme os níveis de obesidade crescem.

### Outliers
Há presença de valores fora do padrão (outliers), representados pelos pontos acima dos limites das caixas.

Esses valores indicam indivíduos significativamente mais velhos dentro de determinadas classes.

➡️ Interpretação:
Os outliers refletem variações reais da população e indicam que a obesidade pode ocorrer também em faixas etárias mais avançadas.

### Conclusão
A idade contribui para o risco, mas atua em conjunto com hábitos e estilo de vida.
""")

    # =========================
    # MODELAGEM
    # =========================
    st.subheader("🧠 Metodologia de modelagem")

    st.markdown("""
### Preparação dos dados
- Criação da variável IMC  
- Remoção de Height e Weight (evitar vazamento de dados)  
- Encoding de variáveis categóricas  

### Modelo
Foi utilizado o algoritmo Random Forest.

### Justificativa
- Lida bem com dados numéricos e categóricos  
- Captura relações não lineares  
- É robusto a outliers  
- Reduz overfitting  

### Estratégia
- Divisão treino/teste (80/20)
- Avaliação com dados não vistos

### Resultado
Acurácia aproximada de 97%, indicando boa generalização.
""")

    # =========================
    # AVALIAÇÃO DO MODELO
    # =========================
    st.subheader("📊 Avaliação do modelo")

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

    st.metric("Acurácia do modelo", f"{acc:.2%}")

    # MATRIZ DE CONFUSÃO
    st.subheader("Matriz de confusão")

    fig_cm, ax_cm = plt.subplots()
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d", cmap="Blues")

    st.pyplot(fig_cm)

    st.markdown("""
A matriz indica alta taxa de acerto, com erros concentrados entre classes próximas.

Isso é esperado devido à continuidade das categorias de obesidade.
""")

    # =========================
    # FEATURE IMPORTANCE (🔥 NOVO)
    # =========================
    st.subheader("📊 Importância das variáveis")

    importances = model.feature_importances_

    importance_df = pd.DataFrame({
        "Variável": X.columns,
        "Importância": importances
    }).sort_values(by="Importância", ascending=False)

    # gráfico
    fig_imp, ax_imp = plt.subplots(figsize=(10,5))
    sns.barplot(
        data=importance_df,
        x="Importância",
        y="Variável",
        palette="viridis",
        ax=ax_imp
    )

    st.pyplot(fig_imp)

    st.markdown("""
### 📌 Interpretação

O gráfico mostra o peso de cada variável na tomada de decisão do modelo.

Observa-se que:

- O IMC é a variável mais relevante  
- Atividade física e alimentação também possuem forte impacto  
- Variáveis comportamentais complementam a previsão  

### 🎯 Conclusão

Os fatores mais importantes identificados pelo modelo estão alinhados com a análise exploratória,
o que reforça a consistência e confiabilidade do resultado.
""")

# =========================================================
# 🧠 CALCULADORA
# =========================================================
elif page == "🧠 Calculadora":

    st.title("🧠 Predição de Obesidade")

    age = st.slider("Idade", 10, 80)
    peso = st.number_input("Peso", 30.0, 200.0)
    altura = st.number_input("Altura", 1.40, 2.20)

    imc = peso / (altura ** 2)

    st.write(f"IMC: {imc:.2f}")

# =========================================================
# 💡 RECOMENDAÇÕES
# =========================================================
else:

    st.title("💡 Recomendações")

    st.markdown("""
- Praticar atividade física regularmente  
- Reduzir consumo de alimentos calóricos  
- Monitorar o IMC  
- Diminuir comportamento sedentário  

Essas recomendações são baseadas nos padrões identificados na análise dos dados.
""")
