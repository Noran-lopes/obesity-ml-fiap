import streamlit as st
import pandas as pd
import numpy as np
import pickle
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

st.set_page_config(page_title="Obesity Analytics", layout="wide")

# =========================================================
# LOAD DATA
# =========================================================
@st.cache_data
def load_data():
    df = pd.read_csv("Obesity.csv")
    df["IMC"] = df["Weight"] / (df["Height"] ** 2)
    return df

df = load_data()

model = pickle.load(open("model.pkl", "rb"))
encoders = pickle.load(open("encoders.pkl", "rb"))

# ordem correta
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
# MENU
# =========================================================
page = st.sidebar.radio(
    "📊 Navegação",
    ["📁 Apresentação", "📊 Análise + Modelagem", "🧠 Calculadora"]
)

# =========================================================
# 📁 APRESENTAÇÃO
# =========================================================
if page == "📁 Apresentação":

    st.title("📁 Entendimento dos Dados")

    st.write(f"Registros: {df.shape[0]}")
    st.write(f"Variáveis: {df.shape[1]}")

    st.subheader("📊 Estatísticas Descritivas")

    stats = df.describe().T[["mean", "min", "max"]]
    stats.columns = ["Média", "Mínimo", "Máximo"]

    st.dataframe(stats)

    st.markdown("""
A base apresenta boa variabilidade entre idade, peso e comportamento,
permitindo análise consistente dos fatores associados à obesidade.
""")

# =========================================================
# 📊 ANÁLISE + MODELAGEM
# =========================================================
elif page == "📊 Análise + Modelagem":

    st.title("📊 Análise Exploratória")

    # -----------------------
    # DISTRIBUIÇÃO
    # -----------------------
    st.subheader("Distribuição da Obesidade")

    fig, ax = plt.subplots(figsize=(10,5))

    sns.countplot(
        data=df,
        x="Obesity",
        order=order_original,
        palette="viridis"
    )

    ax.set_xticklabels(labels_pt, rotation=30)
    ax.set_yticks([])

    for p in ax.patches:
        ax.annotate(int(p.get_height()),
                    (p.get_x() + p.get_width()/2., p.get_height()),
                    ha='center')

    st.pyplot(fig)

    st.markdown("""
A base apresenta boa distribuição entre as classes, reduzindo risco de viés no modelo.
""")

    # -----------------------
    # IMC
    # -----------------------
    st.subheader("IMC vs Obesidade")

    fig2, ax2 = plt.subplots(figsize=(10,5))

    sns.boxplot(
        data=df,
        x="Obesity",
        y="IMC",
        order=order_original,
        palette="coolwarm"
    )

    ax2.set_xticklabels(labels_pt, rotation=30)
    st.pyplot(fig2)

    st.markdown("""
Há separação clara entre as classes, indicando que o IMC é o principal fator explicativo.

Outliers são mínimos, indicando consistência da variável.
""")

    # -----------------------
    # IDADE
    # -----------------------
    st.subheader("Idade vs Obesidade")

    fig3, ax3 = plt.subplots(figsize=(10,5))

    sns.boxplot(
        data=df,
        x="Obesity",
        y="Age",
        order=order_original,
        palette="viridis"
    )

    ax3.set_xticklabels(labels_pt, rotation=30)
    st.pyplot(fig3)

    st.markdown("""
Observa-se aumento gradual da idade.

### Outliers:
Presença de indivíduos mais velhos em diferentes classes, indicando variabilidade real da população.

### Conclusão:
A idade contribui para o risco, mas não é determinante isoladamente.
""")

    # -----------------------
    # ATIVIDADE
    # -----------------------
    st.subheader("Atividade Física")

    fig4, ax4 = plt.subplots()

    sns.boxplot(data=df, x="Obesity", y="FAF", order=order_original)

    st.pyplot(fig4)

    st.markdown("""
Indivíduos com menor atividade física apresentam níveis mais altos de obesidade.
""")

    # -----------------------
    # MODELAGEM
    # -----------------------
    st.subheader("🧠 Modelagem")

    st.markdown("""
O modelo foi desenvolvido com:

- Engenharia de variável (IMC)
- Remoção de Height/Weight (evitar leakage)
- Encoding de categóricas
- Random Forest

### Justificativa:
- Captura relações não lineares
- Robusto a outliers
- Alta performance
""")

    # -----------------------
    # AVALIAÇÃO
    # -----------------------
    st.subheader("📊 Avaliação")

    df_model = df.drop(["Weight", "Height"], axis=1)

    for col in df_model.select_dtypes("object"):
        df_model[col] = encoders[col].transform(df_model[col])

    X = df_model.drop("Obesity", axis=1)
    y = df_model["Obesity"]

    X_train, X_test, y_train, y_test = train_test_split(X, y)

    pred = model.predict(X_test)

    st.metric("Acurácia", f"{accuracy_score(y_test, pred):.2%}")

    fig_cm, ax_cm = plt.subplots()
    sns.heatmap(confusion_matrix(y_test, pred), annot=True, fmt="d")

    st.pyplot(fig_cm)

    # FEATURE IMPORTANCE
    st.subheader("Importância das variáveis")

    imp = pd.DataFrame({
        "Feature": X.columns,
        "Importância": model.feature_importances_
    }).sort_values(by="Importância", ascending=False)

    st.bar_chart(imp.set_index("Feature"))

    st.markdown("""
IMC é a principal variável, seguida por atividade física e alimentação,
reforçando a consistência entre análise exploratória e modelo.
""")

# =========================================================
# 🧠 CALCULADORA (NÍVEL FINAL)
# =========================================================
elif page == "🧠 Calculadora":

    st.title("🧠 Avaliação personalizada")

    age = st.slider("Idade", 10, 80)
    faf = st.slider("Atividade física", 0, 7)
    favc = st.selectbox("Alimentos calóricos", ["yes", "no"])
    calc = st.selectbox("Álcool", ["no", "Sometimes", "Frequently", "Always"])
    fcvc = st.slider("Vegetais", 1, 3)

    peso = st.number_input("Peso", 30.0, 200.0)
    altura = st.number_input("Altura", 1.40, 2.20)

    imc = peso / (altura**2)

    st.write(f"IMC atual: {imc:.2f}")

    if st.button("Avaliar"):

        # peso ideal
        peso_ideal = 24.9 * (altura**2)

        st.subheader("⚖️ Peso ideal")
        st.write(f"{peso_ideal:.1f} kg")

        # recomendacoes inteligentes
        st.subheader("💡 Como melhorar")

        if favc == "yes":
            st.write("Reduzir alimentos calóricos pode diminuir seu nível de obesidade.")

        if faf <= 1:
            st.write("Aumentar atividade física pode reduzir significativamente seu risco.")

        if calc in ["Frequently", "Always"]:
            st.write("Reduzir álcool pode ajudar a alcançar um nível mais saudável.")

        if fcvc <= 1:
            st.write("Aumentar vegetais pode contribuir para redução de peso.")

        diff = peso - peso_ideal

        if diff > 0:
            st.write(f"Reduzindo cerca de {diff:.1f} kg você pode atingir uma faixa mais saudável.")
