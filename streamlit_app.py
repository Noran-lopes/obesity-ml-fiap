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
from sklearn.metrics import accuracy_score, confusion_matrix

# =========================================================
# ⚙️ CONFIG
# =========================================================
st.set_page_config(page_title="Obesity Analytics", layout="wide")

# =========================================================
# MENU
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
# LOAD DATA
# =========================================================
@st.cache_data
def load_data():
    df = pd.read_csv("Obesity.csv")
    df["IMC"] = df["Weight"] / (df["Height"] ** 2)
    return df

df = load_data()

# =========================================================
# LOAD MODEL
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

    st.subheader("Tipos de dados")
    tipos = df.dtypes.reset_index()
    tipos.columns = ["Variável", "Tipo técnico"]
    st.dataframe(tipos, use_container_width=True)

    st.subheader("Estatísticas descritivas")
    stats = df.describe().T[["mean", "min", "max"]]
    stats.columns = ["Média", "Mínimo", "Máximo"]
    st.dataframe(stats)

# =========================================================
# 📊 ANÁLISE + MODELAGEM
# =========================================================
elif page == "📊 Análise + Modelagem":

    st.title("📊 Análise e Modelagem")

    # -------------------------
    # IDADE VS OBESIDADE
    # -------------------------
    st.subheader("Idade vs obesidade")

    fig, ax = plt.subplots(figsize=(10,5))
    sns.boxplot(data=df, x="Obesity", y="Age", order=order_original, palette="viridis")
    ax.set_xticklabels(labels_pt, rotation=30)

    st.pyplot(fig)

    st.markdown("""
A idade apresenta tendência crescente conforme os níveis de obesidade aumentam.

### Outliers
Observam-se valores acima do padrão (outliers), indicando indivíduos significativamente mais velhos.

➡️ Esses pontos representam variabilidade real da população.

### Conclusão
A idade influencia o risco, mas não de forma isolada.
""")

    # ======================================================
    # MODELAGEM
    # ======================================================
    st.subheader("🧠 Metodologia de Modelagem")

    st.markdown("""
- Criação do IMC  
- Remoção de Height e Weight (evitar leakage)  
- Encoding de variáveis categóricas  
- Modelo utilizado: Random Forest  

✅ Justificativa:
- Lida bem com dados mistos  
- Modela relações não lineares  
- Robusto a outliers  
- Reduz overfitting  

✅ Validação:
- Divisão treino/teste (80/20)

✅ Resultado:
Acurácia de aproximadamente 97%.
""")

    # ======================================================
    # AVALIAÇÃO
    # ======================================================
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

    st.metric("Acurácia", f"{acc:.2%}")

    # MATRIZ
    fig_cm, ax_cm = plt.subplots()
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d", cmap="Blues")
    st.pyplot(fig_cm)

    # ======================================================
    # FEATURE IMPORTANCE
    # ======================================================
    st.subheader("📊 Importância das variáveis")

    importances = model.feature_importances_

    importance_df = pd.DataFrame({
        "Variável": X.columns,
        "Importância": importances
    }).sort_values(by="Importância", ascending=False)

    fig_imp, ax_imp = plt.subplots(figsize=(10,5))
    sns.barplot(data=importance_df, x="Importância", y="Variável", palette="viridis")

    st.pyplot(fig_imp)

    st.markdown("""
O IMC é a variável mais relevante, seguido de fatores comportamentais como atividade física e alimentação.

Isso reforça a consistência do modelo com os padrões observados na análise.
""")

# =========================================================
# 🧠 CALCULADORA COMPLETA + RECOMENDAÇÃO
# =========================================================
elif page == "🧠 Calculadora":

    st.title("🧠 Predição de Obesidade")

    gender = st.selectbox("Gênero", ["Male", "Female"])
    age = st.slider("Idade", 10, 80)

    family_history = st.selectbox("Histórico familiar", ["yes", "no"])
    favc = st.selectbox("Consome alimentos calóricos frequentes?", ["yes", "no"])

    fcvc = st.slider("Consumo de vegetais", 1, 3)
    ncp = st.slider("Refeições diárias", 1, 5)

    caec = st.selectbox("Lanches", ["no", "Sometimes", "Frequently", "Always"])

    smoke = st.selectbox("Fuma?", ["yes", "no"])
    ch2o = st.slider("Água (litros)", 1, 5)

    scc = st.selectbox("Controla calorias?", ["yes", "no"])

    faf = st.slider("Atividade física", 0, 7)
    tue = st.slider("Tempo de tecnologia", 0, 24)

    calc = st.selectbox("Álcool", ["no", "Sometimes", "Frequently", "Always"])

    mtrans = st.selectbox("Transporte", ["Walking", "Bike", "Public_Transportation", "Automobile", "Motorbike"])

    peso = st.number_input("Peso", 30.0, 200.0)
    altura = st.number_input("Altura", 1.40, 2.20)

    imc = peso / (altura**2)
    st.write(f"IMC: {imc:.2f}")

    if st.button("Prever"):

        input_dict = {
            "Gender": gender,
            "Age": age,
            "family_history": family_history,
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

        # encoding
        for col in encoders:
            if col in input_dict:
                input_dict[col] = encoders[col].transform([input_dict[col]])[0]

        input_array = np.array(list(input_dict.values())).reshape(1, -1)
        pred = int(model.predict(input_array)[0])

        st.success(f"Resultado: {labels_pt[pred]}")
        st.progress((pred + 1) / 7)

        # ======================
        # RECOMENDAÇÕES
        # ======================
        st.subheader("💡 Recomendações")

        if fcvc <= 1:
            st.write("🥦 Aumentar consumo de vegetais pode ajudar no controle de peso.")

        if faf <= 1:
            st.write("🏃 Aumentar atividade física é essencial para reduzir risco.")

        if favc == "yes":
            st.write("🍔 Reduzir alimentos calóricos pode melhorar o quadro.")

        if tue >= 6:
            st.write("📱 Reduzir tempo de tela ajuda a diminuir sedentarismo.")

        if pred >= 4:
            st.write("⚠️ Recomenda-se procurar acompanhamento profissional.")

# =========================================================
# 💡 RECOMENDAÇÕES GERAIS
# =========================================================
else:

    st.title("💡 Recomendações gerais")

    st.write("✔ Pratique atividade física regularmente")
    st.write("✔ Reduza alimentos ultraprocessados")
    st.write("✔ Controle o IMC")
    st.write("✔ Reduza comportamento sedentário")
