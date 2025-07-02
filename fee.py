import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

st.set_page_config(page_title="School Fee Analyzer", layout="wide")
st.title("📚 School Fee Analysis & Prediction App")

# --- 1. Load default data or user file ---
st.sidebar.header("Upload or Use Sample Data")
uploaded_file = st.sidebar.file_uploader("Choose CSV File", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.sidebar.success("✅ File Uploaded")
else:
    df = pd.DataFrame({
        "Annual Tuition Fee (NPR)": np.random.randint(20000, 100000, 100),
        "Monthly Fee (NPR)": np.random.randint(1500, 8000, 100),
        "Technology Access Index": np.random.uniform(0, 1, 100),
        "Infrastructure Score": np.random.uniform(1, 10, 100),
        "Scholarship % Availability": np.random.randint(0, 100, 100),
        "Fee Increase % (YoY)": np.random.uniform(1, 15, 100)
    })

# --- 2. Manual Entry ---
st.sidebar.header("➕ Add New Entry")
with st.sidebar.form("manual_form"):
    annual_fee = st.number_input("Annual Tuition Fee (NPR)", 1000, 200000, 30000)
    monthly_fee = st.number_input("Monthly Fee (NPR)", 500, 15000, 2500)
    tech_index = st.slider("Technology Access Index", 0.0, 1.0, 0.5)
    infra_score = st.slider("Infrastructure Score", 0.0, 10.0, 5.0)
    scholarship_pct = st.slider("Scholarship % Availability", 0, 100, 20)
    fee_increase = st.slider("Fee Increase % (YoY)", 0.0, 25.0, 5.0)

    submitted = st.form_submit_button("Add to Dataset")
    if submitted:
        new_row = {
            "Annual Tuition Fee (NPR)": annual_fee,
            "Monthly Fee (NPR)": monthly_fee,
            "Technology Access Index": tech_index,
            "Infrastructure Score": infra_score,
            "Scholarship % Availability": scholarship_pct,
            "Fee Increase % (YoY)": fee_increase
        }
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
        st.sidebar.success("✅ Entry Added to Data")

# --- 3. Data Preview ---
st.subheader("📊 Data Preview")
st.dataframe(df.head())

# --- 4. Model Training & Prediction ---
st.subheader("📈 Predict Annual Tuition Fee")

features = ['Monthly Fee (NPR)', 'Technology Access Index', 'Infrastructure Score',
            'Scholarship % Availability', 'Fee Increase % (YoY)']
target = 'Annual Tuition Fee (NPR)'

df.dropna(inplace=True)
X = df[features]
y = df[target]

try:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)

    st.markdown(f"**Model Accuracy (R² Score)**: `{score:.2f}`")
    st.markdown(f"**RMSE**: `{np.sqrt(mean_squared_error(y_test, model.predict(X_test))):.2f}`")

    st.markdown("### 🔮 Try Custom Prediction")
    col1, col2 = st.columns(2)
    with col1:
        input_monthly = st.number_input("Monthly Fee", 1000, 20000, 3000)
        input_tech = st.slider("Technology Access Index", 0.0, 1.0, 0.6)
        input_infra = st.slider("Infrastructure Score", 0.0, 10.0, 6.0)
    with col2:
        input_scholarship = st.slider("Scholarship %", 0, 100, 10)
        input_fee_increase = st.slider("Fee Increase %", 0.0, 20.0, 4.0)

    if st.button("Predict Fee"):
        prediction = model.predict([[input_monthly, input_tech, input_infra, input_scholarship, input_fee_increase]])
        st.success(f"🎯 Predicted Annual Tuition Fee: NPR {prediction[0]:,.2f}")

except Exception as e:
    st.error("❌ Model training failed. Ensure enough valid data and no missing values.")
    st.exception(e)

# --- 5. Visualizations ---
st.subheader("📉 Data Visualizations")

# Histogram
st.markdown("#### 📘 Histogram of Annual Tuition Fee")
fig1, ax1 = plt.subplots()
sns.histplot(df['Annual Tuition Fee (NPR)'], bins=30, kde=True, ax=ax1)
ax1.set_title("Distribution of Annual Tuition Fee")
st.pyplot(fig1)

# Boxplot
st.markdown("#### 📙 Boxplot of Monthly Fee")
fig2, ax2 = plt.subplots()
sns.boxplot(x=df['Monthly Fee (NPR)'], ax=ax2)
ax2.set_title("Monthly Fee Distribution")
st.pyplot(fig2)

# Correlation Heatmap
st.markdown("#### 📗 Correlation Heatmap")
fig3, ax3 = plt.subplots(figsize=(10, 6))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm', ax=ax3)
st.pyplot(fig3)

# Scatterplot
st.markdown("#### 📕 Infra Score vs Annual Tuition Fee")
fig4, ax4 = plt.subplots()
sns.scatterplot(x='Infrastructure Score', y='Annual Tuition Fee (NPR)', data=df, ax=ax4)
st.pyplot(fig4)

# Violin Plot
st.markdown("#### 📒 Violin Plot: Tech Access")
fig5, ax5 = plt.subplots()
sns.violinplot(y=df['Technology Access Index'], ax=ax5)
st.pyplot(fig5)

# Barplot
st.markdown("#### 📔 Fee Increase by Scholarship %")
fig6, ax6 = plt.subplots()
sns.barplot(x='Scholarship % Availability', y='Fee Increase % (YoY)', data=df, ax=ax6)
st.pyplot(fig6)

# Pairplot
st.markdown("#### 📈 Pairplot of Features")
pairplot_fig = sns.pairplot(df[features + [target]])
st.pyplot(pairplot_fig.figure)

st.success("✅ App Ready")
