import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="School Fee Analyzer", layout="wide")
st.title("\U0001F4DA School Fee Analysis & Prediction App")

# --- 1. Load default data or user file ---
st.sidebar.header("Upload or Use Sample Data")

uploaded_file = st.sidebar.file_uploader("Choose CSV File", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.sidebar.success("\u2705 File Uploaded")
else:
    # Sample default data
    df = pd.DataFrame({
        "Annual Tuition Fee (NPR)": np.random.randint(20000, 100000, 100),
        "Monthly Fee (NPR)": np.random.randint(1500, 8000, 100),
        "Technology Access Index": np.random.uniform(0, 1, 100),
        "Infrastructure Score": np.random.uniform(1, 10, 100),
        "Scholarship % Availability": np.random.randint(0, 100, 100),
        "Fee Increase % (YoY)": np.random.uniform(1, 15, 100)
    })

# --- 2. Manual Entry ---
st.sidebar.header("\u2795 Add New Entry")

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
        st.sidebar.success("\u2705 Entry Added to Data")

# --- 3. Data Preview ---
st.subheader("\U0001F4CA Data Preview")
st.dataframe(df.head())

# --- 4. Model Training & Prediction ---
st.subheader("\U0001F4C8 Predict Annual Tuition Fee")

features = ['Monthly Fee (NPR)', 'Technology Access Index', 'Infrastructure Score',
            'Scholarship % Availability', 'Fee Increase % (YoY)']
target = 'Annual Tuition Fee (NPR)'

# Ensure no NaN in training
df.dropna(inplace=True)

X = df[features]
y = df[target]

try:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)

    st.markdown(f"**Model Accuracy (R² Score)**: `{score:.2f}`")

    st.markdown("### \U0001F52E Try Custom Prediction")

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
        st.success(f"Predicted Annual Tuition Fee: NPR {prediction[0]:,.2f}")

except Exception as e:
    st.error("\u274C Model training failed. Ensure enough valid data and no missing values.")
    st.exception(e)

# --- 5. Visualizations ---
st.subheader("\U0001F4C9 Data Visualizations")

tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "\U0001F4D8 Histogram",
    "\U0001F4D9 Boxplot",
    "\U0001F4D7 Heatmap",
    "\U0001F4D5 Infra vs Tuition",
    "\U0001F4D2 Violin Plot",
    "\U0001F4D4 Scholarship vs Fee Increase",
    "\U0001F4C8 Pairplot"
])

with tab1:
    st.markdown("#### Histogram of Annual Tuition Fee")
    fig1, ax1 = plt.subplots()
    sns.histplot(df['Annual Tuition Fee (NPR)'], bins=30, kde=True, ax=ax1)
    ax1.set_title("Distribution of Annual Tuition Fee")
    st.pyplot(fig1)

with tab2:
    st.markdown("#### Boxplot of Monthly Fee")
    fig2, ax2 = plt.subplots()
    sns.boxplot(x=df['Monthly Fee (NPR)'], ax=ax2)
    ax2.set_title("Monthly Fee Distribution")
    st.pyplot(fig2)

with tab3:
    st.markdown("#### Correlation Heatmap")
    fig3, ax3 = plt.subplots(figsize=(10, 6))
    sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm', ax=ax3)
    ax3.set_title("Feature Correlation")
    st.pyplot(fig3)

with tab4:
    st.markdown("#### Infrastructure Score vs Tuition Fee")
    fig4, ax4 = plt.subplots()
    sns.scatterplot(x='Infrastructure Score', y='Annual Tuition Fee (NPR)', data=df, ax=ax4)
    ax4.set_title("Infra Score vs Annual Tuition Fee")
    st.pyplot(fig4)

with tab5:
    st.markdown("#### Violin Plot of Technology Access Index")
    fig5, ax5 = plt.subplots()
    sns.violinplot(y=df['Technology Access Index'], ax=ax5)
    ax5.set_title("Distribution of Tech Access")
    st.pyplot(fig5)

with tab6:
    st.markdown("#### Fee Increase % by Scholarship Availability")
    fig6, ax6 = plt.subplots()
    sns.barplot(x='Scholarship % Availability', y='Fee Increase % (YoY)', data=df, ax=ax6)
    ax6.set_title("Scholarship vs Fee Increase")
    st.pyplot(fig6)

with tab7:
    st.markdown("#### Pairplot of Key Features (May take time)")
    pairplot_fig = sns.pairplot(df[features + [target]])
    st.pyplot(pairplot_fig.figure)

st.success("\u2705 App Ready")
