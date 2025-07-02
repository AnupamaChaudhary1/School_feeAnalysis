import pandas as pd
import numpy as np
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import mean_absolute_error, mean_squared_error, accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

# Load data
@st.cache_data
def load_data():
    return pd.read_csv("cleaned_data.csv")

df = load_data()

st.title("Private School Fee Analysis in Nepal")

# Sidebar tabs
option = st.sidebar.radio("Select Analysis", [
    "Linear Regression (Predict Fee)",
    "Logistic Regression (Scholarship)",
    "KMeans Clustering (Segments)",
    "Random Forest/XGBoost (Academics)",
    "Decision Tree (Fee Increase)"
])

# 1. Linear Regression
if option == "Linear Regression (Predict Fee)":
    st.header("1. Predict Tuition Fee (Linear Regression)")
    X = df[['Infrastructure Score', 'Technology Access Index', 'Average Academic Score (%)']]
    y = df['Annual Tuition Fee (NPR)']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    st.markdown(f"**MAE:** {mean_absolute_error(y_test, y_pred):.2f}")
    st.markdown(f"**MSE:** {mean_squared_error(y_test, y_pred):.2f}")
    st.markdown(f"**RMSE:** {np.sqrt(mean_squared_error(y_test, y_pred)):.2f}")

    st.subheader("Actual vs Predicted Tuition Fee")
    fig1, ax1 = plt.subplots()
    ax1.scatter(y_test, y_pred, alpha=0.6)
    ax1.set_xlabel("Actual")
    ax1.set_ylabel("Predicted")
    st.pyplot(fig1)

# 2. Logistic Regression
elif option == "Logistic Regression (Scholarship)":
    st.header("2. Classify High Scholarship (Logistic Regression)")
    df['High Scholarship'] = (df['Scholarship % Availability'] > df['Scholarship % Availability'].median()).astype(int)
    features = ['Infrastructure Score', 'Monthly Fee (NPR)', 'Admission Fee (NPR)']
    X = df[features]
    y = df['High Scholarship']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LogisticRegression(max_iter=500)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    st.markdown(f"**Accuracy:** {acc:.2f}")

    coef_df = pd.DataFrame({
        'Feature': features,
        'Coefficient': model.coef_[0]
    }).sort_values(by='Coefficient')

    fig2, ax2 = plt.subplots()
    ax2.barh(coef_df['Feature'], coef_df['Coefficient'], color='green')
    ax2.set_title('Feature Influence on High Scholarship')
    st.pyplot(fig2)

# 3. KMeans Clustering
elif option == "KMeans Clustering (Segments)":
    st.header("3. Cluster Schools (KMeans)")
    X = df[['Annual Tuition Fee (NPR)', 'Technology Access Index', 'Infrastructure Score']]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_scaled)
    df['Cluster'] = clusters

    centers = scaler.inverse_transform(kmeans.cluster_centers_)
    mean_fees = pd.Series(centers[:, 0])
    cluster_names = {i: name for i, name in zip(mean_fees.sort_values().index, ['Budget', 'Mid-tier', 'Premium'])}
    df['Segment'] = df['Cluster'].map(cluster_names)

    fig3, ax3 = plt.subplots()
    sns.scatterplot(data=df, x='Annual Tuition Fee (NPR)', y='Infrastructure Score', hue='Segment', ax=ax3, palette='Set2')
    st.pyplot(fig3)

# 4. Academic Classification
elif option == "Random Forest/XGBoost (Academics)":
    st.header("4. Predict High Academic Performance")
    df['High Academic'] = (df['Average Academic Score (%)'] > df['Average Academic Score (%)'].mean()).astype(int)
    features = ['Technology Access Index', 'Infrastructure Score', 'Monthly Fee (NPR)', 'Student-Teacher Ratio']
    X = df[features]
    y = df['High Academic']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    rf = RandomForestClassifier(random_state=42)
    rf.fit(X_train, y_train)
    xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    xgb.fit(X_train, y_train)

    y_pred_rf = rf.predict(X_test)
    y_pred_xgb = xgb.predict(X_test)

    st.markdown(f"**Random Forest Accuracy:** {accuracy_score(y_test, y_pred_rf):.2f}")
    st.markdown(f"**XGBoost Accuracy:** {accuracy_score(y_test, y_pred_xgb):.2f}")

    feat_df = pd.DataFrame({
        'Feature': features,
        'Random Forest': rf.feature_importances_,
        'XGBoost': xgb.feature_importances_
    }).set_index('Feature')

    feat_df.plot(kind='barh', figsize=(8,6))
    plt.title("Feature Importance")
    st.pyplot(plt.gcf())

# 5. Decision Tree
elif option == "Decision Tree (Fee Increase)":
    st.header("5. Fee Increase Classification")
    df['Fee Increase High'] = (df['Fee Increase % (YoY)'] > df['Fee Increase % (YoY)'].median()).astype(int)
    X = df[['Admission Fee (NPR)', 'Technology Access Index', 'Infrastructure Score']]
    y = df['Fee Increase High']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    tree = DecisionTreeClassifier()
    tree.fit(X_train, y_train)
    y_pred = tree.predict(X_test)

    st.markdown(f"**Accuracy:** {accuracy_score(y_test, y_pred):.2f}")

    fig4, ax4 = plt.subplots()
    sns.barplot(x=tree.feature_importances_, y=X.columns, ax=ax4)
    ax4.set_title("Feature Importance")
    st.pyplot(fig4)

    fig5, ax5 = plt.subplots(figsize=(10,5))
    plot_tree(tree, feature_names=X.columns, class_names=['Low','High'], max_depth=1, filled=True, ax=ax5)
    st.pyplot(fig5)
