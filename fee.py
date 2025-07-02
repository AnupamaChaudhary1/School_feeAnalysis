import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import joblib

# Load cleaned data
df = pd.read_csv("cleaned_data.csv")

st.set_page_config(page_title="Nepal Private School Fee ML Dashboard", layout="wide")
st.title("🎓 Private School Fee Analysis in Nepal")

# Create tabs for each use case
tabs = st.tabs([
    "📈 Linear Regression (Tuition Prediction)",
    "📊 Logistic Regression (Scholarship Classification)",
    "🔍 K-Means Clustering (School Segments)",
    "📚 Random Forest/XGBoost (Academic Performance)",
    "💰 Decision Tree (Fee Increase Analysis)"
])

# --- Tab 1: Linear Regression ---
with tabs[0]:
    st.header("Use Case 1: Predict Annual Tuition Fee")
    X = df[['Infrastructure Score', 'Technology Access Index', 'Average Academic Score (%)']]
    y = df['Annual Tuition Fee (NPR)']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_pred = lr.predict(X_test)

    st.markdown(f"**MAE:** {mean_absolute_error(y_test, y_pred):.2f}  ")
    st.markdown(f"**RMSE:** {mean_squared_error(y_test, y_pred, squared=False):.2f}")

    fig1, ax1 = plt.subplots()
    ax1.scatter(y_test, y_pred, alpha=0.5)
    ax1.set_xlabel("Actual Tuition Fee")
    ax1.set_ylabel("Predicted Fee")
    ax1.set_title("Actual vs Predicted Tuition Fee")
    st.pyplot(fig1)

# --- Tab 2: Logistic Regression ---
with tabs[1]:
    st.header("Use Case 2: Classify High/Low Scholarship Schools")
    df['High Scholarship'] = (df['Scholarship % Availability'] > df['Scholarship % Availability'].median()).astype(int)
    features = ['Infrastructure Score', 'Monthly Fee (NPR)', 'Admission Fee (NPR)']
    X = df[features]
    y = df['High Scholarship']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LogisticRegression(max_iter=500)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    st.write(f"**Accuracy:** {accuracy_score(y_test, y_pred):.2f}")

    coef_df = pd.DataFrame({
        'Feature': features,
        'Coefficient': model.coef_[0]
    }).sort_values(by='Coefficient')

    fig2, ax2 = plt.subplots()
    ax2.barh(coef_df['Feature'], coef_df['Coefficient'], color='green')
    ax2.set_title('Feature Influence on High Scholarship')
    st.pyplot(fig2)

# --- Tab 3: K-Means Clustering ---
with tabs[2]:
    st.header("Use Case 3: Segment Schools by Fee Tier")
    X = df[['Annual Tuition Fee (NPR)', 'Technology Access Index', 'Infrastructure Score']]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    df['Cluster'] = kmeans.fit_predict(X_scaled)

    centers = scaler.inverse_transform(kmeans.cluster_centers_)
    center_df = pd.DataFrame(centers, columns=X.columns)

    mean_fees = center_df['Annual Tuition Fee (NPR)']
    sorted_clusters = mean_fees.sort_values().index.tolist()
    cluster_map = {sorted_clusters[0]: 'Budget', sorted_clusters[1]: 'Mid-tier', sorted_clusters[2]: 'Premium'}
    df['Cluster Name'] = df['Cluster'].map(cluster_map)

    fig3, ax3 = plt.subplots()
    sns.scatterplot(data=df, x='Annual Tuition Fee (NPR)', y='Infrastructure Score', hue='Cluster Name', palette='Set2', ax=ax3, s=100)
    ax3.set_title("K-Means Clustering of Schools")
    st.pyplot(fig3)

# --- Tab 4: Academic Prediction ---
with tabs[3]:
    st.header("Use Case 4: Predict High Academic Performance")
    df['High Academic'] = (df['Average Academic Score (%)'] > df['Average Academic Score (%)'].mean()).astype(int)
    features = ['Technology Access Index', 'Infrastructure Score', 'Monthly Fee (NPR)', 'Student-Teacher Ratio']
    X = df[features]
    y = df['High Academic']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    rf = RandomForestClassifier(random_state=42)
    xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    rf.fit(X_train, y_train)
    xgb.fit(X_train, y_train)

    importances = pd.DataFrame({
        'Feature': features,
        'Random Forest': rf.feature_importances_,
        'XGBoost': xgb.feature_importances_
    }).set_index('Feature')

    st.write(f"Random Forest Accuracy: {accuracy_score(y_test, rf.predict(X_test)):.2f}")
    st.write(f"XGBoost Accuracy: {accuracy_score(y_test, xgb.predict(X_test)):.2f}")

    fig4, ax4 = plt.subplots()
    importances.plot(kind='barh', ax=ax4, color=['skyblue', 'orange'])
    ax4.set_title('Feature Importance')
    st.pyplot(fig4)

# --- Tab 5: Decision Tree Fee Increase ---
with tabs[4]:
    st.header("Use Case 5: Predict Fee Increase Pattern")
    df['Fee Increase High'] = (df['Fee Increase % (YoY)'] > df['Fee Increase % (YoY)'].median()).astype(int)
    X = df[['Admission Fee (NPR)', 'Technology Access Index', 'Infrastructure Score']]
    y = df['Fee Increase High']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    tree = DecisionTreeClassifier()
    tree.fit(X_train, y_train)
    y_pred = tree.predict(X_test)
    st.write(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")

    fig5, ax5 = plt.subplots()
    sns.barplot(x=tree.feature_importances_, y=X.columns, ax=ax5)
    ax5.set_title("Feature Importance for Fee Increase")
    st.pyplot(fig5)

    fig6, ax6 = plt.subplots(figsize=(8,5))
    plot_tree(tree, max_depth=1, feature_names=X.columns, class_names=['Low', 'High'], filled=True, ax=ax6)
    st.pyplot(fig6)

    joblib.dump(tree, "fee_model.plk")
