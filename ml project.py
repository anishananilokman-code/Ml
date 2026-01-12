import streamlit as st 
import pandas as pd
import numpy as np
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import xgboost as xgb
from sklearn.svm import SVC

# ===============================
# PAGE CONFIGURATION
# ===============================
st.set_page_config(
    page_title="Employment Sector Prediction",
    page_icon="🏢",
    layout="wide"
)

# ===============================
# CUSTOM STYLING FOR DASHBOARD
# ===============================
def apply_custom_styles():
    st.markdown(
        """
        <style>
        .stApp {
            background-color: #f3f4f6;  /* Light background color */
            color: black; /* Black text color */
        }
        .stButton>button {
            background-color: #0073e6;
            color: white;
            font-weight: bold;
        }
        .block-container {
            background-color: #f9f9f9;  /* Light container background */
            padding: 2rem;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        .sidebar .sidebar-content {
            background-color: #2b2d42;
            color: white;
        }
        .sidebar .sidebar-content a {
            color: #fff;
        }
        .stTextInput>label {
            font-size: 1rem;
        }
        .stSelectbox, .stMultiselect {
            background-color: #ffffff;
            border-radius: 5px;
        }
        .stMetric>label {
            font-size: 1.2rem;
        }
        .stButton {
            margin-top: 10px;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

apply_custom_styles()

# ===============================
# MAIN TITLE
# ===============================
st.markdown("<h1 style='color: #0073e6;'>🏢 Employment Sector Prediction Dashboard</h1>", unsafe_allow_html=True)
st.subheader("Predicting Employment Sectors based on Economic Indicators")
st.caption("📊 GDP | 🏭 Productivity | 💼 Work Hours | 👥 Labor Force")

# ===============================
# DATA LOADING (Excel file)
# ===============================
@st.cache_data
def load_data():
    # Use read_excel instead of read_csv for Excel files
    data = pd.read_excel(r'clean_data.xlsx')
    return data

data = load_data()

# ===============================
# Data Preprocessing
# ===============================
def preprocess_data(data):
    # Feature Engineering Example:
    data['GDP_per_worker'] = data['gdp'] / data['employment']
    data['output_per_hour'] = data['output_hour'] / data['hours']
    data['log_GDP'] = np.log(data['gdp'] + 1)

    # Label Encoding for sector (assuming sector is categorical)
    encoder = LabelEncoder()
    data['sector'] = encoder.fit_transform(data['sector'])

    # Handle Infinite and NaN Values
    data.replace([np.inf, -np.inf], np.nan, inplace=True)  # Replace infinite values with NaN
    data.fillna(data.mean(), inplace=True)  # Replace NaN values with the mean of each column

    # Split data into features and target
    X = data[['gdp', 'employment', 'hours', 'output_per_hour', 'GDP_per_worker', 'log_GDP']]
    y = data['sector']

    # Split into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scale the data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test, encoder, scaler

X_train_scaled, X_test_scaled, y_train, y_test, encoder, scaler = preprocess_data(data)

# ===============================
# NAVIGATION (Sidebar)
# ===============================
st.sidebar.title("Navigation")
section = st.sidebar.radio("Go to", ["Data Overview", "Model", "Prediction"])

# ===============================
# SECTION 1: Data Overview
# ===============================
if section == "Data Overview":
    st.markdown("<h1>Data Overview</h1>", unsafe_allow_html=True)
    st.subheader("Dataset Description")

    # Data description
    st.markdown("""
    The dataset contains several economic indicators for Malaysia over the years.
    **Features in Dataset**:
    - **GDP (Gross Domestic Product)**: Total value of goods and services produced.
    - **Employment**: Number of workers in each sector.
    - **Work Hours**: Average hours worked.
    - **Output per Hour**: Productivity based on hours worked.
    - **Sector**: Categorizes the data into sectors like **Agriculture**, **Manufacturing**, **Services**.
    """)

    st.subheader("Feature Engineering")
    # Feature Engineering Process (e.g., GDP per Worker, Log Transforms, etc.)
    st.write("New features engineered:")
    st.write(data[['sector', 'GDP_per_worker', 'output_per_hour', 'log_GDP']].head())

    # ===============================
    # Exploratory Data Analysis (EDA)
    # ===============================
    st.subheader("Exploratory Data Analysis (EDA)")

    # Plot GDP vs Employment
    fig_gdp = px.scatter(data, x="gdp", y="employment", color="sector", 
                         title="GDP vs Employment by Sector")
    st.plotly_chart(fig_gdp)

    # Correlation matrix
    st.subheader("Correlation Matrix")
    corr_matrix = data.corr()
    fig_corr = plt.figure(figsize=(10, 7))
    sns.heatmap(corr_matrix, annot=True, cmap="Blues", fmt=".2f", cbar=True)
    st.pyplot(fig_corr)

    # Scatter plots for key relationships
    st.subheader("Scatter Plots for Key Variables")
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    x = data['gdp']
    y_employment = data['employment']
    y_hours = data['hours']
    y_output_hour = data['output_hour']
    y_output_emp = data['output_employment']

    # GDP vs Employment
    axes[0].scatter(x, y_employment, color='steelblue', alpha=0.6, label='Observed Data')
    z = np.polyfit(x, y_employment, 1)
    p = np.poly1d(z)
    axes[0].plot(x, p(x), color='darkred', linestyle='--', linewidth=2, label='Linear Fit')
    axes[0].set_xlabel('GDP')
    axes[0].set_ylabel('Employment')
    axes[0].set_title('GDP vs Employment')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # GDP vs Total Working Hours
    axes[1].scatter(x, y_hours, color='green', alpha=0.6, label='Observed Data')
    z = np.polyfit(x, y_hours, 1)
    p = np.poly1d(z)
    axes[1].plot(x, p(x), color='darkred', linestyle='--', linewidth=2, label='Linear Fit')
    axes[1].set_xlabel('GDP')
    axes[1].set_ylabel('Total Working Hours')
    axes[1].set_title('GDP vs Total Working Hours')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # GDP vs Output per Hour
    axes[2].scatter(x, y_output_hour, color='orange', alpha=0.6, label='Observed Data')
    axes[2].set_xlabel('GDP')
    axes[2].set_ylabel('Output per Hour')
    axes[2].set_title('GDP vs Output per Hour')
    axes[2].grid(True, alpha=0.3)

    # GDP vs Output per Employee
    axes[3].scatter(x, y_output_emp, color='purple', alpha=0.6, label='Observed Data')
    axes[3].set_xlabel('GDP')
    axes[3].set_ylabel('Output per Employee')
    axes[3].set_title('GDP vs Output per Employee')
    axes[3].grid(True, alpha=0.3)

    plt.tight_layout()
    st.pyplot(fig)

# ===============================
# SECTION 2: Model
# ===============================
elif section == "Model":
    st.markdown("<h1>Model Selection and Comparison</h1>", unsafe_allow_html=True)

    st.subheader("Model Selection")

    st.markdown("""
    **Selected Models**:
    - **XGBoost**: For its high performance on structured data and ability to handle missing values.
    - **Random Forest**: An ensemble method that improves performance by aggregating multiple decision trees.
    - **Logistic Regression**: A simple baseline model to compare others against.
    - **Support Vector Machine (SVM)**: Effective for non-linear decision boundaries.
    """)

    # ===============================
    # Model Comparison (Performance Metrics)
    # ===============================
    st.subheader("Model Comparison")

    def train_and_evaluate_models(X_train, X_test, y_train, y_test):
        models = {
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'XGBoost': xgb.XGBClassifier(n_estimators=100, random_state=42),
            'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
            'SVM': SVC(kernel='linear', random_state=42)
        }

        model_results = []

        for model_name, model in models.items():
            # Train the model
            model.fit(X_train, y_train)

            # Make predictions
            y_pred = model.predict(X_test)

            # Evaluate performance
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='weighted')
            recall = recall_score(y_test, y_pred, average='weighted')
            f1 = f1_score(y_test, y_pred, average='weighted')

            model_results.append({
                'Model': model_name,
                'Accuracy': accuracy,
                'Precision': precision,
                'Recall': recall,
                'F1-Score': f1
            })

        return pd.DataFrame(model_results)

    model_comparison = train_and_evaluate_models(X_train_scaled, X_test_scaled, y_train, y_test)

    st.dataframe(model_comparison)

    # Visualize model performance
    st.subheader("Model Comparison - Accuracy")
    fig = px.bar(model_comparison, x="Model", y="Accuracy", color="Model", 
                 title="Model Comparison - Accuracy")
    st.plotly_chart(fig)

    st.subheader("Model Comparison - F1-Score")
    fig_f1 = px.bar(model_comparison, x="Model", y="F1-Score", color="Model", 
                    title="Model Comparison - F1-Score")
    st.plotly_chart(fig_f1)

    st.subheader("Best Model Selection")
    best_model_name = model_comparison.loc[model_comparison['Accuracy'].idxmax(), 'Model']
    st.markdown(f"Based on the comparison, **{best_model_name}** is selected as the best model due to its **high accuracy** and **balanced performance**.")

# ===============================
# SECTION 3: Prediction
# ===============================
elif section == "Prediction":
    st.markdown("<h1 style='color: #0073e6;'>Live Prediction: Employment Sector</h1>", unsafe_allow_html=True)

    st.subheader("Enter the economic indicators below to predict the employment sector in Malaysia.")

    # Sidebar inputs to remain visible for prediction
    st.sidebar.markdown("### Enter Key Economic Indicators")

    # Inputs for prediction (more styled with icons and ranges)
    gdp_input = st.sidebar.number_input("GDP (Billion USD)", min_value=0.0, max_value=5000.0, value=1000.0, step=100.0)
    work_hours_input = st.sidebar.number_input("Average Work Hours", min_value=0, max_value=100, value=40, step=1)
    employment_input = st.sidebar.number_input("Employment Figures", min_value=0, max_value=1000000, value=500000, step=5000)
    output_per_hour_input = st.sidebar.number_input("Output per Hour (units)", min_value=0.0, max_value=1000.0, value=150.0, step=10.0)

    # Calculate GDP per Worker and log_GDP (Feature Engineering)
    gdp_per_worker_input = gdp_input / employment_input if employment_input != 0 else 0
    log_gdp_input = np.log(gdp_input + 1)  # Log transformation of GDP

    # Prepare input data for prediction with all 6 features
    input_data = np.array([[gdp_input, employment_input, work_hours_input, output_per_hour_input, gdp_per_worker_input, log_gdp_input]])

    # Scale the input data using the same scaler used during training
    input_data_scaled = scaler.transform(input_data)

    if st.sidebar.button("Predict Sector", key="predict_button"):
        model = xgb.XGBClassifier(n_estimators=100, random_state=42)  # Train your model
        model.fit(X_train_scaled, y_train)  # Using preprocessed data for training
        prediction = model.predict(input_data_scaled)
        predicted_sector = encoder.inverse_transform(prediction)[0]

        # Displaying the prediction with sector icon and color highlight
        st.markdown(f"### 🎯 **Predicted Employment Sector: {predicted_sector}**", unsafe_allow_html=True)
        if predicted_sector == 0:
            st.image("https://image.shutterstock.com/image-vector/agriculture-icon-flat-symbol-illustration-260nw-1049047039.jpg", width=50)
        elif predicted_sector == 1:
            st.image("https://image.shutterstock.com/image-vector/manufacturing-icon-flat-symbol-illustration-260nw-1049047040.jpg", width=50)
        else:
            st.image("https://image.shutterstock.com/image-vector/services-icon-flat-symbol-illustration-260nw-1049047041.jpg", width=50)

        # Displaying more information about the prediction
        st.markdown("""
        The predicted sector is based on the economic indicators provided, such as GDP, employment, 
        and output per hour. This prediction helps understand the likely economic sector that will 
        thrive based on the given parameters.
        """)
        
        # Adding more interactivity
        st.markdown("""
        You can **change the values** above to explore how different economic conditions affect 
        the employment sector prediction.
        """)

    # Visualization for showing the relationship between the features and prediction
    st.subheader("🔍 Visualizing the Key Inputs")
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111)

    # Bar plot for the entered data
    data_for_plot = pd.DataFrame({
        'Input Feature': ['GDP', 'Employment', 'Work Hours', 'Output per Hour'],
        'Value': [gdp_input, employment_input, work_hours_input, output_per_hour_input]
    })
    ax.bar(data_for_plot['Input Feature'], data_for_plot['Value'], color=['#3498db', '#2ecc71', '#e74c3c', '#9b59b6'])
    ax.set_title("Entered Economic Indicators", fontsize=16)
    ax.set_ylabel("Value")
    st.pyplot(fig)


