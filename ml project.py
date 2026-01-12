import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import xgboost as xgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# ===============================
# PAGE CONFIGURATION
# ===============================
st.set_page_config(
    page_title="Employment Sector Prediction",
    page_icon="🏢",
    layout="wide"
)

# ===============================
# BACKGROUND & STYLING
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
            border-radius: 10px;
            box-shadow: 0 2px 6px rgba(0, 0, 0, 0.2);
            transition: background-color 0.3s ease;
        }
        .stButton>button:hover {
            background-color: #005bb5;
        }
        .block-container {
            background-color: rgba(255,255,255,0.92);
            padding: 2rem;
            border-radius: 15px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
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
        .stSidebar {
            background-color: #2b2d42;
        }
        .stSidebar .sidebar-content a {
            color: #fff;
        }
        .stTitle, .stSubheader {
            color: #0073e6;
            font-family: 'Arial', sans-serif;
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
# DATA LOADING (Uploaded CSV file)
# ===============================
@st.cache_data
def load_data():
    data = pd.read_csv(r"clean_data.csv")  # Update the path to your dataset
    # Ensure 'date' column exists and handle it properly
    if 'date' in data.columns:
        data['date'] = pd.to_datetime(data['date'], errors='coerce')  # Coerce invalid dates to NaT
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
tab = st.sidebar.radio("Go to", ["Project Overview & Motivation", "Data Overview", "📊 EDA", "📈 Trends", "📊 Sector Productivity", "📉 Employment Distribution", "💼 Sector GDP Trend", "📊 Model Overview", "📝 Prediction"])

# ===============================
# TAB 1: Project Overview & Motivation
# ===============================
if tab == "Project Overview & Motivation":
    st.markdown("<h1>Project Overview & Motivation</h1>", unsafe_allow_html=True)

    st.subheader("1. Problem Statement")
    st.markdown("""
    The imbalance in the structure of labour force participation and productivity performance between key economic sectors remains, despite Malaysia’s effort to continually upgrade labour market efficiency and sectoral productivity. Certain key sectors have shown relatively high employment participation with low productivity, while others contribute significantly to national output with fewer workers.
    """)

    st.subheader("2. Motivation of Project")
    st.markdown("""
    Different sectors contribute to Malaysian economic growth through a variety of means. While GDP can measure the performance of various sectors economically, a more accurate indicator of productivity is how well labour resources are used. This research project seeks to examine in detail the labour force status and productivity level by sector to help policymakers ensure strength in workforce allocation, productivity improvement, and sustainable economic growth.
    """)

    st.subheader("3. Project Objectives")
    st.markdown("""
    1. To develop a machine learning model that classifies employment sectors based on productivity indicators, working hours, labour force size, and GDP contribution.
    2. To train and compare multiple classification models to identify the best performing algorithm.
    3. To deploy the final model in the dashboard.
    """)

    st.subheader("4. Project Limitations")
    st.markdown("""
    Limitations of the current research include the reliance on secondary data from the **Department of Statistics Malaysia (DOSM)**, which may have some inconsistencies. Additionally, the accuracy of predictions depends on the available indicators and may vary over time.
    """)

# ===============================
# TAB 2: Data Overview
# ===============================
if tab == "Data Overview":
    st.markdown("<h1>Data Overview</h1>", unsafe_allow_html=True)
    st.subheader("Data Sources")

    # Display Data Sources and Description
    st.markdown("""
    - **Department of Statistics Malaysia (DOSM)** provided secondary data used in this study.
    - The dataset includes information on employment, productivity, GDP, working hours, and other key variables.
    """)

    st.subheader("Data Structure and Features")
    st.markdown("""
    **Features**:
    - **Date**: Year of the data.
    - **Sector**: Employment sector.
    - **GDP**: Gross domestic product.
    - **Hours Worked**: Total hours worked.
    - **Employment**: Number of employed persons in the sector.
    - **Output per Hour**: Ratio of GDP to hours worked.
    - **Output Employment**: Ratio of GDP to the number of employed persons.
    """)

    st.write(data.head())

# ===============================
# TAB 3: EDA (Exploratory Data Analysis)
# ===============================
if tab == "📊 EDA":
    st.markdown("<h1>Exploratory Data Analysis (EDA)</h1>", unsafe_allow_html=True)
    st.subheader("Correlation Matrix")

    # Key variables for correlation
    key_vars = ['gdp', 'employment', 'hours', 'output_hour', 'output_employment']

    # Correlation matrix
    corr_matrix = data[key_vars].corr()
    fig_corr = plt.figure(figsize=(8,6))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="plasma")
    plt.title("Correlation Heatmap")
    st.pyplot(fig_corr)

# ===============================
# TAB 4: Trends (Scatter Plots)
# ===============================
if tab == "📈 Trends":
    st.markdown("<h1>Trends and Relationships</h1>", unsafe_allow_html=True)
    st.subheader("Scatter Plots for Key Variables")

    x = data['gdp']
    y_employment = data['employment']
    y_hours = data['hours']
    y_output_hour = data['output_hour']
    y_output_emp = data['output_employment']

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

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
# TAB 5: Sector Productivity (Bar Chart)
# ===============================
if tab == "📊 Sector Productivity":
    st.markdown("<h1>Sector Productivity</h1>", unsafe_allow_html=True)
    st.subheader("Average Output per Hour by Sector")

    # Calculate the average output per hour by sector
    sector_productivity = data.groupby('sector')['output_hour'].mean().sort_values(ascending=False)

    # Plot
    sector_productivity.plot(kind='bar')
    plt.title('Average Output per Hour by Sector')
    plt.ylabel('Output per Hour')
    st.pyplot(plt)

# ===============================
# TAB 6: Employment Distribution (Pie Chart)
# ===============================
if tab == "📉 Employment Distribution":
    st.markdown("<h1>Employment Distribution by Sector</h1>", unsafe_allow_html=True)

    # Calculate total employment by sector
    sector_emp_total = data.groupby('sector')['employment'].sum().sort_values(ascending=False)

    # Plot Pie Chart
    fig_pie = px.pie(sector_emp_total, values=sector_emp_total.values, names=sector_emp_total.index,
                     title="Share of Total Employment by Sector")
    st.plotly_chart(fig_pie)

# ===============================
# TAB 7: Sector GDP Trend (Line Plot)
# ===============================
if tab == "💼 Sector GDP Trend":
    st.markdown("<h1>GDP Trend Over Time by Sector</h1>", unsafe_allow_html=True)

    # Create a dictionary to map encoded labels back to original colors
    original_colors_map = {
        'agriculture': 'green',
        'construction': 'red',
        'manufacturing': 'orange',
        'mining': 'yellow',
        'overall': 'purple',
        'services': 'blue'
    }

    # Plot GDP Trend by Sector
    plt.figure(figsize=(10, 6))

    for sector_name, color in original_colors_map.items():
        sector_data = data[data['sector'] == sector_name]
        sns.lineplot(data=sector_data, x='date', y='gdp', color=color, label=sector_name)

    plt.title('GDP Trend Over Time by Sector')
    plt.xlabel('Date')
    plt.ylabel('GDP')
    plt.legend(title='Sector', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    st.pyplot(plt)

# ===============================
# TAB 8: Model Overview
# ===============================
if tab == "📊 Model Overview":
    st.markdown("<h1>Model Overview and Performance</h1>", unsafe_allow_html=True)
    
    st.markdown("""
    The model classifies employment sectors based on indicators such as **productivity**, **working hours**, **labour force size**, and **GDP contribution**.
    We have trained multiple models, including:
    - **XGBoost** (for its high performance on structured data)
    - **Random Forest**
    - **Logistic Regression**
    - **SVM**
    """)
    
    st.markdown("### Model Evaluation:") 
    
    # Evaluate and compare the performance of models (e.g., using accuracy and F1-score)
    st.markdown("The models were evaluated based on metrics such as **accuracy**, **precision**, **recall**, and **F1-score**.")
    
    st.markdown("""
    **Model Performance**:
    - **XGBoost** outperforms others in terms of accuracy.
    - **SVM** performs well in non-linear boundaries.
    - **Random Forest** provides robustness in predictions across various conditions.
    """)

# ===============================
# TAB 9: Prediction
# ===============================
if tab == "📝 Prediction":
    st.markdown("<h1>Prediction</h1>", unsafe_allow_html=True)
    st.subheader("Enter the economic indicators to predict the sector")

    # User input for prediction
    gdp_input = st.number_input("GDP (Billion USD)", min_value=0.0, max_value=5000.0, value=1000.0, step=100.0)
    hours_input = st.number_input("Hours Worked", min_value=0, max_value=100, value=40)
    employment_input = st.number_input("Employment Figures", min_value=0, max_value=1000000, value=500000)
    output_hour_input = st.number_input("Output per Hour (units)", min_value=0.0, max_value=1000.0, value=150.0)

    # Feature Engineering for Prediction
    GDP_per_worker = gdp_input / employment_input if employment_input != 0 else 0
    log_GDP = np.log(gdp_input + 1)  # Log transformation to handle skewness

    # Prepare input data for prediction
    input_data = np.array([[gdp_input, employment_input, hours_input, output_hour_input, GDP_per_worker, log_GDP]])

    # Scale the data
    input_data_scaled = scaler.transform(input_data)

    if st.button("Predict Sector"):
        # Use trained model for prediction
        model = xgb.XGBClassifier(n_estimators=100, random_state=42)
        model.fit(X_train_scaled, y_train)  # Use the pre-trained data

        # Prediction
        prediction = model.predict(input_data_scaled)
        predicted_sector = encoder.inverse_transform(prediction)[0]
        st.markdown(f"### 🎯 **Predicted Employment Sector: {predicted_sector}**", unsafe_allow_html=True)

# ===============================
# FOOTER
# ===============================
st.markdown("---")
st.caption("💡 Streamlit Dashboard | Employment Sector Prediction | Machine Learning")

