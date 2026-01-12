import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, mean_absolute_error, mean_squared_error
from imblearn.over_sampling import SMOTE
import plotly.express as px

# ── Global variables for prediction (set after training) ──
best_model = None
fitted_scaler = None
fitted_encoder = None

# Load and preprocess the data
@st.cache_data
def load_data():
    try:
        data = pd.read_csv('clean_data.csv')  # Make sure this file exists in the same folder
    except FileNotFoundError:
        st.error("File 'clean_data.csv' not found. Please place it in the same directory as this script.")
        st.stop()
    
    data['date'] = pd.to_datetime(data['date'])

    # Feature Engineering
    data['output_per_hour'] = data['output_hour'] / data['hours'].replace(0, np.nan)
    data['GDP_per_worker'] = data['gdp'] / data['employment'].replace(0, np.nan)
    data['log_GDP'] = np.log(data['gdp'] + 1)
    
    data['employed_employer_percentage']    = data['employed_employer'] / data['employment'] * 100
    data['employed_employee_percentage']    = data['employed_employee'] / data['employment'] * 100
    data['employed_own_account_percentage'] = data['employed_own_account'] / data['employment'] * 100
    data['employed_unpaid_family_percentage']= data['employed_unpaid_family'] / data['employment'] * 100

    data.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    data[numeric_cols] = data[numeric_cols].fillna(data[numeric_cols].mean())

    return data

def preprocess_data(data):
    encoder = LabelEncoder()
    y = data['sector']
    y_encoded = encoder.fit_transform(y)
    
    X = data[['gdp', 'employment', 'hours', 'output_per_hour', 'GDP_per_worker', 'log_GDP',
              'employed_employer_percentage', 'employed_employee_percentage',
              'employed_own_account_percentage', 'employed_unpaid_family_percentage']]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    
    X_train = X_train.replace([np.inf, -np.inf], np.nan).fillna(X_train.mean())
    X_test  = X_test.replace([np.inf, -np.inf], np.nan).fillna(X_train.mean())   # use train means

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    smote = SMOTE(random_state=42)
    X_train_sm, y_train_sm = smote.fit_resample(X_train_scaled, y_train)
    
    return X_train_sm, X_test_scaled, y_train_sm, y_test, encoder, scaler

def train_models(X_train_sm, y_train_sm):
    models = {
        "XGBoost": xgb.XGBClassifier(n_estimators=200, learning_rate=0.1, max_depth=5, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42),
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "SVM": SVC(kernel='rbf', probability=True, random_state=42)
    }
    
    trained = {}
    for name, model in models.items():
        model.fit(X_train_sm, y_train_sm)
        trained[name] = model
    
    return trained

def evaluate_models(trained_models, X_test_scaled, y_test, encoder):
    results = []
    for name, model in trained_models.items():
        y_pred = model.predict(X_test_scaled)
        acc = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        
        results.append({
            "Model": name,
            "Accuracy": acc,
            "MAE": mean_absolute_error(y_test, y_pred),
            "RMSE": np.sqrt(mean_squared_error(y_test, y_pred)),
            "Precision (macro)": report['macro avg']['precision'],
            "Recall (macro)": report['macro avg']['recall'],
            "F1-Score (macro)": report['macro avg']['f1-score'],
        })
    
    return pd.DataFrame(results)

# ── The missing prediction function ───────────────────────────────────
def make_prediction(gdp, employment, hours):
    global best_model, fitted_scaler, fitted_encoder
    
    if best_model is None or fitted_scaler is None or fitted_encoder is None:
        return "Model not ready yet. Please wait a moment."
    
    if employment <= 0:
        return "Employment must be greater than 0"
    
    output_per_hour = hours / employment
    gdp_per_worker  = gdp / employment
    log_gdp         = np.log(gdp + 1)
    
    # Rough fallback values — in production you should use sector-specific medians from training data
    employer_pct     = 10.0
    employee_pct     = 70.0
    own_account_pct  = 15.0
    unpaid_pct       = 5.0
    
    input_dict = {
        'gdp': gdp,
        'employment': employment,
        'hours': hours,
        'output_per_hour': output_per_hour,
        'GDP_per_worker': gdp_per_worker,
        'log_GDP': log_gdp,
        'employed_employer_percentage': employer_pct,
        'employed_employee_percentage': employee_pct,
        'employed_own_account_percentage': own_account_pct,
        'employed_unpaid_family_percentage': unpaid_pct,
    }
    
    input_df = pd.DataFrame([input_dict])
    input_scaled = fitted_scaler.transform(input_df)
    
    pred_encoded = best_model.predict(input_scaled)[0]
    predicted_sector = fitted_encoder.inverse_transform([pred_encoded])[0]
    
    # Show confidence if the model supports predict_proba
    try:
        proba = best_model.predict_proba(input_scaled)[0]
        confidence = proba.max() * 100
        return f"{predicted_sector} ({confidence:.1f}% confidence)"
    except:
        return predicted_sector

# ── Visualization functions ───────────────────────────────────────────
def plot_employment_distribution(data):
    sector_emp = data.groupby('sector')['employment'].sum().sort_values(ascending=False)
    fig = px.pie(sector_emp, values=sector_emp.values, names=sector_emp.index,
                 title="Share of Total Employment by Sector")
    st.plotly_chart(fig, use_container_width=True)

def plot_gdp_trend_by_sector(data):
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=data, x='date', y='gdp', hue='sector', marker='o')
    plt.title('GDP Trend Over Time by Sector')
    plt.xlabel('Date')
    plt.ylabel('GDP')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    st.pyplot(plt)

def plot_scatter_gdp_vs_other(data):
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    plots = [
        ('employment', 'Employment', 'steelblue'),
        ('hours', 'Total Working Hours', 'green'),
        ('output_hour', 'Output per Hour', 'orange'),
        ('output_employment', 'Output per Employee', 'purple')
    ]
    
    for i, (col, title, color) in enumerate(plots):
        ax = axes[i]
        ax.scatter(data['gdp'], data[col], color=color, alpha=0.6)
        ax.set_xlabel('GDP')
        ax.set_ylabel(title)
        ax.set_title(f'GDP vs {title}')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    st.pyplot(fig)

def plot_heatmap(data):
    key_vars = ['gdp', 'employment', 'hours', 'output_hour', 'output_employment']
    corr = data[key_vars].corr()
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm")
    plt.title("Correlation Heatmap")
    st.pyplot(plt)

# ── Main app ──────────────────────────────────────────────────────────
def main():
    st.title("EXPLORING LABOUR MARKET DYNAMICS: EMPLOYMENT BY MSIC IN MALAYSIA")

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 EDA", "📈 Trends", "🤖 ML Results", "🚀 Prediction", "ℹ️ About"
    ])

    with st.spinner("Loading data and training models... Please wait"):
        data = load_data()
        X_train_sm, X_test_scaled, y_train_sm, y_test, encoder, scaler = preprocess_data(data)
        trained_models = train_models(X_train_sm, y_train_sm)
        results_df = evaluate_models(trained_models, X_test_scaled, y_test, encoder)

        # Select best model (highest macro F1-score)
        global best_model, fitted_scaler, fitted_encoder
        best_row = results_df.loc[results_df['F1-Score (macro)'].idxmax()]
        best_name = best_row['Model']
        best_model = trained_models[best_name]
        fitted_scaler = scaler
        fitted_encoder = encoder

        st.session_state['results'] = results_df
        st.session_state['best_model_name'] = best_name

    with tab1:
        st.header("Exploratory Data Analysis (EDA)")
        plot_employment_distribution(data)
        plot_scatter_gdp_vs_other(data)
        plot_heatmap(data)

    with tab2:
        st.header("GDP Trends Over Time")
        plot_gdp_trend_by_sector(data)

    with tab3:
        st.header("Machine Learning Model Performance")
        st.subheader(f"Best Model: **{st.session_state.get('best_model_name', 'N/A')}**")
        st.dataframe(st.session_state['results'].style.format({
            'Accuracy': '{:.4f}',
            'MAE': '{:.4f}',
            'RMSE': '{:.4f}',
            'Precision (macro)': '{:.4f}',
            'Recall (macro)': '{:.4f}',
            'F1-Score (macro)': '{:.4f}'
        }))

    with tab4:
        st.header("Predict Employment Sector")
        st.write("Enter economic values to predict the most likely sector.")

        col1, col2, col3 = st.columns(3)
        with col1:
            gdp_input = st.number_input("GDP (Billion USD)", min_value=0.0, value=500.0, step=10.0)
        with col2:
            emp_input = st.number_input("Total Employment", min_value=1, value=500000, step=10000)
        with col3:
            hours_input = st.number_input("Total Working Hours", min_value=1.0, value=40.0 * 500000, step=100000.0)

        if st.button("🔮 Predict Sector", type="primary"):
            with st.spinner("Predicting..."):
                result = make_prediction(gdp_input, emp_input, hours_input)
                st.success(f"Predicted Sector: **{result}**")

    with tab5:
        st.header("About This Project")
        st.write("""
        This dashboard uses machine learning to predict the dominant economic sector 
        based on GDP, employment, and working hours.
        
        Models trained: XGBoost, Random Forest, Logistic Regression, SVM.
        """)

if __name__ == "__main__":
    main()


