import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import squarify
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    mean_absolute_error, mean_squared_error, classification_report
)
import xgboost as xgb
# ─────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────
st.set_page_config(page_title="Malaysia Labour Market Dashboard", layout="wide")
# ─────────────────────────────────────────────────────────────
# Load & clean data
# ─────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('clean_data.csv')
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df = df.dropna(subset=['date'])
        df = df[df['employment'] > 500].copy() # remove percentage-like rows
        df = df.drop_duplicates()
        df = df.sort_values('date').groupby(['date', 'sector']).first().reset_index()
        return df
    except FileNotFoundError:
        st.error("clean_data.csv not found.")
        st.stop()
data = load_data()
# ─────────────────────────────────────────────────────────────
# Sidebar filters + debug years
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Filters")
   
    st.caption("Available years in dataset:")
    st.write(data['date'].dt.year.value_counts().sort_index())
   
    min_year = int(data['date'].dt.year.min())
    max_year = int(data['date'].dt.year.max())
    year_range = st.slider("Year Range", min_year, max_year, (2016, 2022))
    all_sectors = sorted(data['sector'].unique())
    selected_sectors = st.multiselect("Sectors", all_sectors, default=all_sectors)
filtered_data = data[
    (data['date'].dt.year.between(year_range[0], year_range[1])) &
    (data['sector'].isin(selected_sectors))
].copy()
if filtered_data.empty:
    st.warning(f"No data for {year_range[0]}–{year_range[1]}. Check available years above.")
    st.stop()
# ─────────────────────────────────────────────────────────────
# Train & evaluate models
# ─────────────────────────────────────────────────────────────
@st.cache_resource
def train_and_evaluate():
    df = data.copy()
    le = LabelEncoder()
    y = le.fit_transform(df['sector'])
    features = [
        'gdp', 'employment', 'hours', 'output_hour', 'output_employment',
        'employed_employer', 'employed_employee', 'employed_own_account', 'employed_unpaid_family'
    ]
    features = [f for f in features if f in df.columns]
    X = df[features].fillna(0)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    models = {
        "XGBoost": xgb.XGBClassifier(random_state=42, eval_metric='mlogloss'),
        "Random Forest": RandomForestClassifier(random_state=42),
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "SVM": SVC(probability=True, random_state=42)
    }
    metrics_list = []
    preds_dict = {}
    for name, model in models.items():
        model.fit(X_train_s, y_train)
        y_pred = model.predict(X_test_s)
        preds_dict[name] = y_pred
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average='macro', zero_division=0)
        rec = recall_score(y_test, y_pred, average='macro', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        metrics_list.append({
            'Model': name,
            'Accuracy': acc,
            'Precision (macro)': prec,
            'Recall (macro)': rec,
            'F1-Score (macro)': f1,
            'MAE': mae,
            'RMSE': rmse
        })
    metrics_df = pd.DataFrame(metrics_list)
    best_model_name = metrics_df.loc[metrics_df['F1-Score (macro)'].idxmax(), 'Model']
    best_model = models[best_model_name]
    return {
        'best_model': best_model,
        'best_name': best_model_name,
        'scaler': scaler,
        'encoder': le,
        'metrics_df': metrics_df,
        'y_test': y_test,
        'predictions': preds_dict
    }
model_results = train_and_evaluate()
# ─────────────────────────────────────────────────────────────
# Main title
# ─────────────────────────────────────────────────────────────
st.title("Malaysia Labour Market Dashboard – MSIC Sectors")
st.caption("GDP, Employment & Sector Classification Analysis")
# ─────────────────────────────────────────────────────────────
# Tabs
# ─────────────────────────────────────────────────────────────
tab_kpi, tab_trends, tab_model, tab_predict, tab_data = st.tabs([
    "📊 Key Indicators",
    "📈 Trends & EDA",
    "📊 Model Performance",
    "🔮 Sector Prediction",
    "📋 Data Table"
])
# ─────────────────────────────────────────────────────────────
# Tab 1: Key Indicators
# ─────────────────────────────────────────────────────────────
with tab_kpi:
    st.subheader(f"Key Indicators – Latest year in filter ({filtered_data['date'].dt.year.max()})")
    latest_year = filtered_data['date'].dt.year.max()
    latest = filtered_data[filtered_data['date'].dt.year == latest_year]
    if not latest.empty:
        tot_emp = latest['employment'].sum()
        tot_gdp = latest['gdp'].sum()
        struc = latest[[
            'employed_employee', 'employed_employer',
            'employed_own_account', 'employed_unpaid_family'
        ]].sum()
        tot_workers = struc.sum() or 1
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Total Employment", f"{tot_emp:,.0f}")
        c2.metric("Total GDP", f"RM {tot_gdp:,.0f} mil")
        c3.metric("% Employees", f"{struc['employed_employee']/tot_workers*100:.1f}%")
        c4.metric("% Own-account", f"{struc['employed_own_account']/tot_workers*100:.1f}%")
        c5.metric("% Employers", f"{struc['employed_employer']/tot_workers*100:.1f}%")
    emp_share = filtered_data.groupby('sector')['employment'].sum().reset_index()
    fig_pie = px.pie(emp_share, values='employment', names='sector',
                     title="Share of Total Employment by Sector", hole=0.4)
    st.plotly_chart(fig_pie, use_container_width=True)
# ─────────────────────────────────────────────────────────────
# Tab 2: Trends & EDA (with your requested charts)
# ─────────────────────────────────────────────────────────────
with tab_trends:
    st.subheader("Sector Productivity & Correlations")
    # 1. Treemap + Pie + Bar combo: Share of Total Employment by Sector
    sector_emp_total = filtered_data.groupby('sector')['employment'].sum().sort_values(ascending=False)
    percentages = (sector_emp_total / sector_emp_total.sum() * 100).round(1)
    col_left, col_right = st.columns([2, 1])
    with col_left:
        # Treemap
        fig_treemap, ax = plt.subplots(figsize=(10, 8))
        squarify.plot(sizes=sector_emp_total, label=sector_emp_total.index,
                      value=percentages.apply(lambda x: f'{x}%'), alpha=0.8,
                      color=plt.cm.Blues(np.linspace(0.4, 0.9, len(sector_emp_total))))
        plt.title('Share of Total Employment by Sector (Treemap)', fontsize=14)
        plt.axis('off')
        st.pyplot(fig_treemap)
    with col_right:
        # Pie + Horizontal Bar side by side (smaller)
        fig_combo, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 8))
        # Pie
        ax1.pie(sector_emp_total, labels=sector_emp_total.index, autopct='%1.1f%%',
                startangle=90, colors=plt.cm.Set3(np.linspace(0, 1, len(sector_emp_total))))
        ax1.set_title('Employment Share (%)')
        # Horizontal Bar
        ax2.barh(sector_emp_total.index, sector_emp_total.values,
                 color=plt.cm.Set3(np.linspace(0, 1, len(sector_emp_total))))
        ax2.set_xlabel('Number of Employees')
        ax2.set_title('Absolute Employment')
        ax2.invert_yaxis()
        plt.tight_layout()
        st.pyplot(fig_combo)
    # 2. Average Output per Hour by Sector
    sector_hour_prod = filtered_data.groupby('sector')['output_hour'].mean().sort_values(ascending=False)
    fig_hour = px.bar(
        sector_hour_prod.reset_index(),
        x='sector',
        y='output_hour',
        title='Average Output per Hour by Sector',
        labels={'output_hour': 'Output per Hour', 'sector': 'Sector'}
    )
    st.plotly_chart(fig_hour, use_container_width=True)
    # 3. 4 Scatter plots with linear fit
    st.subheader("GDP vs Key Variables")
    fig_scatter, axes = plt.subplots(1, 4, figsize=(20, 5))
    # GDP vs Employment
    axes[0].scatter(filtered_data['gdp'], filtered_data['employment'], color='steelblue', alpha=0.6)
    z = np.polyfit(filtered_data['gdp'], filtered_data['employment'], 1)
    p = np.poly1d(z)
    axes[0].plot(filtered_data['gdp'], p(filtered_data['gdp']), color='darkred', linestyle='--')
    axes[0].set_xlabel('GDP')
    axes[0].set_ylabel('Employment')
    axes[0].set_title('GDP vs Employment')
    axes[0].grid(True, alpha=0.3)
    # GDP vs Total Working Hours
    axes[1].scatter(filtered_data['gdp'], filtered_data['hours'], color='green', alpha=0.6)
    z = np.polyfit(filtered_data['gdp'], filtered_data['hours'], 1)
    p = np.poly1d(z)
    axes[1].plot(filtered_data['gdp'], p(filtered_data['gdp']), color='darkred', linestyle='--')
    axes[1].set_xlabel('GDP')
    axes[1].set_ylabel('Total Working Hours')
    axes[1].set_title('GDP vs Total Working Hours')
    axes[1].grid(True, alpha=0.3)
    # GDP vs Output per Hour
    axes[2].scatter(filtered_data['gdp'], filtered_data['output_hour'], color='orange', alpha=0.6)
    axes[2].set_xlabel('GDP')
    axes[2].set_ylabel('Output per Hour')
    axes[2].set_title('GDP vs Output per Hour')
    axes[2].grid(True, alpha=0.3)
    # GDP vs Output per Employee
    axes[3].scatter(filtered_data['gdp'], filtered_data['output_employment'], color='purple', alpha=0.6)
    axes[3].set_xlabel('GDP')
    axes[3].set_ylabel('Output per Employee')
    axes[3].set_title('GDP vs Output per Employee')
    axes[3].grid(True, alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig_scatter)
    # 4. Correlation Heatmap
    st.subheader("Correlation Heatmap")
    key_vars = ['gdp', 'employment', 'hours', 'output_hour', 'output_employment']
    key_vars = [v for v in key_vars if v in filtered_data.columns]
    if len(key_vars) >= 2:
        corr = filtered_data[key_vars].corr()
        fig_heat, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="plasma", ax=ax)
        plt.title("Correlation Heatmap")
        st.pyplot(fig_heat)
        st.markdown("**Correlation Matrix (numeric values):**")
        st.dataframe(corr.style.format("{:.3f}"), use_container_width=True)
    else:
        st.info("Not enough variables for correlation heatmap.")
# ─────────────────────────────────────────────────────────────
# Tab 3: Model Performance
# ─────────────────────────────────────────────────────────────
with tab_model:
    st.subheader("Model Performance – Detailed Evaluation")
    df_metrics = model_results['metrics_df']
    styled = df_metrics.style.format({
        col: '{:.4f}' for col in df_metrics.columns if col != 'Model'
    }).highlight_max(
        subset=['Accuracy', 'F1-Score (macro)', 'Precision (macro)', 'Recall (macro)'],
        color='#d4edda'
    )
    st.dataframe(styled, use_container_width=True)
    melt = df_metrics.melt(id_vars='Model',
                           value_vars=['Accuracy', 'F1-Score (macro)', 'Precision (macro)', 'Recall (macro)'],
                           var_name='Metric', value_name='Score')
    fig_bar = px.bar(melt, x='Model', y='Score', color='Metric', barmode='group',
                     title="Model Comparison – Key Metrics")
    fig_bar.update_layout(yaxis_range=[0, 1.05])
    st.plotly_chart(fig_bar, use_container_width=True)
    st.markdown(f"### Classification Report – Best Model ({model_results['best_name']})")
    y_true = model_results['y_test']
    y_pred_best = model_results['predictions'][model_results['best_name']]
    report = classification_report(y_true, y_pred_best,
                                   target_names=model_results['encoder'].classes_,
                                   output_dict=True, zero_division=0)
    report_df = pd.DataFrame(report).transpose()
    st.dataframe(report_df.style.format('{:.4f}'), use_container_width=True)
# ─────────────────────────────────────────────────────────────
# Tab 4: Sector Prediction
# ─────────────────────────────────────────────────────────────
with tab_predict:
    st.subheader("Predict Dominant Sector")
    st.caption("Uses the best model from the Performance tab above")
    c1, c2, c3 = st.columns(3)
    gdp_val = c1.number_input("GDP (million RM)", 0.0, 500000.0, 50000.0)
    emp_val = c2.number_input("Total Employment", 1, 1000000, 100000)
    hours_val = c3.number_input("Total Working Hours", 1.0, 50000000.0, 2000000.0)
    if st.button("Predict"):
        feat_dict = {
            'gdp': gdp_val,
            'employment': emp_val,
            'hours': hours_val,
            'output_hour': gdp_val / hours_val if hours_val > 0 else 0,
            'output_employment': gdp_val / emp_val if emp_val > 0 else 0,
            'employed_employer': emp_val * 0.10,
            'employed_employee': emp_val * 0.70,
            'employed_own_account': emp_val * 0.15,
            'employed_unpaid_family': emp_val * 0.05
        }
        X_new = pd.DataFrame([feat_dict])
        avail_features = X_new.columns.intersection(model_results['scaler'].feature_names_in_)
        X_new_s = model_results['scaler'].transform(X_new[avail_features])
        pred_enc = model_results['best_model'].predict(X_new_s)[0]
        pred_sector = model_results['encoder'].inverse_transform([pred_enc])[0]
        try:
            prob = model_results['best_model'].predict_proba(X_new_s)[0].max() * 100
            st.success(f"**Predicted sector: {pred_sector}** ({prob:.1f}% confidence)")
        except:
            st.success(f"**Predicted sector: {pred_sector}**")
# ─────────────────────────────────────────────────────────────
# Tab 5: Data Table
# ─────────────────────────────────────────────────────────────
with tab_data:
    st.subheader("Filtered Data Table")
    st.caption(f"Showing data for {year_range[0]}–{year_range[1]} and selected sectors")
    st.dataframe(filtered_data.sort_values(['date', 'sector']), use_container_width=True)
    st.download_button(
        "Download filtered data (CSV)",
        filtered_data.to_csv(index=False).encode('utf-8'),
        f"malaysia_labour_{year_range[0]}-{year_range[1]}.csv",
        "text/csv"
    )
st.caption("Dashboard • Updated January 14, 2026")
