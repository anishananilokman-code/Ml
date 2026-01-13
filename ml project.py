import pandas as pd
import numpy as np

# Load your original / uncleaned data (the one with all years)
df_raw = pd.read_csv('your_original_data.csv')  # CHANGE THIS NAME

# Parse date correctly
df_raw['date'] = pd.to_datetime(df_raw['date'], errors='coerce')

# Drop bad dates
df_raw = df_raw.dropna(subset=['date'])

# Remove percentage rows (employment == 100 or very low)
df_raw = df_raw[df_raw['employment'] > 500].copy()

# Remove exact duplicates
df_raw = df_raw.drop_duplicates()

# If you have multiple rows per sector-date, keep the most complete one
df_raw = df_raw.sort_values('date').groupby(['date', 'sector']).first().reset_index()

# Optional: recalculate percentages if they are missing
for col in ['employed_employee', 'employed_employer', 'employed_own_account', 'employed_unpaid_family']:
    if col in df_raw.columns:
        df_raw[f'{col}_percentage'] = (df_raw[col] / df_raw['employment'] * 100).round(1)

# Save the FIXED file
df_raw.to_csv('clean_data_fixed.csv', index=False)

print("Fixed file saved.")
print("Years now present:")
print(df_raw['date'].dt.year.value_counts().sort_index())
print("\nUnique dates:")
print(sorted(df_raw['date'].unique()))
