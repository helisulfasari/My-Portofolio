import pandas as pd

missing_list = [
    '', ' ', 'NA', 'na', 'N/A', 'Unknown', 'unknown', 'UNKNOWN',
    'UNK', 'Null', 'null', 'None', 'Nan', 'NAN', 'NaN']

def preprocess_telco(df):

    # 1. Drop ID
    if 'customerID' in df.columns:
        df = df.drop('customerID', axis=1)

    # 2. Replace custom missing values
    df = df.replace(missing_list, pd.NA)

    # 3. Handle missing values
    fill_mode_cols = [
        'Gender', 'SeniorCitizen', 'Partner', 'Dependents',
        'Contract', 'PaperlessBilling', 'PaymentMethod'
    ]

    for col in fill_mode_cols:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].mode()[0])

    # 4. Numeric columns
    if 'Tenure' in df.columns:
        df['Tenure'] = df['Tenure'].astype(float).fillna(df['Tenure'].mean())

    if 'MonthlyCharges' in df.columns:
        df['MonthlyCharges'] = df['MonthlyCharges'].astype(float).fillna(df['MonthlyCharges'].mean())

    # 5. Convert TotalCharges
    if 'TotalCharges' in df.columns:
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
        df['TotalCharges'] = df['TotalCharges'].fillna(df['TotalCharges'].mean())

    return df
