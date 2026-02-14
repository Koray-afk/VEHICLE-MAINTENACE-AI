import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
import os

def engineer_features(df):
    df['Last_Service_Date'] = pd.to_datetime(df['Last_Service_Date'])
    df['Warranty_Expiry_Date'] = pd.to_datetime(df['Warranty_Expiry_Date'])
    
    ref_date = df['Last_Service_Date'].max() 
    
    df['Days_Since_Service'] = (ref_date - df['Last_Service_Date']).dt.days
    df['Warranty_Remaining'] = (df['Warranty_Expiry_Date'] - ref_date).dt.days
    
    return df.drop(['Last_Service_Date', 'Warranty_Expiry_Date'], axis=1)

def preprocess_pipeline(input_path, output_dir):
    df = pd.read_csv(input_path)
    df = engineer_features(df)
    
    X = df.drop('Need_Maintenance', axis=1)
    y = df['Need_Maintenance']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 2. HANDLE MISSING VALUES (Fit on train, transform on test)
    num_cols = X_train.select_dtypes(include=[np.number]).columns
    for col in num_cols:
        median_val = X_train[col].median()
        X_train[col] = X_train[col].fillna(median_val)
        X_test[col] = X_test[col].fillna(median_val)

    # 3. ROBUST ENCODING
    cat_cols = X_train.select_dtypes(include=['object', 'string']).columns
    encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
    X_train[cat_cols] = encoder.fit_transform(X_train[cat_cols].fillna('Unknown'))
    X_test[cat_cols] = encoder.transform(X_test[cat_cols].fillna('Unknown'))

    # 4. SCALING
    scaler = MinMaxScaler()
    X_train[X_train.columns] = scaler.fit_transform(X_train)
    X_test[X_test.columns] = scaler.transform(X_test)
    # processed data
    os.makedirs(output_dir, exist_ok=True)
    
    X_train.to_csv(os.path.join(output_dir, 'X_train.csv'), index=False)
    X_test.to_csv(os.path.join(output_dir, 'X_test.csv'), index=False)
    y_train.to_csv(os.path.join(output_dir, 'y_train.csv'), index=False)
    y_test.to_csv(os.path.join(output_dir, 'y_test.csv'), index=False)
    
    print(f"\nProcessed data saved to {output_dir}/")
    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    input_path = os.path.join(base_dir, "data", "raw", "vehicle_maintenance_data.csv")
    output_dir = os.path.join(base_dir, "data", "processed")
    
    X_train, X_test, y_train, y_test = preprocess_pipeline(input_path, output_dir)
