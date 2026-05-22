try:
    import pandas as pd
    import numpy as np
    from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
except ImportError:
    from services.fallback_ml import pd, np
    from services.fallback_ml import FallbackStandardScaler as StandardScaler
    from services.fallback_ml import FallbackMinMaxScaler as MinMaxScaler
    from services.fallback_ml import FallbackLabelEncoder as LabelEncoder
from database import log_activity

def preprocess_dataframe(df, target_col=None, fill_missing="auto", handle_outliers="iqr", encode_categorical=True, scale_features="standard"):
    """
    Cleans and preprocesses a Pandas DataFrame.
    Returns: (cleaned_df, preprocessing_metadata)
    """
    cleaned_df = df.copy()
    metadata = {}
    
    # 1. Remove duplicate rows
    initial_rows = len(cleaned_df)
    cleaned_df.drop_duplicates(inplace=True)
    duplicates_removed = initial_rows - len(cleaned_df)
    metadata["duplicates_removed"] = duplicates_removed
    if duplicates_removed > 0:
        log_activity("INFO", f"Removed {duplicates_removed} duplicate rows during preprocessing.")

    # Identify numerical and categorical columns (excluding target column if provided)
    cols_to_process = [c for c in cleaned_df.columns if c != target_col]
    numerical_cols = []
    categorical_cols = []
    
    for c in cols_to_process:
        if cleaned_df[c].dtype == 'object' or cleaned_df[c].dtype == 'bool':
            categorical_cols.append(c)
        else:
            numerical_cols.append(c)
            
    metadata["numerical_columns"] = numerical_cols
    metadata["categorical_columns"] = categorical_cols

    # 2. Handle missing values
    imputed_count = 0
    for col in cleaned_df.columns:
        missing_count = int(cleaned_df[col].isnull().sum())
        if missing_count > 0:
            imputed_count += missing_count
            if col in numerical_cols or col == target_col:
                # Numerical Imputation (Mean)
                mean_val = cleaned_df[col].mean()
                if pd.isna(mean_val):
                    mean_val = 0.0
                cleaned_df[col].fillna(mean_val, inplace=True)
            else:
                # Categorical Imputation (Mode)
                if not cleaned_df[col].mode().empty:
                    mode_val = cleaned_df[col].mode().iloc[0]
                else:
                    mode_val = "UNKNOWN"
                cleaned_df[col].fillna(mode_val, inplace=True)
                
    metadata["imputed_missing_values"] = imputed_count
    if imputed_count > 0:
        log_activity("INFO", f"Imputed {imputed_count} missing cells.")

    # 3. Handle Outliers (IQR Method)
    outliers_detected = 0
    if handle_outliers == "iqr" and len(numerical_cols) > 0:
        for col in numerical_cols:
            Q1 = cleaned_df[col].quantile(0.25)
            Q3 = cleaned_df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            # Count outliers
            col_outliers = ((cleaned_df[col] < lower_bound) | (cleaned_df[col] > upper_bound)).sum()
            outliers_detected += int(col_outliers)
            
            # Clip outliers (cap and floor them) instead of dropping to preserve rows
            cleaned_df[col] = cleaned_df[col].clip(lower_bound, upper_bound)
            
    metadata["outliers_handled"] = outliers_detected
    if outliers_detected > 0:
        log_activity("INFO", f"Capped {outliers_detected} outliers using IQR bounds.")

    # 4. Encode Categorical variables (Label Encoding for simplicity and ML compatibility)
    encoded_cols = []
    if encode_categorical and len(categorical_cols) > 0:
        for col in categorical_cols:
            le = LabelEncoder()
            # Force conversion to string to prevent comparison errors
            cleaned_df[col] = cleaned_df[col].astype(str)
            cleaned_df[col] = le.fit_transform(cleaned_df[col])
            encoded_cols.append(col)
            
    metadata["encoded_columns"] = encoded_cols

    # 5. Scale features
    if scale_features in ["standard", "minmax"] and len(numerical_cols) > 0:
        scaler = StandardScaler() if scale_features == "standard" else MinMaxScaler()
        cleaned_df[numerical_cols] = scaler.fit_transform(cleaned_df[numerical_cols])
        metadata["scaled_features"] = scale_features
        metadata["scaler_type"] = scale_features
        
    log_activity("INFO", f"Data preprocessing finished. Outputs: Rows={len(cleaned_df)}, Cols={len(cleaned_df.columns)}")
    return cleaned_df, metadata
