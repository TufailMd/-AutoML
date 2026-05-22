try:
    import pandas as pd
    import numpy as np
except ImportError:
    from services.fallback_ml import pd, np
# import pd, np
import os
import json
import uuid
import math
# from ..database import save_dataset, log_activity
from database import save_dataset, log_activity

# Create upload directory inside backend (mock S3 storage)
UPLOAD_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "s3_mock", "datasets")
os.makedirs(UPLOAD_DIR, exist_ok=True)

def process_and_save_dataset(file_name, file_content_bytes):
    # 1. Generate unique file ID
    dataset_id = f"ds_{uuid.uuid4().hex[:8]}"
    file_extension = os.path.splitext(file_name)[1].lower()
    
    saved_file_name = f"{dataset_id}{file_extension}"
    saved_file_path = os.path.join(UPLOAD_DIR, saved_file_name)
    
    # 2. Write file bytes
    with open(saved_file_path, "wb") as f:
        f.write(file_content_bytes)
        
    # 3. Read dataset with Pandas
    try:
        if file_extension == '.csv':
            df = pd.read_csv(saved_file_path)
        elif file_extension in ['.xlsx', '.xls']:
            df = pd.read_excel(saved_file_path)
        elif file_extension == '.json':
            df = pd.read_json(saved_file_path)
        else:
            raise ValueError(f"Unsupported file type: {file_extension}")
    except Exception as e:
        # Clean up file on failure
        if os.path.exists(saved_file_path):
            os.remove(saved_file_path)
        log_activity("ERROR", f"Failed to parse dataset {file_name}: {str(e)}")
        raise e

    # 4. Analyze Schema & Columns
    row_count = len(df)
    col_count = len(df.columns)
    size_bytes = len(file_content_bytes)
    
    columns_list = []
    summary = {}
    
    for col in df.columns:
        col_data = df[col]
        missing_count = int(col_data.isnull().sum())
        missing_percentage = float(missing_count / row_count) if row_count > 0 else 0.0
        
        # Schema Detection
        unique_count = int(col_data.nunique())
        dtype = str(col_data.dtype)
        
        if col_data.dtype == 'object' or col_data.dtype == 'bool' or (unique_count < 10 and np.issubdtype(col_data.dtype, np.integer)):
            inferred_type = 'categorical'
        else:
            inferred_type = 'numerical'
            
        columns_list.append({
            "name": col,
            "type": inferred_type,
            "dtype": dtype,
            "unique_count": unique_count,
            "missing_count": missing_count,
            "missing_pct": missing_percentage
        })
        
        # Stats summary
        stats = {
            "missing_pct": missing_percentage,
            "unique_count": unique_count
        }
        
        if inferred_type == 'numerical':
            stats["min"] = float(col_data.min()) if not pd.isna(col_data.min()) else None
            stats["max"] = float(col_data.max()) if not pd.isna(col_data.max()) else None
            stats["mean"] = float(col_data.mean()) if not pd.isna(col_data.mean()) else None
            stats["std"] = float(col_data.std()) if not pd.isna(col_data.std()) else None
        else:
            # Categorical stats
            top_val = col_data.mode().iloc[0] if not col_data.mode().empty else None
            stats["top_value"] = str(top_val) if not pd.isna(top_val) else None
            
        summary[col] = stats

    # 5. Add a 10-row preview
    preview_df = df.head(10).replace({np.nan: None})
    preview_data = preview_df.to_dict(orient="records")
    
    summary["__preview"] = preview_data

    # 6. Save metadata in DB
    save_dataset(
        dataset_id=dataset_id,
        name=file_name,
        file_path=saved_file_path,
        size_bytes=size_bytes,
        row_count=row_count,
        col_count=col_count,
        columns_list=columns_list,
        summary=summary
    )
    
    log_activity("INFO", f"Dataset {file_name} uploaded successfully as ID: {dataset_id}")
    return {
        "id": dataset_id,
        "name": file_name,
        "row_count": row_count,
        "col_count": col_count,
        "columns": columns_list,
        "size_bytes": size_bytes,
        "preview": preview_data
    }

def get_df_from_dataset(dataset_metadata):
    file_path = dataset_metadata["file_path"]
    file_extension = os.path.splitext(file_path)[1].lower()
    if file_extension == '.csv':
        return pd.read_csv(file_path)
    elif file_extension in ['.xlsx', '.xls']:
        return pd.read_excel(file_path)
    elif file_extension == '.json':
        return pd.read_json(file_path)
    raise ValueError(f"Unsupported file format in saved metadata path: {file_path}")

def suggest_problem_type(df, target_col):
    if target_col not in df.columns:
        return "classification"
    unique_vals = df[target_col].nunique()
    target_dtype = df[target_col].dtype
    
    if target_dtype == 'object' or target_dtype == 'bool' or unique_vals < 15:
        return "classification"
    return "regression"
