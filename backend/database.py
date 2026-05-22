import sqlite3
import json
import os

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "automl.db")

def get_db_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # 1. Datasets Table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS datasets (
        id TEXT PRIMARY KEY,
        name TEXT NOT NULL,
        file_path TEXT NOT NULL,
        size_bytes INTEGER,
        row_count INTEGER,
        col_count INTEGER,
        columns_json TEXT,
        summary_json TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """)
    
    # 2. Experiments Table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS experiments (
        id TEXT PRIMARY KEY,
        dataset_id TEXT NOT NULL,
        name TEXT NOT NULL,
        status TEXT NOT NULL, -- 'PENDING', 'RUNNING', 'COMPLETED', 'FAILED'
        algorithm TEXT,
        hyperparams_json TEXT,
        metrics_json TEXT,
        duration TEXT,
        accuracy REAL,
        f1_score REAL,
        loss REAL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY(dataset_id) REFERENCES datasets(id)
    )
    """)
    
    # 3. Models Table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS models (
        id TEXT PRIMARY KEY,
        experiment_id TEXT NOT NULL,
        dataset_id TEXT NOT NULL,
        name TEXT NOT NULL,
        file_path TEXT NOT NULL,
        accuracy REAL,
        f1_score REAL,
        precision REAL,
        recall REAL,
        loss REAL,
        feature_importance_json TEXT,
        shap_values_json TEXT,
        fairness_json TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY(experiment_id) REFERENCES experiments(id),
        FOREIGN KEY(dataset_id) REFERENCES datasets(id)
    )
    """)
    
    # 4. Deployments Table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS deployments (
        id TEXT PRIMARY KEY,
        model_id TEXT NOT NULL,
        name TEXT NOT NULL,
        endpoint TEXT NOT NULL,
        status TEXT NOT NULL, -- 'ACTIVE', 'INACTIVE'
        request_count INTEGER DEFAULT 0,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY(model_id) REFERENCES models(id)
    )
    """)
    
    # 5. Logs Table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS logs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        level TEXT,
        message TEXT,
        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """)
    
    conn.commit()
    conn.close()

# Helper DB actions
def save_dataset(dataset_id, name, file_path, size_bytes, row_count, col_count, columns_list, summary):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO datasets (id, name, file_path, size_bytes, row_count, col_count, columns_json, summary_json) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (dataset_id, name, file_path, size_bytes, row_count, col_count, json.dumps(columns_list), json.dumps(summary))
    )
    conn.commit()
    conn.close()

def get_all_datasets():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM datasets ORDER BY created_at DESC")
    rows = cursor.fetchall()
    conn.close()
    return [dict(row) for row in rows]

def get_dataset(dataset_id):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM datasets WHERE id = ?", (dataset_id,))
    row = cursor.fetchone()
    conn.close()
    return dict(row) if row else None

def save_experiment(exp_id, dataset_id, name, status, algorithm=None, hyperparams=None, metrics=None, duration=None, accuracy=None, f1_score=None, loss=None):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO experiments (id, dataset_id, name, status, algorithm, hyperparams_json, metrics_json, duration, accuracy, f1_score, loss) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (exp_id, dataset_id, name, status, algorithm, json.dumps(hyperparams) if hyperparams else None, json.dumps(metrics) if metrics else None, duration, accuracy, f1_score, loss)
    )
    conn.commit()
    conn.close()

def update_experiment(exp_id, status, metrics=None, duration=None, accuracy=None, f1_score=None, loss=None):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        "UPDATE experiments SET status = ?, metrics_json = ?, duration = ?, accuracy = ?, f1_score = ?, loss = ? WHERE id = ?",
        (status, json.dumps(metrics) if metrics else None, duration, accuracy, f1_score, loss, exp_id)
    )
    conn.commit()
    conn.close()

def get_all_experiments():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM experiments ORDER BY created_at DESC")
    rows = cursor.fetchall()
    conn.close()
    return [dict(row) for row in rows]

def get_experiment(exp_id):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM experiments WHERE id = ?", (exp_id,))
    row = cursor.fetchone()
    conn.close()
    return dict(row) if row else None

def save_model(model_id, exp_id, dataset_id, name, file_path, accuracy, f1_score, precision, recall, loss, feature_importance, shap_values, fairness):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO models (id, experiment_id, dataset_id, name, file_path, accuracy, f1_score, precision, recall, loss, feature_importance_json, shap_values_json, fairness_json) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (model_id, exp_id, dataset_id, name, file_path, accuracy, f1_score, precision, recall, loss, json.dumps(feature_importance), json.dumps(shap_values), json.dumps(fairness))
    )
    conn.commit()
    conn.close()

def get_all_models():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM models ORDER BY created_at DESC")
    rows = cursor.fetchall()
    conn.close()
    return [dict(row) for row in rows]

def get_model(model_id):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM models WHERE id = ?", (model_id,))
    row = cursor.fetchone()
    conn.close()
    return dict(row) if row else None

def save_deployment(dep_id, model_id, name, endpoint, status):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO deployments (id, model_id, name, endpoint, status) VALUES (?, ?, ?, ?, ?)",
        (dep_id, model_id, name, endpoint, status)
    )
    conn.commit()
    conn.close()

def get_all_deployments():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM deployments ORDER BY created_at DESC")
    rows = cursor.fetchall()
    conn.close()
    return [dict(row) for row in rows]

def get_deployment(dep_id):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM deployments WHERE id = ?", (dep_id,))
    row = cursor.fetchone()
    conn.close()
    return dict(row) if row else None

def increment_deployment_request(dep_id):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("UPDATE deployments SET request_count = request_count + 1 WHERE id = ?", (dep_id,))
    conn.commit()
    conn.close()

def log_activity(level, message):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("INSERT INTO logs (level, message) VALUES (?, ?)", (level, message))
    conn.commit()
    conn.close()

def get_all_logs(limit=50):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM logs ORDER BY timestamp DESC LIMIT ?", (limit,))
    rows = cursor.fetchall()
    conn.close()
    return [dict(row) for row in rows]

# Initialize tables
init_db()
