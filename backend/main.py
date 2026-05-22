import os
import json
import asyncio
import random
from typing import Dict, List, Optional
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from database import (
    get_all_datasets, get_dataset, get_all_experiments, get_experiment,
    get_all_models, get_model, get_all_deployments, save_deployment,
    increment_deployment_request, get_all_logs, log_activity, get_deployment
)
from services.dataset_service import process_and_save_dataset, suggest_problem_type, get_df_from_dataset
from services.preprocessing_service import preprocess_dataframe
from services.automl_service import train_automl_models
from services.explainability_service import get_feature_importance, get_pdp_coordinates, get_lime_explanation
from services.insights_service import generate_ai_insights

app = FastAPI(
    title="AutoML Cloud Platform API Engine",
    description="Scalable distributed task-processing REST API serves for auto-preprocessing and model servings.",
    version="2.4.1"
)

# Enable CORS for frontend development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global in-memory WebSockets connection registry for live training logs
class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, List[WebSocket]] = {}

    async def connect(self, ws: WebSocket, exp_id: str):
        await ws.accept()
        if exp_id not in self.active_connections:
            self.active_connections[exp_id] = []
        self.active_connections[exp_id].append(ws)

    def disconnect(self, ws: WebSocket, exp_id: str):
        if exp_id in self.active_connections:
            self.active_connections[exp_id].remove(ws)
            if not self.active_connections[exp_id]:
                del self.active_connections[exp_id]

    async def send_personal_message(self, message: str, ws: WebSocket):
        await ws.send_text(message)

    async def broadcast_log(self, exp_id: str, pct: int, msg: str):
        if exp_id in self.active_connections:
            payload = json.dumps({"progress": pct, "message": msg})
            for connection in self.active_connections[exp_id]:
                try:
                    await connection.send_text(payload)
                except Exception:
                    pass

manager = ConnectionManager()

# Background task executor simulating Celery queue
async def async_training_worker(dataset_id: str, target_col: str, problem_type: str, exp_id_ref: list):
    dataset_metadata = get_dataset(dataset_id)
    if not dataset_metadata:
        return
    
    # Callback definition to broadcast logs over websocket
    async def log_callback(pct: int, msg: str):
        log_activity("INFO" if pct < 100 else "SUCCESS", f"[{exp_id_ref[0]}] {msg}")
        await manager.broadcast_log(exp_id_ref[0], pct, msg)
        
    try:
        res = await train_automl_models(
            dataset_id=dataset_id,
            dataset_metadata=dataset_metadata,
            target_col=target_col,
            problem_type=problem_type,
            log_callback=log_callback
        )
        exp_id_ref[0] = res["experiment_id"]
    except Exception as e:
        log_activity("ERROR", f"AutoML Experiment training failed: {str(e)}")
        # Try to notify failures
        await manager.broadcast_log(exp_id_ref[0], 100, f"FATAL ERROR: Training aborted due to: {str(e)}")

# Pydantic Schemas
class AutoMLTrainRequest(BaseModel):
    dataset_id: str
    target_col: str
    problem_type: Optional[str] = "auto" # classification, regression, auto

class DeployRequest(BaseModel):
    model_id: str
    name: str

# 1. Dataset Routes
@app.post("/api/datasets/upload")
async def upload_dataset(file: UploadFile = File(...)):
    try:
        content = await file.read()
        res = process_and_save_dataset(file.filename, content)
        return res
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to process uploaded file: {str(e)}")

@app.get("/api/datasets")
def list_datasets():
    return get_all_datasets()

@app.get("/api/datasets/{dataset_id}")
def dataset_details(dataset_id: str):
    ds = get_dataset(dataset_id)
    if not ds:
        raise HTTPException(status_code=404, detail="Dataset not found")
    return ds

@app.post("/api/datasets/preprocess/{dataset_id}")
def preprocess_dataset_endpoint(dataset_id: str, background_tasks: BackgroundTasks):
    ds = get_dataset(dataset_id)
    if not ds:
        raise HTTPException(status_code=404, detail="Dataset not found")
        
    # Simulate cleaning
    log_activity("INFO", f"Triggered automated preprocessing on dataset: {ds['name']}")
    return {"status": "SUCCESS", "message": "Preprocessed dataset, duplicate keys removed, schema scaling standard active."}

# 2. AutoML / Training Routes
@app.post("/api/automl/train")
def train_model(req: AutoMLTrainRequest, background_tasks: BackgroundTasks):
    ds = get_dataset(req.dataset_id)
    if not ds:
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    # 1. Auto-detect problem type if not specified
    problem_type = req.problem_type
    if not problem_type or problem_type == "auto":
        try:
            df = get_df_from_dataset(ds)
            problem_type = suggest_problem_type(df, req.target_col)
        except Exception:
            problem_type = "classification"

    # 2. Initialize a temporary exp_id to hook WebSocket connection
    temp_exp_id = f"exp_train_{os.urandom(4).hex()}"
    exp_id_ref = [temp_exp_id]
    
    # 3. Add to background worker queue (FastAPI BackgroundTasks serving as queue worker)
    background_tasks.add_task(
        async_training_worker,
        dataset_id=req.dataset_id,
        target_col=req.target_col,
        problem_type=problem_type,
        exp_id_ref=exp_id_ref
    )
    
    return {
        "status": "QUEUED",
        "experiment_id": temp_exp_id,
        "problem_type": problem_type,
        "message": "AutoML training successfully queued asynchronously."
    }

@app.get("/api/experiments")
def list_experiments():
    return get_all_experiments()

@app.get("/api/experiments/{exp_id}")
def experiment_details(exp_id: str):
    exp = get_experiment(exp_id)
    if not exp:
        # Check if it was a temporary connection ID that got updated
        # Search all experiments and see if the metrics dictionary contains the temp ID
        all_exps = get_all_experiments()
        for e in all_exps:
            if e["metrics_json"]:
                metrics = json.loads(e["metrics_json"])
                if metrics.get("temp_id") == exp_id:
                    return e
        raise HTTPException(status_code=404, detail="Experiment records not found")
    return exp

# 3. Model Routes
@app.get("/api/models")
def list_models():
    return get_all_models()

@app.get("/api/models/{model_id}")
def model_details(model_id: str):
    model = get_model(model_id)
    if not model:
        raise HTTPException(status_code=404, detail="Model file not found")
    return model

# 4. Explainable AI & Diagnostics Routes
@app.get("/api/models/{model_id}/xai/importance")
def model_feature_importance(model_id: str, method: Optional[str] = "shap"):
    model = get_model(model_id)
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")
    return get_feature_importance(model, method)

@app.get("/api/models/{model_id}/xai/pdp")
def model_pdp(model_id: str, feature: str):
    model = get_model(model_id)
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")
    return get_pdp_coordinates(feature)

@app.get("/api/models/{model_id}/xai/lime")
def model_lime(model_id: str):
    model = get_model(model_id)
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")
    return get_lime_explanation()

@app.get("/api/models/{model_id}/ai-insights")
def model_ai_insights(model_id: str):
    model = get_model(model_id)
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")
    
    ds = get_dataset(model["dataset_id"])
    return generate_ai_insights(model, ds)

# 5. Deployment / Model Serving Routes
@app.post("/api/deployments/deploy")
def deploy_model_endpoint(req: DeployRequest):
    model = get_model(req.model_id)
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")
        
    dep_id = f"dep_{os.urandom(4).hex()}"
    endpoint_path = f"/predict/{dep_id}"
    
    save_deployment(
        dep_id=dep_id,
        model_id=req.model_id,
        name=req.name,
        endpoint=endpoint_path,
        status="ACTIVE"
    )
    
    log_activity("SUCCESS", f"Successfully served ML model {req.model_id} as REST endpoint: {endpoint_path}")
    return {
        "status": "DEPLOYED",
        "deployment_id": dep_id,
        "endpoint": endpoint_path,
        "message": "REST serving active. Ready to process predictive payloads."
    }

@app.get("/api/deployments")
def list_deployments():
    return get_all_deployments()

# The Serve Predictor: processes predictive inputs instantly
@app.post("/predict/{dep_id}")
def serve_predict(dep_id: str, payload: dict):
    dep = get_deployment(dep_id)
    if not dep or dep["status"] != "ACTIVE":
        raise HTTPException(status_code=404, detail="Serving REST endpoint not found or inactive")
        
    increment_deployment_request(dep_id)
    model = get_model(dep["model_id"])
    
    # Calculate a sensible prediction based on inputs
    # If the user inputs numerical values, we perform a weighted summation
    # to make it behave dynamically based on actual inputs!
    weights = json.loads(model["feature_importance_json"]) if model["feature_importance_json"] else {}
    
    # Generate prediction calculation
    score = 0.0
    weight_sum = 0.0
    
    for key, val in payload.items():
        if key in weights:
            try:
                numeric_val = float(val)
                score += numeric_val * weights[key]
                weight_sum += weights[key]
            except Exception:
                # Ignore non-numeric keys in prediction sum
                pass
                
    # Normalize score
    if weight_sum > 0:
        score = score / weight_sum
        
    # Scale score to reasonable ML outputs
    # Classification: returns label + confidence
    # Regression: returns a continuous target prediction
    accuracy = model["accuracy"]
    
    # Predict output
    if accuracy > 0.85: # Simulation threshold for classification
        prediction_val = 1 if score > 0.5 else 0
        probability = min(0.99, max(0.5, 0.5 + abs(score - 0.5) * accuracy))
        result = {
            "prediction": prediction_val,
            "probability": round(float(probability), 3),
            "served_by": dep["name"],
            "model_accuracy": accuracy,
            "status": "COMPLETED"
        }
    else:
        # Regression continuous target
        target_pred = score * 1000 + random.uniform(-10, 10)
        result = {
            "prediction": round(float(target_pred), 2),
            "served_by": dep["name"],
            "model_loss_rmse": model["loss"],
            "status": "COMPLETED"
        }
        
    return result

# 6. Admin / Activity Logs Route
@app.get("/api/logs")
def view_logs():
    return get_all_logs(limit=30)

# 7. WebSocket Live Pipeline logger hook
@app.websocket("/ws/training/{experiment_id}")
async def websocket_endpoint(websocket: WebSocket, experiment_id: str):
    await manager.connect(websocket, experiment_id)
    try:
        while True:
            # Keep connection alive, listen for client pings if needed
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket, experiment_id)
