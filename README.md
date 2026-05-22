# AutoML Cloud Platform 🚀

Welcome to the **AutoML Cloud Platform**, a scalable, AI-powered SaaS web application designed to automate the complete machine learning lifecycle—from dataset ingestion and automated preprocessing to distributed AutoML training, explainability analysis (XAI), and instant model deployment as served REST APIs.

The platform is designed around enterprise-grade product-company principles: distributed task execution boundaries, separation of storage and compute, interactive playgrounds, and gorgeous responsive dashboards.

---

## 🎨 Premium Visual Dashboards

This project replicates the highly visual, premium glassmorphic UI cards, harmonic HSL palettes, and custom graphics shown in the product design mockups:
1. **Platform Dashboard (`/dashboard`)**: Displays core metrics, active threads, an interactive SVG convergence curve, compute cluster load meters (with auto-scaling thermal warning panels), and running experiments.
2. **Experiment Tracker (`/experiments`)**: Features metric sweeps comparing Accuracy vs. F1-scores across sweeps, an **AI Insights Diagnostic** panel, and hyperparameter grids.
3. **Explainability Hub (`/explainability`)**: Implements global feature attributions (with live SHAP vs. Permutation toggles), demographic fairness metrics (disparate impact monitoring across Gender, Age, and Location), a local prediction interactive **SHAP Force Plot**, surrogate **LIME scatterplots**, and continuous **Partial Dependence Plots (PDP)**.
4. **AutoML Ingestion & Preprocessing (`/datasets`)**: Drag-and-drop file uploader, schema type detector, missing value ratio charts, and automated cleaning status streams.
5. **REST Serving Playground (`/serving`)**: Served REST endpoints directory, instant serving controls, and a mock Swagger playground client for live testing.
6. **Landing Page (`/landing`)**: A deep space dark hero canvas showcasing the core value propositions and telemetry stats.

---

## 🏗️ System Architecture

```
                       [ User Web Browser ]
                                │
        ┌───────────────────────┴───────────────────────┐
        ▼ (Vite + React)                                ▼ (FastAPI served)
┌─────────────────────────┐                     ┌─────────────────────────┐
│   Landing / Dashboards  │                     │   Main FastAPI App      │
│   React Hooks & Context │                     │   WebSockets Broadcaster│
└─────────────────────────┘                     └─────────────────────────┘
                                                        │
                                                        ▼ (Services Layer)
                                                ┌─────────────────────────┐
                                                │ - Dataset Parser        │
                                                │ - Data Preprocessor     │
                                                │ - AutoML Training       │
                                                │ - Explainability / XAI  │
                                                │ - AI Insights Assistant │
                                                └─────────────────────────┘
                                                        │
                                ┌───────────────────────┴───────────────────────┐
                                ▼ (SQLite DB)                                   ▼ (.s3_mock/)
                        ┌───────────────────────┐                       ┌───────────────────────┐
                        │   Experiments, Users, │                       │   Uploaded Datasets   │
                        │   Models, Deployments │                       │   Trained Model PKLs  │
                        └───────────────────────┘                       └───────────────────────┘
```

---

## ⚙️ Technology Stack

- **Frontend**: Vite + React, styled using custom, premium **Vanilla CSS HSL design tokens** (Outfit & Inter Google Fonts) to maintain custom visual excellence and smooth layouts.
- **Backend**: FastAPI, Uvicorn, and Python.
- **Task Queue Simulation**: Built using FastAPI's asynchronous `BackgroundTasks` and Python `asyncio` queues, providing a zero-config, out-of-the-box distributed execution boundary (simulating Celery/Redis) with WebSockets progress streams.
- **Storage Layer Simulation**: Separates compute from files by reading/writing to a simulated S3 storage directory (`backend/s3_mock/`).
- **Database**: Simulated PostgreSQL utilizing a local SQLite schemas (`backend/automl.db`).

---

## 🚀 How to Run the App (Windows)

We have created a unified Windows startup batch script to launch the entire platform in a single double-click.

### Step 1: Launch the Launcher
Double-click the **`start.bat`** file located in the project root directory.

*Alternatively, run it via PowerShell or Command Prompt from the root directory:*
```cmd
start.bat
```

This will automatically:
1. Open a terminal window starting the **FastAPI Backend Server** at `http://127.0.0.1:8000`.
2. Open a terminal window starting the **Vite React Dev Server** at `http://localhost:5173`.
3. Open your default web browser to the application landing page.

---

## 🧪 Testing the AutoML Workflow

We have pre-generated a **`sample_data/credit_risk.csv`** file for you, containing features that align with the mockup views (Credit Score, Income, Debt-to-Income, etc.) to let you test the platform instantly:

1. **Upload Dataset**: Navigate to the **Training** tab, drag and drop `sample_data/credit_risk.csv` (or click to select), and watch the uploader parse its schema.
2. **Clean Dataset**: Click **Clean Dataset (Preprocess)** to trigger automated scaling, imputation, and outlier capping.
3. **Train Models**: Choose `approved` as your **Target Variable** and click **Run AutoML Pipeline**. Monitor the live logs streaming over WebSockets in the dark terminal widget.
4. **Interpret Explainability**: Switch to the **Deployments** tab (routes to Explainability Hub). Toggle between SHAP and Permutation importance, inspect demographic fairness parity, interact with the SHAP Force Plot values, and explore LIME/PDP curves.
5. **Serve REST API**: Switch to the **Security** tab (routes to Deployments). Select your trained model, input an endpoint name, and click **Deploy**.
6. **Query Predictions**: In the API Playground on the right, toggle your newly served POST endpoint, input numeric fields, click **Send Request**, and review the JSON response!
