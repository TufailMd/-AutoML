import React, { useState, useEffect, useRef } from 'react';
import './App.css';
import { API_BASE, WS_BASE } from './config';

// SVG Icons
const Icons = {
  Overview: () => (
    <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={2} stroke="currentColor">
      <path strokeLinecap="round" strokeLinejoin="round" d="M3.75 6A2.25 2.25 0 0 1 6 3.75h2.25A2.25 2.25 0 0 1 10.5 6v2.25a2.25 2.25 0 0 1-2.25 2.25H6a2.25 2.25 0 0 1-2.25-2.25V6ZM3.75 15.75A2.25 2.25 0 0 1 6 13.5h2.25a2.25 2.25 0 0 1 2.25 2.25V18a2.25 2.25 0 0 1-2.25 2.25H6A2.25 2.25 0 0 1 3.75 18v-2.25ZM13.5 6a2.25 2.25 0 0 1 2.25-2.25H18A2.25 2.25 0 0 1 20.25 6v2.25A2.25 2.25 0 0 1 18 10.5h-2.25a2.25 2.25 0 0 1-2.25-2.25V6ZM13.5 15.75a2.25 2.25 0 0 1 2.25-2.25H18a2.25 2.25 0 0 1 2.25 2.25V18A2.25 2.25 0 0 1 18 20.25h-2.25A2.25 2.25 0 0 1 13.5 18v-2.25Z" />
    </svg>
  ),
  Metrics: () => (
    <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={2} stroke="currentColor">
      <path strokeLinecap="round" strokeLinejoin="round" d="M2.25 18 9 11.25l4.306 4.306a11.95 11.95 0 0 1 5.814-5.518l2.74-1.22m0 0-5.94-2.281m5.94 2.28-2.28 5.941" />
    </svg>
  ),
  Training: () => (
    <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={2} stroke="currentColor">
      <path strokeLinecap="round" strokeLinejoin="round" d="M15.59 14.37a6 6 0 0 1-5.84 7.38v-4.8m5.84-2.58a14.98 14.98 0 0 0 6.16-12.12A14.98 14.98 0 0 0 9.61 8.41m5.98 5.96a14.96 14.96 0 0 1-5.98-5.96m0 0t-2.58 5.84v4.8" />
    </svg>
  ),
  Explainability: () => (
    <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={2} stroke="currentColor">
      <path strokeLinecap="round" strokeLinejoin="round" d="M2.036 12.322a1.012 1.012 0 0 1 0-.639C3.423 7.51 7.36 4.5 12 4.5c4.638 0 8.573 3.007 9.963 7.178.07.207.07.431 0 .639C20.577 16.49 16.64 19.5 12 19.5c-4.638 0-8.573-3.007-9.963-7.178Z" />
      <path strokeLinecap="round" strokeLinejoin="round" d="M15 12a3 3 0 1 1-6 0 3 3 0 0 1 6 0Z" />
    </svg>
  ),
  Deployments: () => (
    <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={2} stroke="currentColor">
      <path strokeLinecap="round" strokeLinejoin="round" d="M12 9v6m3-3H9m12 0a9 9 0 1 1-18 0 9 9 0 0 1 18 0Z" />
    </svg>
  ),
  Security: () => (
    <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={2} stroke="currentColor">
      <path strokeLinecap="round" strokeLinejoin="round" d="M9 12.75 11.25 15 15 9.75m-3-7.036A11.959 11.959 0 0 1 3.598 6 11.99 11.99 0 0 0 3 9.749c0 5.592 3.824 10.29 9 11.622 5.176-1.332 9-6.03 9-11.622 0-1.31-.21-2.571-.598-3.751h-.152c-3.196 0-6.1-1.248-8.25-3.285Z" />
    </svg>
  )
};

const BASE_URL = 'http://127.0.0.1:8000';

function App() {
  const [activeTab, setActiveTab] = useState('landing');
  const [datasets, setDatasets] = useState([]);
  const [experiments, setExperiments] = useState([]);
  const [models, setModels] = useState([]);
  const [deployments, setDeployments] = useState([]);
  
  // Selected targets
  const [selectedDataset, setSelectedDataset] = useState(null);
  const [targetCol, setTargetCol] = useState('');
  const [selectedModel, setSelectedModel] = useState(null);
  
  // AutoML Queue Logs
  const [trainingLogs, setTrainingLogs] = useState([]);
  const [trainingProgress, setTrainingProgress] = useState(0);
  const [isTraining, setIsTraining] = useState(false);
  const [activeExpId, setActiveExpId] = useState(null);
  
  // XAI Substates
  const [shapMethod, setShapMethod] = useState('shap');
  const [featureImportance, setFeatureImportance] = useState([]);
  const [limeData, setLimeData] = useState(null);
  const [pdpCoords, setPdpCoords] = useState(null);
  const [pdpFeature, setPdpFeature] = useState('');
  const [aiInsights, setAiInsights] = useState(null);
  
  // Serving Test Client State
  const [servedEndpoint, setServedEndpoint] = useState('');
  const [servedInputs, setServedInputs] = useState({});
  const [servedOutput, setServedOutput] = useState(null);
  const [deploymentName, setDeploymentName] = useState('');
  
  // Preprocessor simulator
  const [cleaningStatus, setCleaningStatus] = useState('');

  // Fetch initial records
  useEffect(() => {
    fetchDatasets();
    fetchExperiments();
    fetchModels();
    fetchDeployments();
  }, []);

  // Fetch specific XAI details when selectedModel changes
  useEffect(() => {
    if (selectedModel) {
      fetchFeatureImportance(selectedModel.id, shapMethod);
      fetchLimeData(selectedModel.id);
      fetchAiInsights(selectedModel.id);
      
      // Auto pick first feature for PDP
      try {
        const f_imp = JSON.parse(selectedModel.feature_importance_json);
        const first_f = Object.keys(f_imp)[0];
        if (first_f) {
          setPdpFeature(first_f);
          fetchPdpCoords(selectedModel.id, first_f);
        }
      } catch (e) {
        setPdpFeature('user_credit_score');
        fetchPdpCoords(selectedModel.id, 'user_credit_score');
      }
    }
  }, [selectedModel]);

  // Update feature importance on method toggle (SHAP vs Permutation)
  useEffect(() => {
    if (selectedModel) {
      fetchFeatureImportance(selectedModel.id, shapMethod);
    }
  }, [shapMethod]);

  // Core Data Fetchers
  const fetchDatasets = async () => {
    try {
      const res = await fetch(`${BASE_URL}/api/datasets`);
      const data = await res.json();
      setDatasets(data);
      if (data.length > 0 && !selectedDataset) {
        setSelectedDataset(data[0]);
      }
    } catch (e) {
      console.error("Error loading datasets", e);
    }
  };

  const fetchExperiments = async () => {
    try {
      const res = await fetch(`${BASE_URL}/api/experiments`);
      const data = await res.json();
      setExperiments(data);
    } catch (e) {
      console.error("Error loading experiments", e);
    }
  };

  const fetchModels = async () => {
    try {
      const res = await fetch(`${BASE_URL}/api/models`);
      const data = await res.json();
      setModels(data);
      if (data.length > 0 && !selectedModel) {
        setSelectedModel(data[0]);
      }
    } catch (e) {
      console.error("Error loading models", e);
    }
  };

  const fetchDeployments = async () => {
    try {
      const res = await fetch(`${BASE_URL}/api/deployments`);
      const data = await res.json();
      setDeployments(data);
      if (data.length > 0) {
        setServedEndpoint(data[0].endpoint);
      }
    } catch (e) {
      console.error("Error loading deployments", e);
    }
  };

  const fetchFeatureImportance = async (modelId, method) => {
    try {
      const res = await fetch(`${BASE_URL}/api/models/${modelId}/xai/importance?method=${method}`);
      const data = await res.json();
      setFeatureImportance(data);
    } catch (e) {
      console.error(e);
    }
  };

  const fetchLimeData = async (modelId) => {
    try {
      const res = await fetch(`${BASE_URL}/api/models/${modelId}/xai/lime`);
      const data = await res.json();
      setLimeData(data);
    } catch (e) {
      console.error(e);
    }
  };

  const fetchPdpCoords = async (modelId, feature) => {
    try {
      const res = await fetch(`${BASE_URL}/api/models/${modelId}/xai/pdp?feature=${feature}`);
      const data = await res.json();
      setPdpCoords(data);
    } catch (e) {
      console.error(e);
    }
  };

  const fetchAiInsights = async (modelId) => {
    try {
      const res = await fetch(`${BASE_URL}/api/models/${modelId}/ai-insights`);
      const data = await res.json();
      setAiInsights(data);
    } catch (e) {
      console.error(e);
    }
  };

  // Actions
  const handleFileUpload = async (e) => {
    const file = e.target.files[0];
    if (!file) return;

    setCleaningStatus('Uploading dataset to S3 cluster...');
    const formData = new FormData();
    formData.append('file', file);

    try {
      const res = await fetch(`${BASE_URL}/api/datasets/upload`, {
        method: 'POST',
        body: formData
      });
      const data = await res.json();
      setCleaningStatus('Auto-Schema detect optimal...');
      setTimeout(() => {
        setCleaningStatus('');
        fetchDatasets();
        setSelectedDataset(data);
        if (data.columns && data.columns.length > 0) {
          setTargetCol(data.columns[data.columns.length - 1].name);
        }
      }, 800);
    } catch (err) {
      setCleaningStatus('Upload failed. Try again.');
    }
  };

  const triggerPreprocessing = async () => {
    if (!selectedDataset) return;
    setCleaningStatus('Removing duplicate rows...');
    try {
      await fetch(`${BASE_URL}/api/datasets/preprocess/${selectedDataset.id}`, { method: 'POST' });
      setTimeout(() => setCleaningStatus('Imputing missing values via mean/mode...'), 600);
      setTimeout(() => setCleaningStatus('Encoding categorical features & scaling standard...'), 1200);
      setTimeout(() => {
        setCleaningStatus('');
        alert("Automated dataset preprocessing completed successfully!");
      }, 1800);
    } catch (e) {
      setCleaningStatus('');
    }
  };

  const startAutoMLTraining = async () => {
    if (!selectedDataset || !targetCol) {
      alert("Please select a target column to train.");
      return;
    }

    setIsTraining(true);
    setTrainingLogs(["Submitting task to Redis queue worker..."]);
    setTrainingProgress(0);

    try {
      const res = await fetch(`${BASE_URL}/api/automl/train`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          dataset_id: selectedDataset.id,
          target_col: targetCol
        })
      });
      const data = await res.json();
      const tempId = data.experiment_id;
      setActiveExpId(tempId);

      // Connect to websocket to stream logs in real-time!
      const ws = new WebSocket(`ws://127.0.0.1:8000/ws/training/${tempId}`);
      ws.onmessage = (event) => {
        const payload = JSON.parse(event.data);
        setTrainingProgress(payload.progress);
        setTrainingLogs(prev => [...prev, payload.message]);
        
        if (payload.progress >= 100) {
          setIsTraining(false);
          ws.close();
          fetchExperiments();
          fetchModels();
        }
      };
      
      ws.onerror = () => {
        setIsTraining(false);
      };
    } catch (e) {
      setIsTraining(false);
    }
  };

  const deployModel = async () => {
    if (!selectedModel) return;
    const name = deploymentName || `API_${selectedModel.name.replace(/\s+/g, '_')}_v2`;
    
    try {
      const res = await fetch(`${BASE_URL}/api/deployments/deploy`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          model_id: selectedModel.id,
          name: name
        })
      });
      const data = await res.json();
      alert("Model served instantly as active REST API!");
      setDeploymentName('');
      fetchDeployments();
      setServedEndpoint(data.endpoint);
    } catch (e) {
      console.error(e);
    }
  };

  const queryServedPrediction = async () => {
    if (!servedEndpoint) return;
    try {
      const res = await fetch(`${BASE_URL}${servedEndpoint}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(servedInputs)
      });
      const data = await res.json();
      setServedOutput(data);
    } catch (e) {
      console.error(e);
    }
  };

  // Pre-load serving payload keys when active endpoint changes
  useEffect(() => {
    if (servedEndpoint) {
      const dep = deployments.find(d => d.endpoint === servedEndpoint);
      if (dep) {
        const mdl = models.find(m => m.id === dep.model_id);
        if (mdl) {
          try {
            const imp = JSON.parse(mdl.feature_importance_json);
            const inputs = {};
            Object.keys(imp).forEach(k => {
              // Pre-fill realistic values
              if (k === 'user_credit_score') inputs[k] = 720;
              else if (k === 'annual_income_normalized') inputs[k] = 85000;
              else if (k === 'debt_to_income_ratio') inputs[k] = 0.25;
              else if (k === 'employment_duration') inputs[k] = 8;
              else if (k === 'loan_amount') inputs[k] = 20000;
              else inputs[k] = 1.0;
            });
            setServedInputs(inputs);
          } catch (e) {}
        }
      }
    }
  }, [servedEndpoint, deployments, models]);

  // Sidebar Layout Wrapper
  const renderSidebar = () => {
    const menuItems = [
      { id: 'dashboard', label: 'Overview', icon: Icons.Overview },
      { id: 'experiments', label: 'Live Metrics', icon: Icons.Metrics },
      { id: 'datasets', label: 'Training', icon: Icons.Training },
      { id: 'explainability', label: 'Deployments', icon: Icons.Explainability },
      { id: 'serving', label: 'Security', icon: Icons.Security }
    ];

    return (
      <div className="sidebar">
        <div className="sidebar-logo">
          AutoML<span>.Engine</span>
        </div>
        
        <div className="project-selector">
          <div className="project-title">Project Alpha</div>
          <div className="project-status">Active: v2.4.1</div>
        </div>

        <ul className="menu-list">
          {menuItems.map(item => {
            const Icon = item.icon;
            const tabId = item.id;
            // Sidebar tab routing
            const isActive = activeTab === tabId;
            return (
              <li key={tabId}>
                <button 
                  className={`menu-item-btn ${isActive ? 'active' : ''}`}
                  onClick={() => setActiveTab(tabId)}
                >
                  <Icon />
                  {item.label}
                </button>
              </li>
            );
          })}
        </ul>

        <div className="sidebar-footer">
          <button className="btn-premium primary" style={{fontSize: '11px', padding: '8px'}} onClick={() => setActiveTab('landing')}>
            Landing Page
          </button>
          <div style={{fontSize: '11px', color: 'hsl(var(--text-muted))', textAlign: 'center'}}>
            © 2026 AutoML Engine.
          </div>
        </div>
      </div>
    );
  };

  return (
    <div>
      {/* 1. Landing Hero (Custom Dark Theme View) */}
      {activeTab === 'landing' && (
        <div className="dark-theme-block" style={{ backgroundColor: 'hsl(var(--bg-slate))', color: 'hsl(var(--text-dark))', minHeight: '100vh', padding: '40px 80px', display: 'flex', flexDirection: 'column', gap: '60px' }}>
          {/* Header */}
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <div style={{ fontFamily: 'var(--font-heading)', fontSize: '24px', fontWeight: 800 }}>
              AutoML<span style={{ color: 'hsl(var(--teal-base))' }}>.Engine</span>
            </div>
            <div style={{ display: 'flex', gap: '30px', alignItems: 'center' }}>
              <span style={{ cursor: 'pointer', fontWeight: 600 }} onClick={() => setActiveTab('dashboard')}>Dashboard</span>
              <span style={{ cursor: 'pointer', fontWeight: 600 }} onClick={() => setActiveTab('experiments')}>Experiments</span>
              <span style={{ cursor: 'pointer', fontWeight: 600 }} onClick={() => setActiveTab('explainability')}>Explainability</span>
              <button className="btn-premium primary" onClick={() => setActiveTab('dashboard')}>Initialize Engine</button>
            </div>
          </div>

          {/* Hero Body */}
          <div style={{ display: 'grid', gridTemplateColumns: '1.2fr 1fr', gap: '60px', alignItems: 'center', marginTop: '20px' }}>
            <div style={{ display: 'flex', flexDirection: 'column', gap: '32px' }}>
              <div style={{ display: 'inline-flex', alignSelf: 'flex-start', padding: '6px 16px', background: 'rgba(0, 255, 255, 0.08)', border: '1px solid rgba(0,255,255,0.2)', color: '#00ffff', borderRadius: '30px', fontSize: '11px', fontWeight: 700, letterSpacing: '0.05em', textTransform: 'uppercase' }}>
                SCALE YOUR INTELLIGENCE
              </div>
              <h1 style={{ fontSize: '56px', lineHeight: 1.1 }}>
                AutoML at <span style={{ color: 'hsl(var(--teal-light))', fontStyle: 'italic' }}>scale.</span>
              </h1>
              <p style={{ color: 'hsl(var(--text-muted))', fontSize: '16px', lineHeight: 1.6 }}>
                Automate the end-to-end lifecycle of machine learning. From raw telemetry to production-ready inference, engineered for extreme data density and low-code deployments.
              </p>
              <div style={{ display: 'flex', gap: '16px' }}>
                <button className="btn-premium primary" onClick={() => setActiveTab('dashboard')}>
                  INITIALIZE_ENGINE
                </button>
                <button className="btn-premium secondary" style={{ background: 'transparent', color: 'white', border: '1px solid rgba(255,255,255,0.2)' }} onClick={() => setActiveTab('explainability')}>
                  DOCUMENTATION
                </button>
              </div>
            </div>

            {/* Premium Card Display */}
            <div style={{ background: 'radial-gradient(circle at center, hsla(var(--teal-base), 0.2) 0%, transparent 70%)', padding: '20px' }}>
              <div className="card-glass" style={{ border: '1px solid rgba(255,255,255,0.1)', padding: '24px', display: 'flex', flexDirection: 'column', gap: '16px' }}>
                <div style={{ borderBottom: '1px solid rgba(255,255,255,0.1)', paddingBottom: '12px', fontSize: '13px', fontFamily: 'monospace', color: 'hsl(var(--teal-light))' }}>
                  ACTIVE PIPELINE STATUS
                </div>
                <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '14px' }}>
                  <span>Data Preprocessing</span>
                  <span style={{ color: '#00ffaa' }}>[OPTIMIZED]</span>
                </div>
                <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '14px' }}>
                  <span>Asynchronous Train Queue</span>
                  <span style={{ color: '#ffff00' }}>[RUNNING]</span>
                </div>
                <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '14px' }}>
                  <span>Explainability Audit</span>
                  <span style={{ color: '#00e1ff' }}>[COMPLIANT]</span>
                </div>
                <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '14px' }}>
                  <span>Predict Serving Latency</span>
                  <span style={{ color: '#ff00ff' }}>&lt;5MS</span>
                </div>
              </div>
            </div>
          </div>

          {/* Infrastructure Stats */}
          <div style={{ borderTop: '1px solid rgba(255,255,255,0.1)', paddingTop: '40px', display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '40px', marginTop: '40px' }}>
            <div>
              <div style={{ fontSize: '32px', fontWeight: 800 }}>10PB+</div>
              <div style={{ fontSize: '12px', color: 'hsl(var(--text-muted))', marginTop: '4px' }}>DATA HANDLED</div>
            </div>
            <div>
              <div style={{ fontSize: '32px', fontWeight: 800 }}>0.9s</div>
              <div style={{ fontSize: '12px', color: 'hsl(var(--text-muted))', marginTop: '4px' }}>MEAN COLD START</div>
            </div>
            <div>
              <div style={{ fontSize: '32px', fontWeight: 800 }}>99.99%</div>
              <div style={{ fontSize: '12px', color: 'hsl(var(--text-muted))', marginTop: '4px' }}>UPTIME SLA</div>
            </div>
            <div>
              <div style={{ fontSize: '32px', fontWeight: 800 }} onClick={() => setActiveTab('dashboard')} style={{ cursor: 'pointer', color: 'hsl(var(--teal-light))' }}>LAUNCH PLATFORM →</div>
              <div style={{ fontSize: '12px', color: 'hsl(var(--text-muted))', marginTop: '4px' }}>ENTERPRISE READY</div>
            </div>
          </div>
        </div>
      )}

      {/* 2. Main Platform Shell (Light HSL Theme dashboards) */}
      {activeTab !== 'landing' && (
        <div className="app-container">
          {renderSidebar()}
          
          <div className="main-content">
            {/* Top Navigation Bar */}
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', borderBottom: '1px solid hsl(var(--border-light))', paddingBottom: '20px' }}>
              <div>
                <h1 style={{ fontSize: '36px', color: 'hsl(var(--text-dark))' }}>
                  {activeTab === 'dashboard' && "Platform Dashboard"}
                  {activeTab === 'experiments' && "Experiment Tracker"}
                  {activeTab === 'datasets' && "AutoML Pipeline"}
                  {activeTab === 'explainability' && "Explainability Hub"}
                  {activeTab === 'serving' && "Deployment Serving"}
                </h1>
                <p style={{ color: 'hsl(var(--text-muted))', fontSize: '13px', marginTop: '4px' }}>
                  {activeTab === 'dashboard' && "Live metrics, active pipelines, and cluster thread distribution."}
                  {activeTab === 'experiments' && "Hyperparameter sweep metrics, training runs, and diagnostics."}
                  {activeTab === 'datasets' && "Automated data preprocessing & asynchronous model training."}
                  {activeTab === 'explainability' && "Interpret model behavior, demographic fairness, and feature weightings."}
                  {activeTab === 'serving' && "One-click deployments served as REST prediction served nodes."}
                </p>
              </div>

              {/* Status Header Badge */}
              <div style={{ display: 'flex', gap: '16px', alignItems: 'center' }}>
                <span className="badge-status optimal">SYSTEM_HEALTH_OPTIMAL</span>
                <span style={{ fontSize: '12px', fontWeight: 600, color: 'hsl(var(--text-muted))' }}>GPU_CLUSTER_01: ONLINE</span>
              </div>
            </div>

            {/* TAB CONTENTS */}
            
            {/* Tab: Platform Dashboard */}
            {activeTab === 'dashboard' && (
              <div style={{ display: 'flex', flexDirection: 'column', gap: '32px' }}>
                {/* 4 Core Stats Cards */}
                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '24px' }}>
                  <div className="card-glass">
                    <div style={{ fontSize: '12px', color: 'hsl(var(--text-muted))', fontWeight: 600 }}>AVG_ACCURACY</div>
                    <div style={{ fontSize: '36px', fontWeight: 800, marginTop: '8px', color: 'hsl(var(--teal-base))' }}>98.42% <span style={{fontSize: '14px', color: 'green'}}>+0.12%</span></div>
                  </div>
                  <div className="card-glass">
                    <div style={{ fontSize: '12px', color: 'hsl(var(--text-muted))', fontWeight: 600 }}>TRAINING_LOSS</div>
                    <div style={{ fontSize: '36px', fontWeight: 800, marginTop: '8px', color: 'hsl(var(--text-dark))' }}>0.024 <span style={{fontSize: '14px', color: 'green'}}>↓ 4%</span></div>
                  </div>
                  <div className="card-glass">
                    <div style={{ fontSize: '12px', color: 'hsl(var(--text-muted))', fontWeight: 600 }}>F1_SCORE_WEIGHTED</div>
                    <div style={{ fontSize: '36px', fontWeight: 800, marginTop: '8px', color: 'hsl(var(--text-dark))' }}>0.971</div>
                  </div>
                  <div className="card-glass">
                    <div style={{ fontSize: '12px', color: 'hsl(var(--text-muted))', fontWeight: 600 }}>ACTIVE_THREADS</div>
                    <div style={{ fontSize: '36px', fontWeight: 800, marginTop: '8px', color: 'hsl(var(--indigo-base))' }}>1,024</div>
                  </div>
                </div>

                {/* Dashboard Middle Section */}
                <div style={{ display: 'grid', gridTemplateColumns: '1.8fr 1fr', gap: '32px' }}>
                  {/* Interactive SVG Convergence Line Chart */}
                  <div className="card-glass" style={{ minHeight: '360px' }}>
                    <h3 style={{ fontSize: '18px', marginBottom: '24px' }}>Training Convergence Velocity</h3>
                    <svg viewBox="0 0 500 200" style={{ width: '100%', height: '220px', overflow: 'visible' }}>
                      {/* Grid Lines */}
                      <line x1="40" y1="20" x2="480" y2="20" stroke="#f0f0f0" strokeWidth="1" />
                      <line x1="40" y1="70" x2="480" y2="70" stroke="#f0f0f0" strokeWidth="1" />
                      <line x1="40" y1="120" x2="480" y2="120" stroke="#f0f0f0" strokeWidth="1" />
                      <line x1="40" y1="170" x2="480" y2="170" stroke="#f0f0f0" strokeWidth="2" />
                      
                      {/* Convergence Curve */}
                      <path d="M 40 170 Q 150 160 220 120 T 380 50 T 480 30" fill="none" stroke="hsl(var(--teal-base))" strokeWidth="3" />
                      <circle cx="220" cy="120" r="4" fill="hsl(var(--teal-base))" />
                      <circle cx="380" cy="50" r="4" fill="hsl(var(--teal-base))" />
                      <circle cx="480" cy="30" r="4" fill="hsl(var(--teal-base))" />

                      {/* X Axes */}
                      <text x="40" y="190" fontSize="10" fill="#999" textAnchor="middle">00:00</text>
                      <text x="150" y="190" fontSize="10" fill="#999" textAnchor="middle">04:00</text>
                      <text x="260" y="190" fontSize="10" fill="#999" textAnchor="middle">08:00</text>
                      <text x="370" y="190" fontSize="10" fill="#999" textAnchor="middle">12:00</text>
                      <text x="480" y="190" fontSize="10" fill="#999" textAnchor="middle">20:00</text>
                    </svg>
                  </div>

                  {/* Compute Cluster Health Monitor */}
                  <div className="card-glass" style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
                    <h3 style={{ fontSize: '18px' }}>Compute Cluster Health</h3>
                    
                    <div>
                      <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '12px', fontWeight: 600, marginBottom: '6px' }}>
                        <span>GPU UTILIZATION (NVIDIA H100)</span>
                        <span style={{ color: 'hsl(var(--teal-base))' }}>82%</span>
                      </div>
                      <div className="shap-bar-track"><div className="shap-bar-fill" style={{ width: '82%', backgroundColor: 'hsl(var(--teal-base))' }}></div></div>
                    </div>

                    <div>
                      <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '12px', fontWeight: 600, marginBottom: '6px' }}>
                        <span>VRAM ALLOCATION</span>
                        <span style={{ color: 'hsl(var(--indigo-base))' }}>64.2 GB / 80 GB</span>
                      </div>
                      <div className="shap-bar-track"><div className="shap-bar-fill" style={{ width: '80%', backgroundColor: 'hsl(var(--indigo-base))' }}></div></div>
                    </div>

                    <div>
                      <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '12px', fontWeight: 600, marginBottom: '6px' }}>
                        <span>CPU LOAD (AMD EPYC)</span>
                        <span>41%</span>
                      </div>
                      <div className="shap-bar-track"><div className="shap-bar-fill" style={{ width: '41%', backgroundColor: '#666' }}></div></div>
                    </div>

                    {/* Alert Warning Box */}
                    <div style={{ background: 'hsl(var(--color-critical-bg))', border: '1px solid hsla(0, 100%, 50%, 0.1)', color: 'hsl(var(--color-critical))', padding: '14px', borderRadius: '8px', fontSize: '12px', display: 'flex', gap: '10px', marginTop: '10px' }}>
                      <span style={{ fontWeight: 800 }}>⚠️ WARNING</span>
                      <span>Thermal throttling detected on Node_7. Scaling tasks to active Node_8.</span>
                    </div>
                  </div>
                </div>

                {/* Experiments Dashboard Tracker Table */}
                <div className="card-glass">
                  <h3 style={{ fontSize: '18px', marginBottom: '20px' }}>Active AutoML Experiments</h3>
                  <div className="premium-table-container">
                    <table className="premium-table">
                      <thead>
                        <tr>
                          <th>ID</th>
                          <th>Model Architecture</th>
                          <th>Status</th>
                          <th>Duration</th>
                          <th>Accuracy</th>
                        </tr>
                      </thead>
                      <tbody>
                        {experiments.length === 0 ? (
                          <tr>
                            <td colSpan="5" style={{ textAlign: 'center', color: 'hsl(var(--text-muted))' }}>No active experiments found. Launch training in AutoML Pipeline tab.</td>
                          </tr>
                        ) : (
                          experiments.map(exp => (
                            <tr key={exp.id}>
                              <td style={{fontFamily: 'monospace'}}>{exp.id}</td>
                              <td>{exp.name}</td>
                              <td>
                                <span className={`badge-status ${exp.status === 'COMPLETED' ? 'optimal' : exp.status === 'RUNNING' ? 'training' : 'critical'}`}>
                                  {exp.status}
                                </span>
                              </td>
                              <td>{exp.duration || "Pending"}</td>
                              <td style={{fontWeight: 700}}>{exp.accuracy ? `${(exp.accuracy*100).toFixed(2)}%` : "N/A"}</td>
                            </tr>
                          ))
                        )}
                      </tbody>
                    </table>
                  </div>
                </div>
              </div>
            )}

            {/* Tab: Experiment Tracker */}
            {activeTab === 'experiments' && (
              <div style={{ display: 'flex', flexDirection: 'column', gap: '32px' }}>
                <div style={{ display: 'grid', gridTemplateColumns: '1.8fr 1fr', gap: '32px' }}>
                  {/* Metric Convergence Bar Sweep Chart */}
                  <div className="card-glass" style={{ minHeight: '340px' }}>
                    <h3 style={{ fontSize: '18px', marginBottom: '24px' }}>Metric Convergence Sweep</h3>
                    <div style={{ display: 'flex', justifyContent: 'space-around', alignItems: 'flex-end', height: '200px', borderBottom: '2px solid #ddd', paddingBottom: '10px' }}>
                      {[
                        { run: "Run_01", acc: 0.61, f1: 0.59 },
                        { run: "Run_02", acc: 0.77, f1: 0.75 },
                        { run: "Run_03", acc: 0.85, f1: 0.82 },
                        { run: "Run_04", acc: 0.96, f1: 0.94 },
                        { run: "Run_05", acc: 0.48, f1: 0.45 },
                        { run: "Run_06", acc: 0.68, f1: 0.65 }
                      ].map((item, i) => (
                        <div key={i} style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '8px', width: '60px' }}>
                          <div style={{ display: 'flex', gap: '4px', height: '140px', alignItems: 'flex-end' }}>
                            {/* Accuracy Bar (Teal) */}
                            <div style={{ width: '16px', height: `${item.acc * 140}px`, background: 'hsl(var(--teal-light))', borderRadius: '4px 4px 0 0', position: 'relative' }} title={`Acc: ${item.acc}`}></div>
                            {/* F1 Bar (Indigo) */}
                            <div style={{ width: '16px', height: `${item.f1 * 140}px`, background: 'hsl(var(--indigo-base))', borderRadius: '4px 4px 0 0', opacity: 0.8 }} title={`F1: ${item.f1}`}></div>
                          </div>
                          <span style={{ fontSize: '10px', color: '#666', fontFamily: 'monospace' }}>{item.run}</span>
                        </div>
                      ))}
                    </div>
                  </div>

                  {/* AI Insights Card */}
                  <div className="card-glass" style={{ borderTop: '4px solid hsl(var(--indigo-base))', display: 'flex', flexDirection: 'column', gap: '16px' }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '10px', color: 'hsl(var(--indigo-base))', fontWeight: 700, fontSize: '14px' }}>
                      <span>✨</span> AI INSIGHTS DIAGNOSTICS
                    </div>
                    {aiInsights ? (
                      <div style={{ display: 'flex', flexDirection: 'column', gap: '14px', fontSize: '13px', lineHeight: 1.5 }}>
                        <p style={{ fontWeight: 600 }}>{aiInsights.summary}</p>
                        
                        <div style={{ borderLeft: '3px solid hsl(var(--teal-base))', paddingLeft: '10px', display: 'flex', flexDirection: 'column', gap: '8px' }}>
                          {aiInsights.findings.map((f, idx) => (
                            <div key={idx}>• {f}</div>
                          ))}
                        </div>

                        <div style={{ background: '#f8f8fa', padding: '12px', borderRadius: '8px', fontSize: '11px', color: '#444' }}>
                          <span style={{ fontWeight: 800 }}>RECOMMENDATION SWEPT:</span>
                          <p style={{ marginTop: '4px' }}>{aiInsights.recommendations[1] || "Increase learning_rate step values to jump past saddle points."}</p>
                        </div>

                        <div style={{ display: 'flex', justifyContent: 'space-between', borderTop: '1px solid #eee', paddingTop: '10px', fontSize: '12px', fontWeight: 700 }}>
                          <span>CONFIDENCE SCORE</span>
                          <span style={{ color: 'hsl(var(--teal-base))' }}>{aiInsights.confidence_score}</span>
                        </div>
                      </div>
                    ) : (
                      <span style={{ color: 'hsl(var(--text-muted))' }}>No analysis loaded. Select a trained model in Explainability first.</span>
                    )}
                  </div>
                </div>

                {/* Hyperparameter sweeps grid */}
                <div className="card-glass">
                  <h3 style={{ fontSize: '18px', marginBottom: '20px' }}>Hyperparameter Sweep Results</h3>
                  <div className="premium-table-container">
                    <table className="premium-table">
                      <thead>
                        <tr>
                          <th>Run Name</th>
                          <th>Status</th>
                          <th>Batch Size</th>
                          <th>Learning Rate</th>
                          <th>Optimizer</th>
                          <th>Accuracy Equivalent</th>
                        </tr>
                      </thead>
                      <tbody>
                        <tr>
                          <td style={{fontFamily: 'monospace'}}>Run_04_Extreme_v2</td>
                          <td><span className="badge-status optimal">COMPLETED</span></td>
                          <td>128</td>
                          <td style={{fontFamily: 'monospace'}}>0.00042</td>
                          <td>AdamW</td>
                          <td style={{fontWeight: 700, color: 'hsl(var(--teal-base))'}}>0.9621</td>
                        </tr>
                        <tr>
                          <td style={{fontFamily: 'monospace'}}>Run_03_Sparse_Grid</td>
                          <td><span className="badge-status training">TRAINING</span></td>
                          <td>64</td>
                          <td style={{fontFamily: 'monospace'}}>0.00100</td>
                          <td>RMSprop</td>
                          <td style={{color: 'hsl(var(--text-muted))'}}>--</td>
                        </tr>
                        <tr>
                          <td style={{fontFamily: 'monospace'}}>Run_02_Base_Line</td>
                          <td><span className="badge-status critical">FAILED</span></td>
                          <td>32</td>
                          <td style={{fontFamily: 'monospace'}}>0.01000</td>
                          <td>SGD</td>
                          <td style={{fontWeight: 700, color: '#f00'}}>0.4501</td>
                        </tr>
                      </tbody>
                    </table>
                  </div>
                </div>
              </div>
            )}

            {/* Tab: AutoML Pipeline / Datasets upload */}
            {activeTab === 'datasets' && (
              <div style={{ display: 'flex', flexDirection: 'column', gap: '32px' }}>
                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1.2fr', gap: '32px' }}>
                  {/* Dataset upload and preview */}
                  <div className="card-glass" style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
                    <h3 style={{ fontSize: '18px' }}>Dataset Repository</h3>
                    
                    {/* Drag and Drop Zone */}
                    <div style={{ border: '2px dashed hsl(var(--border-light))', padding: '40px 20px', borderRadius: 'var(--radius-sm)', textAlign: 'center', cursor: 'pointer', background: 'hsl(var(--bg-sidebar))' }}>
                      <input 
                        type="file" 
                        accept=".csv,.xlsx,.xls,.json" 
                        id="dataset-upload-input" 
                        style={{ display: 'none' }} 
                        onChange={handleFileUpload} 
                      />
                      <label htmlFor="dataset-upload-input" style={{ cursor: 'pointer' }}>
                        <div style={{ fontSize: '32px', marginBottom: '10px' }}>📁</div>
                        <span style={{ fontWeight: 600, fontSize: '14px' }}>Click to select a local dataset</span>
                        <div style={{ fontSize: '11px', color: 'hsl(var(--text-muted))', marginTop: '6px' }}>Supported: CSV, Excel, JSON</div>
                      </label>
                    </div>

                    {/* Preprocessing Actions */}
                    {selectedDataset && (
                      <div style={{ display: 'flex', flexDirection: 'column', gap: '14px', borderTop: '1px solid #eee', paddingTop: '20px' }}>
                        <div style={{ fontSize: '13px' }}>
                          <span style={{ fontWeight: 700 }}>Active Dataset:</span> {selectedDataset.name}
                        </div>
                        <div style={{ display: 'flex', gap: '12px' }}>
                          <button className="btn-premium secondary" style={{ flex: 1 }} onClick={triggerPreprocessing}>
                            Clean Dataset (Preprocess)
                          </button>
                        </div>
                      </div>
                    )}

                    {cleaningStatus && (
                      <div style={{ background: 'hsl(var(--teal-glow))', color: 'hsl(var(--teal-base))', padding: '14px', borderRadius: '8px', fontSize: '12px', fontWeight: 600, animation: 'pulse-loading 1.2s infinite' }}>
                        ⏳ {cleaningStatus}
                      </div>
                    )}
                  </div>

                  {/* AutoML Target selection & Training Console */}
                  <div className="card-glass" style={{ display: 'flex', flexDirection: 'column', gap: '24px' }}>
                    <h3 style={{ fontSize: '18px' }}>Launch AutoML Training Engine</h3>
                    
                    {selectedDataset ? (
                      <div style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
                        <div>
                          <label style={{ display: 'block', fontSize: '12px', fontWeight: 700, color: 'hsl(var(--text-muted))', marginBottom: '8px' }}>
                            SELECT TARGET VARIABLE (LABEL)
                          </label>
                          <select 
                            value={targetCol} 
                            onChange={(e) => setTargetCol(e.target.value)}
                            style={{ width: '100%', padding: '12px', borderRadius: '8px', border: '1px solid hsl(var(--border-light))', fontSize: '14px' }}
                          >
                            <option value="">-- Choose target column --</option>
                            {selectedDataset.columns && selectedDataset.columns.map(col => (
                              <option key={col.name} value={col.name}>{col.name} ({col.type})</option>
                            ))}
                          </select>
                        </div>

                        <button 
                          className="btn-premium primary" 
                          style={{ width: '100%', padding: '14px' }}
                          onClick={startAutoMLTraining}
                          disabled={isTraining}
                        >
                          {isTraining ? "Distributed Engine Training Models..." : "RUN AUTOML PIPELINE"}
                        </button>
                      </div>
                    ) : (
                      <span style={{ color: 'hsl(var(--text-muted))' }}>Please import a CSV dataset file first to load AutoML controls.</span>
                    )}

                    {/* WebSockets Training Live Terminal Logs */}
                    {(isTraining || trainingLogs.length > 0) && (
                      <div style={{ background: '#0e1117', color: '#00ffaa', padding: '16px', borderRadius: '8px', fontFamily: 'monospace', fontSize: '11px', maxHeight: '180px', overflowY: 'auto', display: 'flex', flexDirection: 'column', gap: '6px' }}>
                        <div style={{ display: 'flex', justifyContent: 'space-between', borderBottom: '1px solid rgba(0, 255, 170, 0.2)', paddingBottom: '6px', marginBottom: '4px' }}>
                          <span>LIVE TELEMETRY STREAM</span>
                          <span>{trainingProgress}%</span>
                        </div>
                        {trainingLogs.map((log, idx) => (
                          <div key={idx}>{log}</div>
                        ))}
                      </div>
                    )}
                  </div>
                </div>

                {/* Dataset Preview Schema Grid */}
                {selectedDataset && (
                  <div className="card-glass">
                    <h3 style={{ fontSize: '18px', marginBottom: '20px' }}>Dataset Schema & Previews</h3>
                    <div className="premium-table-container">
                      <table className="premium-table">
                        <thead>
                          <tr>
                            <th>Column Name</th>
                            <th>Inferred Schema</th>
                            <th>Type</th>
                            <th>Unique Values</th>
                            <th>Missing Ratio</th>
                          </tr>
                        </thead>
                        <tbody>
                          {selectedDataset.columns && selectedDataset.columns.map(col => (
                            <tr key={col.name}>
                              <td style={{fontFamily: 'monospace', fontWeight: 600}}>{col.name}</td>
                              <td>
                                <span className={`badge-status ${col.type === 'categorical' ? 'optimal' : 'monitor'}`}>
                                  {col.type}
                                </span>
                              </td>
                              <td style={{fontFamily: 'monospace', color: '#777'}}>{col.dtype}</td>
                              <td>{col.unique_count}</td>
                              <td>{col.missing_pct > 0 ? `${(col.missing_pct*100).toFixed(1)}%` : "0.0%"}</td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  </div>
                )}
              </div>
            )}

            {/* Tab: Explainability Hub */}
            {activeTab === 'explainability' && (
              <div style={{ display: 'flex', flexDirection: 'column', gap: '32px' }}>
                {/* Model Selector Bar */}
                <div className="card-glass" style={{ padding: '16px 28px', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                  <div style={{ display: 'flex', gap: '14px', alignItems: 'center' }}>
                    <span style={{ fontWeight: 700, fontSize: '14px' }}>Active served candidate:</span>
                    <select 
                      value={selectedModel ? selectedModel.id : ''} 
                      onChange={(e) => {
                        const m = models.find(x => x.id === e.target.value);
                        if (m) setSelectedModel(m);
                      }}
                      style={{ padding: '8px 16px', borderRadius: '6px', border: '1px solid hsl(var(--border-light))', fontSize: '13px' }}
                    >
                      {models.map(m => (
                        <option key={m.id} value={m.id}>{m.name} ({m.id}) - Acc: {(m.accuracy*100).toFixed(1)}%</option>
                      ))}
                    </select>
                  </div>

                  <span style={{ fontSize: '11px', color: 'hsl(var(--indigo-base))', fontWeight: 700 }}>SHAP_STABILITY: HIGH</span>
                </div>

                {selectedModel ? (
                  <div style={{ display: 'flex', flexDirection: 'column', gap: '32px' }}>
                    {/* Top Row: Global Feature Importance & Fairness */}
                    <div style={{ display: 'grid', gridTemplateColumns: '1.8fr 1fr', gap: '32px' }}>
                      {/* Global Feature Importance Progress bars */}
                      <div className="card-glass">
                        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '20px' }}>
                          <h3 style={{ fontSize: '18px' }}>Global Feature Importance</h3>
                          {/* Toggle buttons SHAP vs Permutation */}
                          <div style={{ display: 'flex', border: '1px solid #ddd', borderRadius: '6px', overflow: 'hidden' }}>
                            <button 
                              style={{ padding: '6px 12px', border: 'none', background: shapMethod === 'shap' ? '#007f80' : 'transparent', color: shapMethod === 'shap' ? 'white' : '#666', fontSize: '11px', fontWeight: 700, cursor: 'pointer' }}
                              onClick={() => setShapMethod('shap')}
                            >
                              SHAP
                            </button>
                            <button 
                              style={{ padding: '6px 12px', border: 'none', background: shapMethod === 'permutation' ? '#007f80' : 'transparent', color: shapMethod === 'permutation' ? 'white' : '#666', fontSize: '11px', fontWeight: 700, cursor: 'pointer' }}
                              onClick={() => setShapMethod('permutation')}
                            >
                              PERMUTATION
                            </button>
                          </div>
                        </div>

                        <div className="shap-bar-container">
                          {featureImportance.map((f, idx) => (
                            <div className="shap-bar-row" key={idx}>
                              <div className="shap-bar-labels">
                                <span className="shap-bar-name">{f.feature}</span>
                                <span className="shap-bar-value">{f.value}</span>
                              </div>
                              <div className="shap-bar-track">
                                <div className="shap-bar-fill" style={{ width: `${f.value * 100}%` }}></div>
                              </div>
                            </div>
                          ))}
                        </div>
                      </div>

                      {/* Fairness Analytics table */}
                      <div className="card-glass" style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
                        <h3 style={{ fontSize: '18px' }}>Fairness Analytics</h3>
                        <p style={{ fontSize: '12px', color: 'hsl(var(--text-muted))', lineHeight: 1.4 }}>
                          Disparate impact parity ratios computed across protected subgroups.
                        </p>
                        
                        <div style={{ display: 'flex', flexDirection: 'column', gap: '14px' }}>
                          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', borderBottom: '1px solid #eee', paddingBottom: '8px' }}>
                            <div>
                              <span style={{ fontSize: '12px', fontWeight: 700 }}>GENDER_BIAS</span>
                              <div style={{ fontSize: '11px', color: '#666' }}>Parity: 0.98</div>
                            </div>
                            <span className="badge-status optimal">OPTIMAL</span>
                          </div>

                          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', borderBottom: '1px solid #eee', paddingBottom: '8px' }}>
                            <div>
                              <span style={{ fontSize: '12px', fontWeight: 700 }}>AGE_BIAS</span>
                              <div style={{ fontSize: '11px', color: '#666' }}>Parity: 0.81</div>
                            </div>
                            <span className="badge-status monitor">MONITOR</span>
                          </div>

                          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', borderBottom: '1px solid #eee', paddingBottom: '8px' }}>
                            <div>
                              <span style={{ fontSize: '12px', fontWeight: 700 }}>LOCATION_BIAS</span>
                              <div style={{ fontSize: '11px', color: '#666' }}>Parity: 0.95</div>
                            </div>
                            <span className="badge-status optimal">OPTIMAL</span>
                          </div>
                        </div>

                        <div style={{ background: '#fcf8e3', border: '1px solid #fbeed5', color: '#c09853', padding: '12px', borderRadius: '8px', fontSize: '11px', marginTop: '10px' }}>
                          ⚠️ <span style={{ fontWeight: 800 }}>MITIGATION_RECOMMENDED</span>: Discrepancy detected in 'Age' feature weighting bounds. Consider training bias mitigation filters.
                        </div>
                      </div>
                    </div>

                    {/* Middle Row: Local Prediction SHAP Force Plot */}
                    <div className="card-glass">
                      <h3 style={{ fontSize: '18px', marginBottom: '10px' }}>SHAP Force Plot (Local Prediction Interpretation)</h3>
                      <p style={{ fontSize: '12px', color: 'hsl(var(--text-muted))', marginBottom: '24px' }}>
                        Attribution details for prediction instance ID: <span style={{fontFamily: 'monospace'}}>TXN-90210-A</span>
                      </p>
                      
                      {/* Force Plot SVG Visualizer */}
                      <div style={{ background: '#f8f9fa', padding: '24px', borderRadius: '8px', border: '1px solid #eee' }}>
                        <div style={{ display: 'flex', height: '40px', borderRadius: '6px', overflow: 'hidden', border: '1px solid #ddd' }}>
                          <div style={{ flex: 1.5, background: 'linear-gradient(to right, #f2dede, #ebccd1)', display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: '11px', fontWeight: 700, color: '#a94442' }}>
                            Base Value: 0.42
                          </div>
                          <div style={{ width: '80px', background: 'white', display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', borderLeft: '2px solid black', borderRight: '2px solid black', zIndex: 1 }}>
                            <span style={{ fontSize: '9px', fontWeight: 700, color: '#666' }}>OUTPUT</span>
                            <span style={{ fontSize: '16px', fontWeight: 800 }}>0.86</span>
                          </div>
                          <div style={{ flex: 2.5, background: 'linear-gradient(to right, #d9edf7, #bce8f1)', display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: '11px', fontWeight: 700, color: '#31708f' }}>
                            Push attribution (Correlation indicators)
                          </div>
                        </div>

                        {/* Interactive push/pull details */}
                        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(5, 1fr)', gap: '20px', marginTop: '24px', fontSize: '12px', textAlign: 'center' }}>
                          <div>
                            <span style={{ color: 'green', fontWeight: 700 }}>+0.12</span>
                            <div style={{ fontWeight: 600, fontFamily: 'monospace' }}>Credit_Score = 780</div>
                          </div>
                          <div>
                            <span style={{ color: 'green', fontWeight: 700 }}>+0.09</span>
                            <div style={{ fontWeight: 600, fontFamily: 'monospace' }}>Income = $125k</div>
                          </div>
                          <div>
                            <span style={{ color: 'red', fontWeight: 700 }}>-0.05</span>
                            <div style={{ fontWeight: 600, fontFamily: 'monospace' }}>DTI = 28%</div>
                          </div>
                          <div>
                            <span style={{ color: 'green', fontWeight: 700 }}>+0.04</span>
                            <div style={{ fontWeight: 600, fontFamily: 'monospace' }}>Years_Emp = 12</div>
                          </div>
                          <div>
                            <span style={{ color: 'red', fontWeight: 700 }}>-0.02</span>
                            <div style={{ fontWeight: 600, fontFamily: 'monospace' }}>Loan_Amt = $45k</div>
                          </div>
                        </div>
                      </div>
                    </div>

                    {/* Bottom Row: LIME & PDP side-by-side */}
                    <div style={{ display: 'grid', gridTemplateColumns: '1.2fr 1fr', gap: '32px' }}>
                      {/* LIME Scatterplot panel */}
                      <div className="card-glass">
                        <h3 style={{ fontSize: '18px', marginBottom: '6px' }}>LIME Explanation</h3>
                        <p style={{ fontSize: '12px', color: 'hsl(var(--text-muted))', marginBottom: '20px' }}>
                          Local Surrogate Interpretability Scatterplot model.
                        </p>
                        
                        <div style={{ height: '200px', background: '#fafafa', border: '1px solid #eee', borderRadius: '8px', position: 'relative', overflow: 'hidden', padding: '10px' }}>
                          {/* LIME Points rendering inside a simulated plot */}
                          {limeData && limeData.points.map((p, idx) => (
                            <div 
                              key={idx} 
                              style={{ 
                                position: 'absolute', 
                                left: `${p.x * 90 + 5}%`, 
                                bottom: `${p.y * 80 + 10}%`, 
                                width: '6px', 
                                height: '6px', 
                                borderRadius: '50%', 
                                backgroundColor: p.class === 1 ? '#7f00ff' : '#007f80',
                                opacity: 0.7 
                              }}
                            ></div>
                          ))}
                          {/* Surrogate Split Line */}
                          <div style={{ position: 'absolute', left: '0', bottom: '10%', width: '100%', height: '1px', background: '#e066ff', transform: 'rotate(-25deg)', transformOrigin: 'left bottom', borderTop: '2px dashed #7f00ff', opacity: 0.8 }}></div>
                        </div>
                      </div>

                      {/* PDP Plot Curve panel */}
                      <div className="card-glass">
                        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '20px' }}>
                          <h3 style={{ fontSize: '18px' }}>Partial Dependence (PDP)</h3>
                          {/* Selector to trigger new PDP calculations */}
                          <select 
                            value={pdpFeature} 
                            onChange={(e) => {
                              setPdpFeature(e.target.value);
                              fetchPdpCoords(selectedModel.id, e.target.value);
                            }}
                            style={{ padding: '4px 8px', borderRadius: '4px', border: '1px solid #ddd', fontSize: '12px' }}
                          >
                            {selectedModel.feature_importance_json && Object.keys(JSON.parse(selectedModel.feature_importance_json)).map(k => (
                              <option key={k} value={k}>{k}</option>
                            ))}
                          </select>
                        </div>
                        
                        <p style={{ fontSize: '12px', color: 'hsl(var(--text-muted))', marginBottom: '20px' }}>
                          Relationship between selected feature and predicted outcome.
                        </p>

                        {/* Line Curve SVG visualizer */}
                        {pdpCoords && (
                          <div style={{ height: '180px', background: '#fafafa', border: '1px solid #eee', borderRadius: '8px', padding: '10px' }}>
                            <svg viewBox="0 0 200 100" style={{ width: '100%', height: '100%', overflow: 'visible' }}>
                              <path 
                                d={`M ${pdpCoords.x.map((x, idx) => `${idx * 8 + 10} ${100 - pdpCoords.y[idx] * 80}`).join(' L ')}`} 
                                fill="none" 
                                stroke="#007f80" 
                                strokeWidth="2" 
                              />
                            </svg>
                          </div>
                        )}
                      </div>
                    </div>
                  </div>
                ) : (
                  <span style={{ color: 'hsl(var(--text-muted))' }}>No trained model files found. Upload a dataset and train a model first.</span>
                )}
              </div>
            )}

            {/* Tab: Serving REST Playground */}
            {activeTab === 'serving' && (
              <div style={{ display: 'flex', flexDirection: 'column', gap: '32px' }}>
                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1.2fr', gap: '32px' }}>
                  {/* Deploy model control */}
                  <div className="card-glass" style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
                    <h3 style={{ fontSize: '18px' }}>Serve Model REST API</h3>
                    
                    {models.length > 0 ? (
                      <div style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
                        <div>
                          <label style={{ display: 'block', fontSize: '12px', fontWeight: 700, color: 'hsl(var(--text-muted))', marginBottom: '8px' }}>
                            CHOOSE TRAINED MODEL TO SERVE
                          </label>
                          <select 
                            value={selectedModel ? selectedModel.id : ''} 
                            onChange={(e) => {
                              const m = models.find(x => x.id === e.target.value);
                              if (m) setSelectedModel(m);
                            }}
                            style={{ width: '100%', padding: '12px', borderRadius: '8px', border: '1px solid hsl(var(--border-light))', fontSize: '14px' }}
                          >
                            {models.map(m => (
                              <option key={m.id} value={m.id}>{m.name} ({m.id})</option>
                            ))}
                          </select>
                        </div>

                        <div>
                          <label style={{ display: 'block', fontSize: '12px', fontWeight: 700, color: 'hsl(var(--text-muted))', marginBottom: '8px' }}>
                            DEPLOYMENT ENPOINT NAME
                          </label>
                          <input 
                            type="text" 
                            placeholder="e.g. credit_risk_api_v1"
                            value={deploymentName}
                            onChange={(e) => setDeploymentName(e.target.value)}
                            style={{ width: '100%', padding: '12px', borderRadius: '8px', border: '1px solid hsl(var(--border-light))', fontSize: '14px' }}
                          />
                        </div>

                        <button className="btn-premium purple-glow" style={{ padding: '14px' }} onClick={deployModel}>
                          🚀 DEPLOY AS REST ENDPOINT
                        </button>
                      </div>
                    ) : (
                      <span style={{ color: 'hsl(var(--text-muted))' }}>No trained model found. Launch AutoML in Training first.</span>
                    )}
                  </div>

                  {/* REST Serving Playground console client */}
                  <div className="card-glass" style={{ display: 'flex', flexDirection: 'column', gap: '24px' }}>
                    <h3 style={{ fontSize: '18px' }}>API Playground</h3>
                    <p style={{ fontSize: '12px', color: 'hsl(var(--text-muted))' }}>
                      Send sample JSON inputs to your active served REST endpoint and review the dynamic prediction outputs!
                    </p>

                    {deployments.length > 0 ? (
                      <div style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
                        <div>
                          <label style={{ display: 'block', fontSize: '12px', fontWeight: 700, color: 'hsl(var(--text-muted))', marginBottom: '8px' }}>
                            SELECT ACTIVE REST API
                          </label>
                          <select 
                            value={servedEndpoint} 
                            onChange={(e) => setServedEndpoint(e.target.value)}
                            style={{ width: '100%', padding: '12px', borderRadius: '8px', border: '1px solid hsl(var(--border-light))', fontSize: '14px' }}
                          >
                            {deployments.map(dep => (
                              <option key={dep.id} value={dep.endpoint}>{dep.name} (POST {dep.endpoint})</option>
                            ))}
                          </select>
                        </div>

                        {/* Interactive fields input */}
                        <div style={{ display: 'flex', flexDirection: 'column', gap: '14px', background: '#fafafa', padding: '16px', borderRadius: '8px', border: '1px solid #eee' }}>
                          <span style={{ fontWeight: 700, fontSize: '12px', borderBottom: '1px solid #eee', paddingBottom: '6px' }}>JSON PARAMETERS</span>
                          
                          {Object.keys(servedInputs).map(key => (
                            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '16px', alignItems: 'center' }} key={key}>
                              <span style={{ fontSize: '12px', fontFamily: 'monospace', fontWeight: 600 }}>{key}:</span>
                              <input 
                                type="number" 
                                value={servedInputs[key]} 
                                onChange={(e) => setServedInputs(prev => ({ ...prev, [key]: e.target.value }))}
                                style={{ padding: '6px 10px', borderRadius: '4px', border: '1px solid #ddd', fontSize: '13px' }}
                              />
                            </div>
                          ))}
                        </div>

                        <button className="btn-premium primary" style={{ width: '100%', padding: '14px' }} onClick={queryServedPrediction}>
                          SEND POST /PREDICT REQUEST
                        </button>
                      </div>
                    ) : (
                      <span style={{ color: 'hsl(var(--text-muted))' }}>No active deployed APIs. Please serve a trained model first.</span>
                    )}

                    {/* prediction serve result console */}
                    {servedOutput && (
                      <div style={{ background: '#0e1117', color: '#00ffaa', padding: '16px', borderRadius: '8px', fontFamily: 'monospace', fontSize: '11px' }}>
                        <div style={{ borderBottom: '1px solid rgba(0, 255, 170, 0.2)', paddingBottom: '6px', marginBottom: '8px', fontWeight: 800 }}>
                          RESPONSE BODY (JSON)
                        </div>
                        <pre style={{ overflowX: 'auto' }}>{JSON.stringify(servedOutput, null, 2)}</pre>
                      </div>
                    )}
                  </div>
                </div>

                {/* Serving Deployments Directory */}
                <div className="card-glass">
                  <h3 style={{ fontSize: '18px', marginBottom: '20px' }}>Active Deployments Endpoint Directory</h3>
                  <div className="premium-table-container">
                    <table className="premium-table">
                      <thead>
                        <tr>
                          <th>Name</th>
                          <th>Method & Served URL</th>
                          <th>Status</th>
                          <th>Request Count</th>
                        </tr>
                      </thead>
                      <tbody>
                        {deployments.length === 0 ? (
                          <tr>
                            <td colSpan="4" style={{ textAlign: 'center', color: 'hsl(var(--text-muted))' }}>No active serving deployments. Serve a model from AutoML above.</td>
                          </tr>
                        ) : (
                          deployments.map(dep => (
                            <tr key={dep.id}>
                              <td style={{fontWeight: 700}}>{dep.name}</td>
                              <td style={{fontFamily: 'monospace', color: 'hsl(var(--teal-base))'}}>
                                POST {BASE_URL}{dep.endpoint}
                              </td>
                              <td><span className="badge-status optimal">ACTIVE</span></td>
                              <td style={{fontWeight: 700}}>{dep.request_count}</td>
                            </tr>
                          ))
                        )}
                      </tbody>
                    </table>
                  </div>
                </div>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}

export default App;
