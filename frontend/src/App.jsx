import { useState, useEffect } from 'react'
import './App.css'

const API_URL = 'http://localhost:8000'

// Biomarker normalization ranges (real-world medical units)
const BIOMARKER_RANGES = {
  age: { min: 20, max: 85, unit: 'years', step: 1, default: 45 },
  income_ratio: { min: 0, max: 5, unit: 'ratio', step: 0.1, default: 2.5 },
  body_mass_index: { min: 15, max: 50, unit: 'kg/m²', step: 0.1, default: 25 },
  height_cm: { min: 140, max: 210, unit: 'cm', step: 1, default: 170 },
  heart_rate_bpm: { min: 40, max: 140, unit: 'bpm', step: 1, default: 72 },
  white_blood_cells: { min: 2, max: 20, unit: '10³/µL', step: 0.1, default: 7 },
  platelets_count: { min: 50, max: 500, unit: '10³/µL', step: 5, default: 250 },
  hemoglobin: { min: 8, max: 18, unit: 'g/dL', step: 0.1, default: 14 },
  mcv: { min: 60, max: 110, unit: 'fL', step: 1, default: 88 },
  creatinine: { min: 0.3, max: 5, unit: 'mg/dL', step: 0.1, default: 1.0 },
  liver_ast: { min: 5, max: 150, unit: 'U/L', step: 1, default: 25 },
  bilirubin: { min: 0.1, max: 3, unit: 'mg/dL', step: 0.1, default: 0.8 },
  liver_ggt: { min: 5, max: 200, unit: 'U/L', step: 1, default: 30 },
  uric_acid: { min: 2, max: 12, unit: 'mg/dL', step: 0.1, default: 5.5 },
  sodium: { min: 130, max: 155, unit: 'mmol/L', step: 1, default: 140 },
  potassium: { min: 2.5, max: 6, unit: 'mmol/L', step: 0.1, default: 4.2 },
  cholesterol: { min: 100, max: 400, unit: 'mg/dL', step: 5, default: 200 },
  alcohol: { min: 0, max: 21, unit: 'drinks/week', step: 1, default: 2 }
}

// Healthy/normal biomarker values based on clinical reference ranges
// Optimized to reduce metabolic syndrome risk prediction - EXTREME OPTIMAL
// Sources: Mayo Clinic, Cleveland Clinic, Medscape, NIH
const HEALTHY_BIOMARKERS = {
  age: 25,                    // Young adult (lowest metabolic risk)
  income_ratio: 5.0,          // High income (max health correlation)
  body_mass_index: 19.5,      // Very lean optimal BMI (18.5-24.9 range)
  height_cm: 180,             // Taller frame
  heart_rate_bpm: 55,         // Elite athlete resting rate
  white_blood_cells: 5.5,     // Low-normal (4.5-11.0)
  platelets_count: 220,       // Low-normal (150-400)
  hemoglobin: 15.0,           // Optimal (12-17 g/dL)
  mcv: 92,                    // Optimal (80-100 fL)
  creatinine: 0.8,            // Excellent kidney function
  liver_ast: 15,              // Very low-normal liver (8-33 U/L)
  bilirubin: 0.3,             // Very low-normal (0.1-1.2 mg/dL)
  liver_ggt: 12,              // Very low-normal (5-40 U/L)
  uric_acid: 3.5,             // Low-normal (2.5-7.0 mg/dL)
  sodium: 140,                // Optimal (136-145 mmol/L)
  potassium: 4.0,             // Optimal (3.5-5.0 mmol/L)
  cholesterol: 130,           // Very optimal (<200, athletes often <150)
  alcohol: 0                  // Non-drinker
}

// Normalize a real-world value to 0-1 scale
const normalizeValue = (name, value) => {
  const range = BIOMARKER_RANGES[name]
  if (!range) return value
  return (value - range.min) / (range.max - range.min)
}

// Features for each model
const MODEL_FEATURES = {
  classification: [
    { name: 'age', label: 'Age' },
    { name: 'income_ratio', label: 'Income Ratio' },
    { name: 'body_mass_index', label: 'Body Mass Index' },
    { name: 'height_cm', label: 'Height' },
    { name: 'heart_rate_bpm', label: 'Heart Rate' },
    { name: 'white_blood_cells', label: 'White Blood Cells' },
    { name: 'platelets_count', label: 'Platelets Count' },
    { name: 'hemoglobin', label: 'Hemoglobin' },
    { name: 'mcv', label: 'Mean Corpuscular Volume' },
    { name: 'creatinine', label: 'Creatinine' },
    { name: 'liver_ast', label: 'Aspartate Aminotransferase' },
    { name: 'bilirubin', label: 'Bilirubin' },
    { name: 'liver_ggt', label: 'Gamma-Glutamyl Transferase' },
    { name: 'uric_acid', label: 'Uric Acid' },
    { name: 'sodium', label: 'Sodium' },
    { name: 'potassium', label: 'Potassium' },
    { name: 'cholesterol', label: 'Cholesterol' },
    { name: 'alcohol', label: 'Alcohol per Week' },
  ],
  regression: [
    { name: 'age', label: 'Age' },
    { name: 'income_ratio', label: 'Income Ratio' },
    { name: 'body_mass_index', label: 'Body Mass Index' },
    { name: 'height_cm', label: 'Height' },
    { name: 'heart_rate_bpm', label: 'Heart Rate' },
    { name: 'white_blood_cells', label: 'White Blood Cells' },
    { name: 'platelets_count', label: 'Platelets Count' },
    { name: 'hemoglobin', label: 'Hemoglobin' },
    { name: 'mcv', label: 'Mean Corpuscular Volume' },
    { name: 'creatinine', label: 'Creatinine' },
    { name: 'liver_ast', label: 'Aspartate Aminotransferase' },
    { name: 'bilirubin', label: 'Bilirubin' },
    { name: 'liver_ggt', label: 'Gamma-Glutamyl Transferase' },
    { name: 'uric_acid', label: 'Uric Acid' },
    { name: 'sodium', label: 'Sodium' },
    { name: 'potassium', label: 'Potassium' },
    { name: 'cholesterol', label: 'Cholesterol' },
    { name: 'alcohol', label: 'Alcohol per Week' },
  ],
  mtl: [
    { name: 'age', label: 'Age' },
    { name: 'income_ratio', label: 'Income Ratio' },
    { name: 'body_mass_index', label: 'Body Mass Index' },
    { name: 'height_cm', label: 'Height' },
    { name: 'heart_rate_bpm', label: 'Heart Rate' },
    { name: 'white_blood_cells', label: 'White Blood Cells' },
    { name: 'platelets_count', label: 'Platelets Count' },
    { name: 'hemoglobin', label: 'Hemoglobin' },
    { name: 'mcv', label: 'Mean Corpuscular Volume' },
    { name: 'creatinine', label: 'Creatinine' },
    { name: 'liver_ast', label: 'Aspartate Aminotransferase' },
    { name: 'bilirubin', label: 'Bilirubin' },
    { name: 'liver_ggt', label: 'Gamma-Glutamyl Transferase' },
    { name: 'uric_acid', label: 'Uric Acid' },
    { name: 'sodium', label: 'Sodium' },
    { name: 'potassium', label: 'Potassium' },
    { name: 'cholesterol', label: 'Cholesterol' },
    { name: 'alcohol', label: 'Alcohol per Week' },
  ]
}

// Additional features for regression model
const REGRESSION_EXTRA_FEATURES = [
  { name: 'has_cvd', label: 'Has CVD', type: 'binary' },
  { name: 'high_waist', label: 'High Waist', type: 'binary' },
  { name: 'high_triglycerides', label: 'High Triglycerides', type: 'binary' },
  { name: 'low_hdl', label: 'Low HDL', type: 'binary' },
  { name: 'high_bp', label: 'High BP', type: 'binary' },
]

const CLASS_COLORS = {
  'Normal': '#10b981',
  'Microalbuminuria': '#f59e0b',
  'Macroalbuminuria': '#ef4444'
}

function App() {
  const [activeTab, setActiveTab] = useState('home')
  const [selectedModel, setSelectedModel] = useState('classification')
  const [features, setFeatures] = useState({})
  const [gender, setGender] = useState(1)
  const [ethnicity, setEthnicity] = useState(1)
  const [smoking, setSmoking] = useState(1)
  const [binaryFeatures, setBinaryFeatures] = useState({})
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)
  const [modelsAvailable, setModelsAvailable] = useState({})

  // Initialize features for selected model with real-world defaults
  useEffect(() => {
    const modelFeatures = MODEL_FEATURES[selectedModel] || []
    const initialFeatures = {}
    modelFeatures.forEach(f => {
      const range = BIOMARKER_RANGES[f.name]
      initialFeatures[f.name] = range ? range.default : 0.5
    })
    setFeatures(initialFeatures)

    if (selectedModel === 'regression') {
      const initialBinary = {}
      REGRESSION_EXTRA_FEATURES.forEach(f => { initialBinary[f.name] = 0 })
      setBinaryFeatures(initialBinary)
    }
    setResult(null)
  }, [selectedModel])

  // Check available models on load
  useEffect(() => {
    fetch(`${API_URL}/`)
      .then(res => res.json())
      .then(data => setModelsAvailable(data.models_available || {}))
      .catch(() => { })
  }, [])

  const buildFeatureVector = () => {
    const modelFeatures = MODEL_FEATURES[selectedModel] || []
    // Normalize real-world values to 0-1 scale for model input
    const featureValues = modelFeatures.map(f => {
      const value = features[f.name]
      const range = BIOMARKER_RANGES[f.name]
      if (range) {
        return normalizeValue(f.name, value !== undefined ? value : range.default)
      }
      return value !== undefined ? value : 0.5
    })

    // Add gender one-hot
    featureValues.push(gender === 1 ? 1 : 0, gender === 2 ? 1 : 0)
      // Add ethnicity one-hot
      ;[1, 2, 3, 4, 6, 7].forEach(e => featureValues.push(e === ethnicity ? 1 : 0))
      // Add smoking one-hot
      ;[1, 2, 3].forEach(s => featureValues.push(s === smoking ? 1 : 0))
    featureValues.push(0) // smoking_nan

    // Add regression-specific binary features
    if (selectedModel === 'regression') {
      REGRESSION_EXTRA_FEATURES.forEach(f => {
        featureValues.push(binaryFeatures[f.name] || 0)
      })
    }

    return featureValues
  }

  const predict = async () => {
    setLoading(true)
    setResult(null)
    try {
      const response = await fetch(`${API_URL}/predict/${selectedModel}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ features: buildFeatureVector() })
      })
      const data = await response.json()
      if (!response.ok) throw new Error(data.detail || 'Prediction failed')
      setResult(data)
    } catch (error) {
      setResult({ error: error.message })
    }
    setLoading(false)
  }

  const currentFeatures = MODEL_FEATURES[selectedModel] || []

  return (
    <div className="app">
      <div className="hero-bg"></div>

      {/* Navigation */}
      <nav className="navbar">
        <div className="nav-brand">
          <span className="brand-icon">🧬</span>
          <span className="brand-text">Clinical AI</span>
        </div>
        <div className="nav-links">
          <button className={activeTab === 'home' ? 'active' : ''} onClick={() => setActiveTab('home')}>Home</button>
          <button className={activeTab === 'predict' ? 'active' : ''} onClick={() => setActiveTab('predict')}>Predict</button>
          <button className={activeTab === 'about' ? 'active' : ''} onClick={() => setActiveTab('about')}>About</button>
          <button className={activeTab === 'models' ? 'active' : ''} onClick={() => setActiveTab('models')}>Models</button>
        </div>
      </nav>

      <main className="main-content">

        {/* HOME */}
        {activeTab === 'home' && (
          <section className="hero-section">
            <div className="hero-content">
              <h1 className="hero-title">
                <span className="gradient-text">Clinical Prediction</span>
                <br />Intelligence Platform
              </h1>
              <p className="hero-subtitle">
                Advanced deep learning models for predicting kidney disease,
                cardiovascular risk, and metabolic syndrome from NHANES biomarkers.
              </p>
              <div className="hero-buttons">
                <button className="btn-primary" onClick={() => setActiveTab('predict')}>
                  Try Prediction →
                </button>
                <button className="btn-secondary" onClick={() => setActiveTab('models')}>
                  View Models
                </button>
              </div>

              <div className="stats-row">
                <div className="stat-card"><span className="stat-number">3</span><span className="stat-label">AI Models</span></div>
                <div className="stat-card"><span className="stat-number">34K+</span><span className="stat-label">Training Samples</span></div>
                <div className="stat-card"><span className="stat-number">30+</span><span className="stat-label">Biomarkers</span></div>
                <div className="stat-card"><span className="stat-number">90%</span><span className="stat-label">Accuracy</span></div>
              </div>
            </div>
            <div className="hero-visual">
              <img src="/hero.png" alt="Medical AI" className="hero-image" />
            </div>
          </section>
        )}

        {/* PREDICT */}
        {activeTab === 'predict' && (
          <section className="predict-section">
            <h2 className="section-title">Make a Prediction</h2>

            <div className="model-selector">
              <button
                className={`model-btn ${selectedModel === 'classification' ? 'active' : ''}`}
                onClick={() => setSelectedModel('classification')}
              >
                🎯 Classification
                <span>Kidney Disease Stage</span>
                <span className="feature-count">{30} features</span>
              </button>
              <button
                className={`model-btn ${selectedModel === 'regression' ? 'active' : ''}`}
                onClick={() => setSelectedModel('regression')}
              >
                📈 Regression
                <span>ACR Value Prediction</span>
                <span className="feature-count">{35} features</span>
              </button>
              <button
                className={`model-btn ${selectedModel === 'mtl' ? 'active' : ''}`}
                onClick={() => setSelectedModel('mtl')}
              >
                🧠 Multi-Task
                <span>4 Clinical Outcomes</span>
                <span className="feature-count">{30} features</span>
              </button>
            </div>

            <div className="predict-container">
              {/* Input Form */}
              <div className="input-panel">
                <div className="input-panel-header">
                  <div>
                    <h3>📊 Patient Biomarkers</h3>
                    <p className="input-desc">Enter values in standard medical units - {selectedModel.toUpperCase()} model</p>
                  </div>
                  <button
                    className="healthy-stats-btn"
                    onClick={() => {
                      setFeatures({ ...HEALTHY_BIOMARKERS });
                      setGender(1);
                      setSmoking(1);
                      setBinaryFeatures({
                        hypertension: 0, diabetes: 0, stroke: 0, heart_attack: 0,
                        heart_failure: 0, coronary_disease: 0, angina: 0,
                        kidney_weak: 0, kidney_dialysis: 0
                      });
                      setResult(null);
                    }}
                  >
                    💚 Healthy Stats
                  </button>
                </div>

                <div className="features-grid">
                  {currentFeatures.map((f) => {
                    const range = BIOMARKER_RANGES[f.name] || { min: 0, max: 1, step: 0.01, unit: '', default: 0.5 }
                    const value = features[f.name] !== undefined ? features[f.name] : range.default
                    return (
                      <div key={f.name} className="feature-input">
                        <label>{f.label} <span className="unit">({range.unit})</span></label>
                        <input
                          type="range"
                          min={range.min} max={range.max} step={range.step}
                          value={value}
                          onChange={(e) => setFeatures({ ...features, [f.name]: parseFloat(e.target.value) })}
                        />
                        <span className="feature-value">{Number.isInteger(range.step) ? value : value.toFixed(1)}</span>
                      </div>
                    )
                  })}
                </div>

                <div className="categorical-inputs">
                  <div className="cat-group">
                    <label>Gender</label>
                    <select value={gender} onChange={e => setGender(parseInt(e.target.value))}>
                      <option value={1}>Male</option>
                      <option value={2}>Female</option>
                    </select>
                  </div>
                  <div className="cat-group">
                    <label>Smoking</label>
                    <select value={smoking} onChange={e => setSmoking(parseInt(e.target.value))}>
                      <option value={1}>Never</option>
                      <option value={2}>Former</option>
                      <option value={3}>Current</option>
                    </select>
                  </div>
                </div>

                {/* Extra binary features for regression */}
                {selectedModel === 'regression' && (
                  <div className="binary-features">
                    <h4>Clinical History</h4>
                    <div className="binary-grid">
                      {REGRESSION_EXTRA_FEATURES.map(f => (
                        <label key={f.name} className="binary-checkbox">
                          <input
                            type="checkbox"
                            checked={binaryFeatures[f.name] === 1}
                            onChange={e => setBinaryFeatures({ ...binaryFeatures, [f.name]: e.target.checked ? 1 : 0 })}
                          />
                          <span>{f.label}</span>
                        </label>
                      ))}
                    </div>
                  </div>
                )}

                <button className="predict-btn" onClick={predict} disabled={loading}>
                  {loading ? '⏳ Predicting...' : '🔮 Predict'}
                </button>
              </div>

              {/* Results */}
              <div className="results-panel">
                <h3>🎯 Prediction Result</h3>
                {!result && !loading && (
                  <div className="no-result">
                    <span className="no-result-icon">🔬</span>
                    <p>Select a model and click Predict</p>
                  </div>
                )}

                {loading && (
                  <div className="loading"><div className="spinner"></div><p>Analyzing...</p></div>
                )}

                {result && !result.error && selectedModel === 'classification' && (
                  <div className="result-card">
                    <div className="result-main" style={{ backgroundColor: CLASS_COLORS[result.class_name] + '20', borderColor: CLASS_COLORS[result.class_name] }}>
                      <span className="result-class" style={{ color: CLASS_COLORS[result.class_name] }}>
                        {(result.probabilities[result.class_name] * 100).toFixed(1)}% likely to be {result.class_name}
                      </span>
                    </div>
                    <div className="probability-bars">
                      {Object.entries(result.probabilities || {}).map(([cls, prob]) => (
                        <div key={cls} className="prob-bar">
                          <span className="prob-label">{(prob * 100).toFixed(1)}% {cls}</span>
                          <div className="prob-track">
                            <div className="prob-fill" style={{ width: `${prob * 100}%`, backgroundColor: CLASS_COLORS[cls] }}></div>
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                )}

                {result && !result.error && selectedModel === 'regression' && (
                  <div className="result-card">
                    <div className="result-main" style={{ backgroundColor: CLASS_COLORS[result.risk_category] + '20', borderColor: CLASS_COLORS[result.risk_category] }}>
                      <span className="result-value">{result.acr_value}</span>
                      <span className="result-unit">mg/g ACR</span>
                    </div>
                    <div className="risk-badge" style={{ backgroundColor: CLASS_COLORS[result.risk_category] }}>{result.risk_category}</div>
                  </div>
                )}

                {result && !result.error && selectedModel === 'mtl' && (() => {
                  // Calculate metabolic syndrome diagnostic (3 out of 5 rule)
                  const metSyn = result.metabolic_syndrome || {}
                  const markerNames = {
                    waist: 'Likely high waist circumference',
                    triglycerides: 'Likely high triglycerides',
                    hdl: 'Likely low HDL cholesterol',
                    blood_pressure: 'Likely high blood pressure',
                    glucose: 'Likely high fasting glucose'
                  }
                  const elevatedList = Object.entries(metSyn)
                    .filter(([_, v]) => v > 0.65)
                    .map(([k, _]) => markerNames[k] || k)
                  const elevatedMarkers = elevatedList.length
                  const metSynRisk = elevatedMarkers >= 3 ? 'High' : 'Low'
                  const metSynLabel = elevatedMarkers >= 3 ? 'Positive for Metabolic Syndrome' : 'Low Risk'

                  return (
                    <div className="result-card mtl-result">
                      <div className="mtl-outcome">
                        <span className="mtl-label">❤️ Cardiovascular Risk</span>
                        <span className={`mtl-risk ${result.cardiovascular_disease?.risk?.toLowerCase()}`}>
                          {result.cardiovascular_disease?.risk} ({(result.cardiovascular_disease?.probability * 100).toFixed(1)}%)
                        </span>
                      </div>
                      <div className="mtl-outcome">
                        <span className="mtl-label">🏃 Metabolic Syndrome</span>
                        <span className={`mtl-risk ${metSynRisk.toLowerCase()}`}>
                          {metSynLabel}
                        </span>
                        <div className="mtl-markers">
                          {elevatedMarkers > 0 ? (
                            <>
                              <span className="markers-title">Elevated markers ({elevatedMarkers}/5):</span>
                              <ul className="marker-list">
                                {elevatedList.map(marker => (
                                  <li key={marker}>⚠️ {marker}</li>
                                ))}
                              </ul>
                            </>
                          ) : (
                            <span className="markers-good">✓ No elevated markers detected</span>
                          )}
                        </div>
                      </div>
                      <div className="mtl-outcome">
                        <span className="mtl-label">🫁 Renal (Kidney) Health Risk</span>
                        <span className={`mtl-risk ${result.kidney_dysfunction?.stage?.toLowerCase()}`}>
                          {result.kidney_dysfunction?.stage === 'Normal' && 'Normal (ACR <30 mg/g)'}
                          {result.kidney_dysfunction?.stage === 'Micro' && 'Microalbuminuria (ACR 30-300 mg/g)'}
                          {result.kidney_dysfunction?.stage === 'Macro' && 'Macroalbuminuria (ACR >300 mg/g - High Risk)'}
                        </span>
                      </div>
                      <div className="mtl-outcome">
                        <span className="mtl-label">🫀 Hepatic (Liver) Health Risk</span>
                        <span className={`mtl-risk ${result.liver_dysfunction?.risk?.toLowerCase()}`}>
                          {result.liver_dysfunction?.risk} ({(result.liver_dysfunction?.probability * 100).toFixed(1)}%)
                        </span>
                      </div>
                    </div>
                  )
                })()}

                {result && result.error && (
                  <div className="error-result"><span>⚠️ {result.error}</span></div>
                )}
              </div>
            </div>
          </section>
        )}

        {/* ABOUT */}
        {activeTab === 'about' && (
          <section className="about-section">
            <h2 className="section-title">About This Project</h2>
            <div className="about-grid">
              <div className="about-card"><span className="about-icon">🎯</span><h3>Purpose</h3><p>Predict future health risks from patient biomarkers using advanced AI, enabling early intervention and preventive care.</p></div>
              <div className="about-card"><span className="about-icon">📊</span><h3>Data Source</h3><p>NHANES Cycles 2013-2023 with 34,000+ adult records (age 20+) from the U.S. population.</p></div>
              <div className="about-card"><span className="about-icon">🧠</span><h3>Technology</h3><p>Multi-Task Learning (MTL) architecture with shared backbone, PyTorch neural networks, served via FastAPI.</p></div>
              <div className="about-card"><span className="about-icon">⚕️</span><h3>Clinical Targets</h3><p><strong>Kidney:</strong> ACR levels • <strong>CVD:</strong> Heart disease history • <strong>Metabolic:</strong> 5-component syndrome • <strong>Liver:</strong> Enzyme markers</p></div>
            </div>
            <div className="tech-stack">
              <h3>Tech Stack</h3>
              <div className="tech-badges">
                <span className="tech-badge">PyTorch</span>
                <span className="tech-badge">FastAPI</span>
                <span className="tech-badge">React</span>
                <span className="tech-badge">Python</span>
              </div>
            </div>
          </section>
        )}

        {/* MODELS */}
        {activeTab === 'models' && (
          <section className="models-section">
            <h2 className="section-title">Available Models</h2>
            <div className="models-grid">
              <div className={`model-card ${modelsAvailable.classification ? '' : 'unavailable'}`}>
                <div className="model-header"><span className="model-icon">🎯</span><h3>Classification</h3></div>
                <p className="model-desc">Predicts kidney disease stage (Normal/Micro/Macro).</p>
                <div className="model-details">
                  <div className="detail-row"><span>Features</span><span>30</span></div>
                  <div className="detail-row"><span>Classes</span><span>Normal, Micro, Macro</span></div>
                  <div className="detail-row"><span>Accuracy</span><span>90%</span></div>
                </div>
                <button className="model-use-btn" onClick={() => { setSelectedModel('classification'); setActiveTab('predict') }}>Use Model →</button>
              </div>
              <div className={`model-card ${modelsAvailable.regression ? '' : 'unavailable'}`}>
                <div className="model-header"><span className="model-icon">📈</span><h3>Regression</h3></div>
                <p className="model-desc">Predicts continuous kidney ACR value.</p>
                <div className="model-details">
                  <div className="detail-row"><span>Features</span><span>35</span></div>
                  <div className="detail-row"><span>Output</span><span>ACR (mg/g)</span></div>
                  <div className="detail-row"><span>Task</span><span>Continuous Prediction</span></div>
                </div>
                <button className="model-use-btn" onClick={() => { setSelectedModel('regression'); setActiveTab('predict') }}>Use Model →</button>
              </div>
              <div className={`model-card ${modelsAvailable.mtl ? '' : 'unavailable'}`}>
                <div className="model-header"><span className="model-icon">🧠</span><h3>Multi-Task</h3></div>
                <p className="model-desc">Predicts 4 clinical outcomes simultaneously.</p>
                <div className="model-details">
                  <div className="detail-row"><span>Features</span><span>30</span></div>
                  <div className="detail-row"><span>Outputs</span><span>CVD, Metabolic, Kidney, Liver</span></div>
                  <div className="detail-row"><span>CVD ROC-AUC</span><span>83%</span></div>
                  <div className="detail-row"><span>Liver ROC-AUC</span><span>93%</span></div>
                </div>
                <button className="model-use-btn" onClick={() => { setSelectedModel('mtl'); setActiveTab('predict') }}>Use Model →</button>
              </div>
            </div>
          </section>
        )}
      </main>

      <footer className="footer">
        <p>© 2024 Clinical Prediction Platform | Built with PyTorch & React</p>
      </footer>
    </div>
  )
}

export default App
