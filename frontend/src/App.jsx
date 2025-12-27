import { useState, useEffect } from 'react'
import './App.css'

const API_URL = 'http://localhost:8000'

// Features for each model
const MODEL_FEATURES = {
  classification: [
    { name: 'age', label: 'Age' },
    { name: 'income_ratio', label: 'Income Ratio' },
    { name: 'body_mass_index', label: 'BMI' },
    { name: 'height_cm', label: 'Height (cm)' },
    { name: 'heart_rate_bpm', label: 'Heart Rate' },
    { name: 'white_blood_cells', label: 'WBC Count' },
    { name: 'platelets_count', label: 'Platelets' },
    { name: 'hemoglobin', label: 'Hemoglobin' },
    { name: 'mcv', label: 'MCV' },
    { name: 'creatinine', label: 'Creatinine' },
    { name: 'liver_ast', label: 'Liver AST' },
    { name: 'bilirubin', label: 'Bilirubin' },
    { name: 'liver_ggt', label: 'Liver GGT' },
    { name: 'uric_acid', label: 'Uric Acid' },
    { name: 'sodium', label: 'Sodium' },
    { name: 'potassium', label: 'Potassium' },
    { name: 'cholesterol', label: 'Cholesterol' },
    { name: 'alcohol', label: 'Alcohol/Week' },
  ],
  regression: [
    { name: 'age', label: 'Age' },
    { name: 'income_ratio', label: 'Income Ratio' },
    { name: 'body_mass_index', label: 'BMI' },
    { name: 'height_cm', label: 'Height (cm)' },
    { name: 'heart_rate_bpm', label: 'Heart Rate' },
    { name: 'white_blood_cells', label: 'WBC Count' },
    { name: 'platelets_count', label: 'Platelets' },
    { name: 'hemoglobin', label: 'Hemoglobin' },
    { name: 'mcv', label: 'MCV' },
    { name: 'creatinine', label: 'Creatinine' },
    { name: 'liver_ast', label: 'Liver AST' },
    { name: 'bilirubin', label: 'Bilirubin' },
    { name: 'liver_ggt', label: 'Liver GGT' },
    { name: 'uric_acid', label: 'Uric Acid' },
    { name: 'sodium', label: 'Sodium' },
    { name: 'potassium', label: 'Potassium' },
    { name: 'cholesterol', label: 'Cholesterol' },
    { name: 'alcohol', label: 'Alcohol/Week' },
  ],
  mtl: [
    { name: 'age', label: 'Age' },
    { name: 'income_ratio', label: 'Income Ratio' },
    { name: 'body_mass_index', label: 'BMI' },
    { name: 'height_cm', label: 'Height (cm)' },
    { name: 'heart_rate_bpm', label: 'Heart Rate' },
    { name: 'white_blood_cells', label: 'WBC Count' },
    { name: 'platelets_count', label: 'Platelets' },
    { name: 'hemoglobin', label: 'Hemoglobin' },
    { name: 'mcv', label: 'MCV' },
    { name: 'creatinine', label: 'Creatinine' },
    { name: 'liver_ast', label: 'Liver AST' },
    { name: 'bilirubin', label: 'Bilirubin' },
    { name: 'liver_ggt', label: 'Liver GGT' },
    { name: 'uric_acid', label: 'Uric Acid' },
    { name: 'sodium', label: 'Sodium' },
    { name: 'potassium', label: 'Potassium' },
    { name: 'cholesterol', label: 'Cholesterol' },
    { name: 'alcohol', label: 'Alcohol/Week' },
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

  // Initialize features for selected model
  useEffect(() => {
    const modelFeatures = MODEL_FEATURES[selectedModel] || []
    const initialFeatures = {}
    modelFeatures.forEach(f => { initialFeatures[f.name] = 0.5 })
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
    const featureValues = modelFeatures.map(f => features[f.name] || 0.5)

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
                <div className="stat-card"><span className="stat-number">62%</span><span className="stat-label">Accuracy</span></div>
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
                <h3>📊 Patient Biomarkers</h3>
                <p className="input-desc">Enter normalized values (0-1 scale) - {selectedModel.toUpperCase()} model</p>

                <div className="features-grid">
                  {currentFeatures.map((f) => (
                    <div key={f.name} className="feature-input">
                      <label>{f.label}</label>
                      <input
                        type="range"
                        min="0" max="1" step="0.01"
                        value={features[f.name] || 0.5}
                        onChange={(e) => setFeatures({ ...features, [f.name]: parseFloat(e.target.value) })}
                      />
                      <span className="feature-value">{(features[f.name] || 0.5).toFixed(2)}</span>
                    </div>
                  ))}
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
                      <span className="result-class" style={{ color: CLASS_COLORS[result.class_name] }}>{result.class_name}</span>
                      <span className="result-code">Class {result.prediction}</span>
                    </div>
                    <div className="probability-bars">
                      {Object.entries(result.probabilities || {}).map(([cls, prob]) => (
                        <div key={cls} className="prob-bar">
                          <span className="prob-label">{cls}</span>
                          <div className="prob-track">
                            <div className="prob-fill" style={{ width: `${prob * 100}%`, backgroundColor: CLASS_COLORS[cls] }}></div>
                          </div>
                          <span className="prob-value">{(prob * 100).toFixed(1)}%</span>
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

                {result && !result.error && selectedModel === 'mtl' && (
                  <div className="result-card mtl-result">
                    <div className="mtl-outcome">
                      <span className="mtl-label">❤️ Cardiovascular</span>
                      <span className={`mtl-risk ${result.cardiovascular_disease?.risk?.toLowerCase()}`}>
                        {result.cardiovascular_disease?.risk} ({(result.cardiovascular_disease?.probability * 100).toFixed(1)}%)
                      </span>
                    </div>
                    <div className="mtl-outcome">
                      <span className="mtl-label">🏃 Metabolic Syndrome</span>
                      <div className="mtl-details">
                        {result.metabolic_syndrome && Object.entries(result.metabolic_syndrome).map(([k, v]) => (
                          <span key={k} className="mtl-sub">{k}: {(v * 100).toFixed(0)}%</span>
                        ))}
                      </div>
                    </div>
                    <div className="mtl-outcome">
                      <span className="mtl-label">🫘 Kidney</span>
                      <span className={`mtl-risk ${result.kidney_dysfunction?.stage?.toLowerCase()}`}>
                        {result.kidney_dysfunction?.stage}
                      </span>
                    </div>
                    <div className="mtl-outcome">
                      <span className="mtl-label">🫀 Liver</span>
                      <span className={`mtl-risk ${result.liver_dysfunction?.risk?.toLowerCase()}`}>
                        {result.liver_dysfunction?.risk} ({(result.liver_dysfunction?.probability * 100).toFixed(1)}%)
                      </span>
                    </div>
                  </div>
                )}

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
              <div className="about-card"><span className="about-icon">🎯</span><h3>Purpose</h3><p>Predict clinical outcomes from NHANES biomarkers using deep learning.</p></div>
              <div className="about-card"><span className="about-icon">📊</span><h3>Data Source</h3><p>NHANES survey with 34,000+ patient records.</p></div>
              <div className="about-card"><span className="about-icon">🧠</span><h3>Technology</h3><p>PyTorch neural networks served via FastAPI.</p></div>
              <div className="about-card"><span className="about-icon">⚕️</span><h3>Targets</h3><p>Kidney, cardiovascular, metabolic, and liver dysfunction.</p></div>
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
                  <div className="detail-row"><span>Accuracy</span><span>62.18%</span></div>
                </div>
                <button className="model-use-btn" onClick={() => { setSelectedModel('classification'); setActiveTab('predict') }}>Use Model →</button>
              </div>
              <div className={`model-card ${modelsAvailable.regression ? '' : 'unavailable'}`}>
                <div className="model-header"><span className="model-icon">📈</span><h3>Regression</h3></div>
                <p className="model-desc">Predicts continuous kidney ACR value.</p>
                <div className="model-details">
                  <div className="detail-row"><span>Features</span><span>35</span></div>
                  <div className="detail-row"><span>Output</span><span>ACR (mg/g)</span></div>
                </div>
                <button className="model-use-btn" onClick={() => { setSelectedModel('regression'); setActiveTab('predict') }}>Use Model →</button>
              </div>
              <div className={`model-card ${modelsAvailable.mtl ? '' : 'unavailable'}`}>
                <div className="model-header"><span className="model-icon">🧠</span><h3>Multi-Task</h3></div>
                <p className="model-desc">Predicts 4 clinical outcomes simultaneously.</p>
                <div className="model-details">
                  <div className="detail-row"><span>Features</span><span>30</span></div>
                  <div className="detail-row"><span>Outputs</span><span>CVD, Metabolic, Kidney, Liver</span></div>
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
