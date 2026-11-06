import React, { useState, useEffect } from 'react'
import { 
  ShieldAlert, 
  ShieldCheck, 
  Activity, 
  Zap, 
  RefreshCw, 
  RotateCcw,
  ArrowRightLeft, 
  Wallet, 
  TrendingUp, 
  AlertTriangle 
} from 'lucide-react'

const TRANSACTION_TYPES = ['TRANSFER', 'CASH_OUT', 'PAYMENT', 'CASH_IN', 'DEBIT']

const DEFAULT_FORM_DATA = {
  step: 1,
  type: 'TRANSFER',
  amount: 0.0,
  nameOrig: '',
  oldbalanceOrg: 0.0,
  newbalanceOrig: 0.0,
  nameDest: '',
  oldbalanceDest: 0.0,
  newbalanceDest: 0.0,
}

const PRESETS = [
  {
    name: 'Suspicious Account Drain (Transfer)',
    data: {
      step: 14,
      type: 'TRANSFER',
      amount: 181000.0,
      nameOrig: 'C1231006815',
      oldbalanceOrg: 181000.0,
      newbalanceOrig: 0.0,
      nameDest: 'C553264065',
      oldbalanceDest: 0.0,
      newbalanceDest: 0.0,
    }
  },
  {
    name: 'Suspicious Cash Out',
    data: {
      step: 42,
      type: 'CASH_OUT',
      amount: 250000.0,
      nameOrig: 'C981240192',
      oldbalanceOrg: 250000.0,
      newbalanceOrig: 0.0,
      nameDest: 'M871239121',
      oldbalanceDest: 5000.0,
      newbalanceDest: 5000.0,
    }
  },
  {
    name: 'Standard Merchant Payment',
    data: {
      step: 1,
      type: 'PAYMENT',
      amount: 9839.64,
      nameOrig: 'C1231006815',
      oldbalanceOrg: 170136.0,
      newbalanceOrig: 160296.36,
      nameDest: 'M1979787155',
      oldbalanceDest: 0.0,
      newbalanceDest: 0.0,
    }
  }
]

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://127.0.0.1:8000'

export default function App() {
  const [formData, setFormData] = useState(PRESETS[0].data)
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState(null)
  const [error, setError] = useState(null)
  const [backendHealth, setBackendHealth] = useState({ status: 'checking', model_artifact_present: false })

  // Check backend health
  const checkHealth = async () => {
    try {
      const res = await fetch(`${API_BASE_URL}/health`)
      if (res.ok) {
        const data = await res.json()
        setBackendHealth(data)
      } else {
        setBackendHealth({ status: 'degraded', model_artifact_present: false })
      }
    } catch {
      setBackendHealth({ status: 'offline', model_artifact_present: false })
    }
  }

  useEffect(() => {
    checkHealth()
  }, [])

  const handleChange = (e) => {
    const { name, value } = e.target
    setFormData(prev => ({
      ...prev,
      [name]: ['step'].includes(name) 
        ? parseInt(value) || 0 
        : ['amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest'].includes(name)
        ? parseFloat(value) || 0
        : value
    }))
  }

  const handleScore = async (e) => {
    e.preventDefault()
    setLoading(true)
    setError(null)

    try {
      const response = await fetch(`${API_BASE_URL}/score`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(formData)
      })

      if (!response.ok) {
        const errData = await response.json().catch(() => ({}))
        throw new Error(errData.detail || `Server returned status ${response.status}`)
      }

      const data = await response.json()
      setResult(data)
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  const applyPreset = (preset) => {
    setFormData(preset.data)
    setResult(null)
    setError(null)
  }

  const handleReset = () => {
    setFormData(DEFAULT_FORM_DATA)
    setResult(null)
    setError(null)
    checkHealth()
  }

  const isFraud = result?.is_fraud_prediction === 1
  const probPercent = result ? (result.fraud_probability * 100).toFixed(1) : 0

  return (
    <div>
      {/* Top Navigation */}
      <header className="header">
        <div className="header-container">
          <div className="logo-group">
            <div className="logo-badge">
              <Zap size={22} color="#ffffff" />
            </div>
            <div>
              <h1 className="system-title">PaySim Fraud Intelligence</h1>
              <p style={{ fontSize: '0.75rem', color: 'var(--text-dim)' }}>
                Real-Time ML Decision Engine • XGBoost
              </p>
            </div>
          </div>

          <div style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
            <div className="status-badge">
              <span className={`status-dot ${backendHealth.status === 'healthy' ? 'healthy' : 'degraded'}`} />
              <span style={{ color: backendHealth.status === 'healthy' ? '#10b981' : '#f59e0b' }}>
                API: {backendHealth.status.toUpperCase()}
              </span>
            </div>
            <button className="btn-secondary" onClick={handleReset} title="Reset all input values and results">
              <RotateCcw size={14} /> Refresh / Reset
            </button>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="dashboard-grid">
        {/* Transaction Input Form */}
        <section className="glass-panel">
          <div className="panel-header">
            <div className="panel-title">
              <ArrowRightLeft size={18} color="var(--accent-cyan)" />
              <span>Simulate Transaction</span>
            </div>
            <div className="preset-chips">
              {PRESETS.map((p, idx) => (
                <button 
                  key={idx} 
                  type="button" 
                  className="chip-btn" 
                  onClick={() => applyPreset(p)}
                >
                  {p.name}
                </button>
              ))}
            </div>
          </div>

          <form onSubmit={handleScore} className="form-body">
            <div className="form-grid-3">
              <div className="input-group">
                <label className="input-label">Step (Hour)</label>
                <input 
                  type="number" 
                  name="step" 
                  value={formData.step} 
                  onChange={handleChange} 
                  min="1" 
                  className="input-control mono" 
                  required 
                />
              </div>

              <div className="input-group">
                <label className="input-label">Transaction Type</label>
                <select 
                  name="type" 
                  value={formData.type} 
                  onChange={handleChange} 
                  className="input-control"
                >
                  {TRANSACTION_TYPES.map(t => (
                    <option key={t} value={t}>{t}</option>
                  ))}
                </select>
              </div>

              <div className="input-group">
                <label className="input-label">Amount ($)</label>
                <input 
                  type="number" 
                  step="any" 
                  name="amount" 
                  value={formData.amount} 
                  onChange={handleChange} 
                  min="0" 
                  className="input-control mono" 
                  required 
                />
              </div>
            </div>

            {/* Sender / Origin Details */}
            <div style={{ borderTop: '1px solid var(--border-color)', paddingTop: '1rem' }}>
              <p style={{ fontSize: '0.8rem', fontWeight: 600, color: 'var(--text-muted)', marginBottom: '0.75rem' }}>
                ORIGIN ACCOUNT DETAILS
              </p>
              <div className="form-grid-3">
                <div className="input-group">
                  <label className="input-label">Sender ID</label>
                  <input 
                    type="text" 
                    name="nameOrig" 
                    value={formData.nameOrig} 
                    onChange={handleChange} 
                    className="input-control mono" 
                    required 
                  />
                </div>
                <div className="input-group">
                  <label className="input-label">Old Balance Origin ($)</label>
                  <input 
                    type="number" 
                    step="any" 
                    name="oldbalanceOrg" 
                    value={formData.oldbalanceOrg} 
                    onChange={handleChange} 
                    className="input-control mono" 
                    required 
                  />
                </div>
                <div className="input-group">
                  <label className="input-label">New Balance Origin ($)</label>
                  <input 
                    type="number" 
                    step="any" 
                    name="newbalanceOrig" 
                    value={formData.newbalanceOrig} 
                    onChange={handleChange} 
                    className="input-control mono" 
                    required 
                  />
                </div>
              </div>
            </div>

            {/* Recipient / Destination Details */}
            <div style={{ borderTop: '1px solid var(--border-color)', paddingTop: '1rem' }}>
              <p style={{ fontSize: '0.8rem', fontWeight: 600, color: 'var(--text-muted)', marginBottom: '0.75rem' }}>
                DESTINATION ACCOUNT DETAILS
              </p>
              <div className="form-grid-3">
                <div className="input-group">
                  <label className="input-label">Recipient ID</label>
                  <input 
                    type="text" 
                    name="nameDest" 
                    value={formData.nameDest} 
                    onChange={handleChange} 
                    className="input-control mono" 
                    required 
                  />
                </div>
                <div className="input-group">
                  <label className="input-label">Old Balance Dest ($)</label>
                  <input 
                    type="number" 
                    step="any" 
                    name="oldbalanceDest" 
                    value={formData.oldbalanceDest} 
                    onChange={handleChange} 
                    className="input-control mono" 
                    required 
                  />
                </div>
                <div className="input-group">
                  <label className="input-label">New Balance Dest ($)</label>
                  <input 
                    type="number" 
                    step="any" 
                    name="newbalanceDest" 
                    value={formData.newbalanceDest} 
                    onChange={handleChange} 
                    className="input-control mono" 
                    required 
                  />
                </div>
              </div>
            </div>

            <div style={{ display: 'flex', gap: '0.75rem', marginTop: '0.5rem' }}>
              <button type="submit" className="btn-primary" disabled={loading} style={{ flex: 1 }}>
                {loading ? (
                  <>
                    <RefreshCw className="spin" size={18} /> Analyzing Transaction...
                  </>
                ) : (
                  <>
                    <Activity size={18} /> Run ML Fraud Assessment
                  </>
                )}
              </button>
              <button 
                type="button" 
                className="btn-secondary" 
                onClick={handleReset} 
                disabled={loading}
                title="Reset form values and verdict"
                style={{ padding: '0 1.25rem' }}
              >
                <RotateCcw size={16} /> Reset
              </button>
            </div>
          </form>
        </section>

        {/* Prediction Results Display */}
        <section className="glass-panel">
          <div className="panel-header">
            <div className="panel-title">
              <TrendingUp size={18} color="var(--accent-primary)" />
              <span>Assessment Verdict</span>
            </div>
          </div>

          <div className="result-container">
            {error && (
              <div style={{
                background: 'rgba(239, 68, 68, 0.1)',
                border: '1px solid var(--fraud-red)',
                padding: '1rem',
                borderRadius: 'var(--radius-sm)',
                color: '#fca5a5',
                fontSize: '0.875rem',
                display: 'flex',
                gap: '0.75rem',
                alignItems: 'center'
              }}>
                <AlertTriangle size={20} />
                <div>{error}</div>
              </div>
            )}

            {!result && !error && (
              <div style={{
                textAlign: 'center',
                padding: '3rem 1rem',
                color: 'var(--text-muted)'
              }}>
                <Wallet size={48} style={{ opacity: 0.3, marginBottom: '1rem' }} />
                <p style={{ fontWeight: 600 }}>No transaction analyzed yet</p>
                <p style={{ fontSize: '0.8rem', color: 'var(--text-dim)', marginTop: '0.25rem' }}>
                  Select a preset or enter transaction parameters on the left to evaluate fraud risk.
                </p>
              </div>
            )}

            {result && (
              <>
                <div className={`risk-meter-card ${isFraud ? 'fraud' : 'legit'}`}>
                  <div className={`verdict-tag ${isFraud ? 'fraud' : 'legit'}`}>
                    {isFraud ? <ShieldAlert size={24} /> : <ShieldCheck size={24} />}
                    {isFraud ? 'FRAUD SUSPICION DETECTED' : 'TRANSACTION LEGITIMATE'}
                  </div>

                  <div className="probability-score mono">
                    {probPercent}%
                  </div>
                  <p style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>
                    Calculated Fraud Risk Probability
                  </p>

                  <div className="progress-bar-track">
                    <div 
                      className={`progress-bar-fill ${isFraud ? 'fraud' : 'legit'}`}
                      style={{ width: `${Math.min(Math.max(result.fraud_probability * 100, 3), 100)}%` }}
                    />
                  </div>
                </div>

                {/* Score breakdown metrics */}
                <div style={{
                  display: 'grid',
                  gridTemplateColumns: '1fr 1fr',
                  gap: '1rem',
                  marginTop: '0.5rem'
                }}>
                  <div style={{
                    background: 'rgba(255, 255, 255, 0.03)',
                    border: '1px solid var(--border-color)',
                    padding: '0.85rem',
                    borderRadius: 'var(--radius-sm)'
                  }}>
                    <span style={{ fontSize: '0.75rem', color: 'var(--text-dim)' }}>DECISION THRESHOLD</span>
                    <p className="mono" style={{ fontSize: '1.1rem', fontWeight: 700, marginTop: '0.2rem' }}>
                      {result.decision_threshold.toFixed(4)}
                    </p>
                  </div>

                  <div style={{
                    background: 'rgba(255, 255, 255, 0.03)',
                    border: '1px solid var(--border-color)',
                    padding: '0.85rem',
                    borderRadius: 'var(--radius-sm)'
                  }}>
                    <span style={{ fontSize: '0.75rem', color: 'var(--text-dim)' }}>RAW PROBABILITY</span>
                    <p className="mono" style={{ fontSize: '1.1rem', fontWeight: 700, marginTop: '0.2rem' }}>
                      {result.fraud_probability.toFixed(6)}
                    </p>
                  </div>
                </div>
              </>
            )}
          </div>
        </section>
      </main>
    </div>
  )
}
