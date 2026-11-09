# PaySim Fraud Detection Capstone System

An end-to-end, production-grade fraud detection platform built on the PaySim synthetic mobile-money transaction dataset (6.36M transactions). The project features leakage-safe feature engineering, chronological validation, tuned XGBoost classification, MLflow tracking, an interactive React dashboard on Vercel, containerized FastAPI inference on Render, Evidently AI drift monitoring, Apache Kafka streaming, and PyTorch Geometric GraphSAGE.

---

## 🌐 Live Deployments

| Component | Platform | Status | URL / Link |
| :--- | :--- | :--- | :--- |
| **Interactive Dashboard** | Vercel | Active | [PaySim Fraud Intelligence Dashboard](https://pay-sim-fraud-detection-nine.vercel.app) |
| **Prediction REST API** | Render | Active | [API Root & Endpoints](https://paysim-fraud-detection-w1lm.onrender.com) |
| **Interactive API Docs** | Swagger UI | Active | [Interactive Swagger Docs (`/docs`)](https://paysim-fraud-detection-w1lm.onrender.com/docs) |
| **Service Health Check** | FastAPI | Active | [Health Probe (`/health`)](https://paysim-fraud-detection-w1lm.onrender.com/health) |

> [!NOTE]
> The backend runs on Render's free tier, which sleeps after 15 minutes of inactivity. If the service is idle, the first request will take ~45–50 seconds to complete its cold-start spin-up.

---

## 📊 Benchmark Model Performance

Trained chronologically on PaySim transactions (70% train / 15% validation / 15% test) with validation-tuned optimal decision cutoffs:

| Metric | Test Set Score | Note |
| :--- | :--- | :--- |
| **PR-AUC (Precision-Recall)** | **0.999999** | Primary metric due to 0.13% class imbalance |
| **ROC-AUC** | **0.999999** | Overall ranking discrimination capacity |
| **F1-Score** | **0.9996** | Harmonic mean of precision & recall |
| **Precision** | **1.0000 (100%)** | 0 False Positives on test holdout |
| **Recall** | **0.9992 (99.9%)** | 4,017 out of 4,020 fraud cases intercepted |
| **Tuned Decision Threshold** | **0.9848** | Tuned strictly on validation holdout to eliminate alert fatigue |

#### Holdout Confusion Matrix ($N = 415,562$ test transactions):
- **True Negatives (Legitimate passed):** 411,542
- **False Positives (False alarms):** 0
- **False Negatives (Missed fraud):** 3
- **True Positives (Fraud intercepted):** 4,017

---

## 1. Project Architecture & Directory Layout

```text
├── data/
│   └── raw/
│       └── paysim.csv            <- Standard raw dataset location 
├── dataset/
│   └── paysim.csv            <- Local raw PaySim dataset (6.36M records)
├── frontend/                     <- Interactive React + Vite Dashboard
│   ├── index.html
│   ├── package.json
│   ├── vite.config.js
│   ├── .env.example              <- Environment config template
│   └── src/
│       ├── main.jsx
│       ├── App.jsx               <- Transaction simulator & dynamic risk gauge
│       └── index.css             <- Dark glassmorphism design system
├── notebooks/
│   └── paysim_fraud_detection.ipynb <- Comprehensive end-to-end exploratory & modeling notebook
├── models/
│   ├── feature_schema.json       <- Ordered model feature names
│   ├── decision_threshold.json   <- Optimal probability cutoff tuned on validation set
│   └── xgboost_fraud.joblib      <- Serialized production XGBoost model artifact
├── reports/
│   ├── data_quality.json         <- Column checks, missing values, duplicates report
│   ├── class_balance.png         <- Visualization of class distribution
│   ├── fraud_by_transaction_type.csv <- Fraud counts & rates per transaction type
│   ├── confusion_matrix.png      <- Test set confusion matrix plot
│   └── metrics.json              <- PR-AUC, ROC-AUC, precision, recall, F1 results
├── src/fraud_capstone/
│   ├── __init__.py
│   ├── config.py                 <- Central configuration and auto-detection
│   ├── data.py                   <- Loading, schema validation & data quality summary
│   ├── eda.py                    <- Imbalance metrics, group-by summaries & plots
│   ├── features.py               <- Leakage-safe feature extraction & schema alignment
│   ├── split.py                  <- Chronological (70/15/15) train/val/test splitting
│   ├── evaluate.py               <- Metrics, threshold search, confusion matrix
│   ├── train.py                  <- End-to-end training pipeline CLI
│   ├── predict.py                <- Scoring logic & predictor class
│   ├── api.py                    <- FastAPI REST prediction service with CORS
│   ├── drift_report.py           <- Evidently AI feature drift report generator
│   ├── kafka_producer.py         <- Kafka transaction streaming & consumer scoring
│   ├── graph_data.py             <- Temporal DAG construction for transactions
│   ├── graph_model.py            <- GraphSAGE PyTorch Geometric architecture
│   └── graph_train.py            <- GNN training & chronological test evaluation
├── tests/
│   ├── test_features.py          <- Leakage prevention & mathematical derivation tests
│   ├── test_split.py             <- Chronological ordering & partition size tests
│   └── test_api.py               <- FastAPI health, scoring, and error handling tests
├── pyproject.toml                <- Project dependencies & packaging
├── Dockerfile                    <- Production container configuration for FastAPI
├── docker-compose.yml            <- Local Kafka & Zookeeper configuration
└── k8s-deployment.yaml           <- Kubernetes deployment & service manifest
```

---

## 2. Environment Setup & Installation

### Prerequisites
- Python 3.11 or 3.12
- [uv](https://github.com/astral-sh/uv) (fast Python package manager)
- Node.js (v18+) & npm (for React dashboard)

### Python Dependencies
Install packages cleanly using `uv`:

```bash
# Core dependencies + development & testing tools
uv sync --extra dev

# Optional extensions
uv sync --extra tracking     # MLflow tracking
uv sync --extra monitoring   # Evidently AI data drift
uv sync --extra streaming    # Kafka streaming
uv sync --extra graph        # PyTorch & PyTorch Geometric
uv sync --extra all          # All dependencies
```

---

## 3. Usage & Execution Instructions

### A. Data Setup
The PaySim dataset is supported in either of the following locations:
- `dataset/paysim.csv` *(auto-detected in project)*
- `data/raw/paysim.csv`

The configuration automatically resolves whichever location is present. You can also specify an explicit path via `--data`.

### B. Training Pipeline
Run the full training pipeline, which validates data quality, generates EDA reports, splits chronologically, engineers features, trains XGBoost, selects the validation threshold, evaluates the test set, and serializes artifacts:

```powershell
uv run python -m fraud_capstone.train
```

Quick iteration mode with row cap:
```powershell
uv run python -m fraud_capstone.train --data "./dataset/paysim.csv" --nrows 100000
```

With MLflow experiment tracking:
```powershell
uv run python -m fraud_capstone.train --data "./dataset/paysim.csv" --tracking
uv run mlflow ui --host 127.0.0.1 --port 5000
```

### C. Running Automated Tests
Run unit, feature leakage, and API integration tests:
```powershell
uv run pytest
```

### D. Starting the FastAPI Prediction Service (Local)
Launch the REST API:
```powershell
uv run uvicorn fraud_capstone.api:app --reload --host 127.0.0.1 --port 8000
```
- Interactive Swagger docs: `http://127.0.0.1:8000/docs`
- Health check: `GET http://127.0.0.1:8000/health`
- Real-time scoring: `POST http://127.0.0.1:8000/score`

#### Sample Inference Request:
```bash
curl -X POST "http://127.0.0.1:8000/score" \
     -H "Content-Type: application/json" \
     -d '{
       "step": 1,
       "type": "TRANSFER",
       "amount": 181.0,
       "nameOrig": "C1231006815",
       "oldbalanceOrg": 181.0,
       "newbalanceOrig": 0.0,
       "nameDest": "C553264065",
       "oldbalanceDest": 0.0,
       "newbalanceDest": 0.0
     }'
```

#### Sample Response:
```json
{
  "fraud_probability": 0.9421,
  "is_fraud_prediction": 1,
  "decision_threshold": 0.9848,
  "status": "success"
}
```

### E. Starting the React Frontend (Local)
```powershell
cd frontend
npm install
npm run dev
```

Open `http://localhost:3000` (or `http://localhost:5173`) in your browser to access:
- **Transaction Simulator**: Real-time custom simulation with 1-click presets (*Suspicious Account Drain*, *Suspicious Cash Out*, *Standard Merchant Payment*).
- **Engine Health Monitor**: Live connection badge verifying the backend connection and model artifact presence.
- **Dynamic Risk Gauge**: Displays real-time calibrated fraud risk probability, binary verdict tag, and tuned decision threshold.

---

## 4. Advanced Extensions

### Evidently Drift Monitoring
Generate an interactive HTML report comparing reference and current feature distributions:
```powershell
uv run python -m fraud_capstone.drift_report --data "./dataset/paysim.csv"
```

### Kafka Streaming Simulation
1. Start Kafka and Zookeeper:
   ```bash
   docker-compose up -d
   ```
2. Start fraud detection consumer:
   ```powershell
   uv run python -m fraud_capstone.kafka_producer --mode consume
   ```
3. Replay transactions via producer:
   ```powershell
   uv run python -m fraud_capstone.kafka_producer --mode produce --limit 100
   ```

### Graph Neural Network (GraphSAGE)
Construct temporal bipartite graphs and train GraphSAGE:
```powershell
uv run python -m fraud_capstone.graph_train --data "./dataset/paysim.csv" --sample 10000
```

### Docker & Kubernetes Deployment
Build and run the container locally:
```bash
docker build -t paysim-fraud-api:latest .
docker run -p 8000:8000 -e PORT=8000 paysim-fraud-api:latest
```

Deploy to Kubernetes / Minikube:
```bash
kubectl apply -f k8s-deployment.yaml
```

---

## 5. Key Learnings & Engineering Insights

1. **Target Leakage Prevention in Financial Streams:**
   - Avoiding future information leakage is crucial in payment networks. Standard random k-fold cross-validation inflates performance metrics due to temporal correlations and repeated customer identities. Chronological splitting (70% train / 15% validation / 15% test) is essential to evaluate models under realistic out-of-time conditions.
   - `isFlaggedFraud` was identified as an existing heuristic rule and explicitly excluded from training features to avoid target leakage.

2. **Severe Class Imbalance & Metric Alignment:**
   - With fraud occurrences at ~0.13%, optimizing for raw Accuracy or ROC-AUC is misleading. The model was optimized for **PR-AUC (Precision-Recall AUC)** and tuned using threshold search on the chronological validation set to maximize recall while bounding false positive alert fatigue.
   - Tuning the threshold on the validation set produced an optimal threshold of **0.9848**, yielding a precision of 100% and recall of 99.9% on the test holdout set.

3. **Cloud Production & Cold-Start Resilience:**
   - Modern PEP 517 build backends (Hatchling) require `README.md` and package sources to exist during `uv pip install -e .`. Proper Docker layer ordering was implemented to ensure container reproducibility across local Docker and Render.
   - Decoupled client-side configuration (`import.meta.env.VITE_API_URL`) allows seamless transitions between local environments and cloud deployments without hardcoded network references.

4. **Graph Structures Complement Tabular Classifiers:**
   - Fraud syndicates frequently hop across synthetic accounts (mule networks). Temporal bipartite graphs (Origin $\rightarrow$ Destination) modeled with GraphSAGE capture structural network anomalies that isolated tabular transaction rows alone cannot detect.

---

## 6. Future Improvements & Roadmap

- [ ] **Dual Tabular-Graph Ensemble:** Feed GraphSAGE node embeddings directly into the XGBoost classifier for combined tabular-relational scoring.
- [ ] **Real-time Feature Store:** Implement Redis / Feast to maintain rolling window aggregates (e.g., *transaction count in last 1 hour*, *mean amount over 24h*) during live streaming inference.
- [ ] **Automated Retraining Pipeline:** Set up an Airflow / GitHub Actions workflow triggered by Evidently AI drift alerts to retrain models on new transaction windows automatically.
- [ ] **Adaptive Dynamic Thresholding:** Implement risk-tiered thresholding that adjusts dynamically depending on transaction volume or merchant category risk profile.
- [ ] **Explainability in Dashboard:** Integrate SHAP waterfall plots into the React frontend to explain to fraud analysts *why* a particular transaction received a high risk score.
