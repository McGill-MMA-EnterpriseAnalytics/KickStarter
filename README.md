# Kickstarter Success Prediction

A data science project to predict Kickstarter campaign success using advanced machine learning techniques.

## Project Overview

This project analyzes Kickstarter campaign data to develop models that predict campaign success. The insights and models can help creators optimize their campaigns for better funding outcomes.

## Key Project Features

- Data preprocessing and advanced feature engineering
- Machine learning model training with hyperparameter optimization
- Model interpretation and explainability analysis
- MLOps implementation with CI/CD, Docker, and MLflow
- API for model serving and predictions


## Project Structure

```
├── data/                  # Data files (raw and processed)
├── notebooks/             # Jupyter notebooks for exploration and analysis
├── src/                   # Source code
│   ├── data/              # Data processing scripts
│   ├── features/          # Feature engineering code
│   ├── models/            # Model training and evaluation
│   ├── visualization/     # Data visualization code
│   └── utils/             # Utility functions
├── tests/                 # Unit and integration tests
├── docker/                # Docker configuration
├── mlflow/                # MLflow tracking and experiments
├── api/                   # API for model serving
├── docs/                  # Documentation
├── .github/               # GitHub Actions workflows
├── requirements.txt       # Project dependencies
├── setup.py               # Package installation
└── README.md              # Project documentation
```

```
├── .github
│   └── workflows
│       └── ci-cd.yml             # GitHub Actions pipeline
├── Docker
│   ├── Dockerfile.train          # Training image
│   └── Dockerfile.infer          # Inference image
├── Data
│   └── kickstarter_final_processed.csv  # Preprocessed CSV (not committed if large; see .gitignore)
├── Best_Model
│   └── best_pipeline.pkl         # Saved sklearn pipeline
├── Models                        # Saved trained models
├── src
│   ├── __init__.py
│   ├── data_loader.py            # CSV ingest & train/test split
│   ├── preprocessing.py          # Leakage removal, imputation, feature engineering
│   ├── feature_selection.py      # RF importances, ANOVA, RFECV
│   ├── tuning.py                 # Optuna hyperparameter search
│   ├── modeling.py               # Train/evaluate baselines, MLflow logging
│   ├── evaluation.py             # Build & plot comparison DataFrames
│   ├── fairness.py               # Fairlearn metrics by group
│   ├── explainability.py         # Feature importance, SHAP, learning curves
│   ├── drift_detection.py        # Evidently.ai drift reporting
│   ├── pipeline.py               # End-to-end orchestration
│   └── inference.py              # FastAPI prediction service
├── tests
│   ├── test_smoke.py             # Smoke tests for imports & /predict
│   └── test_inference_demo.py    # Demo tests for load_pipeline & demo_prediction
├── selected_features.txt         # List of features used by pipeline
├── requirement.txt              # Pin all Python dependencies
├── setup.py                      # Package metadata & entry points
├── README.md                     # Project overview & instructions
└── .gitignore                    # e.g. data/, models/, mlruns/, __pycache__/
├── mlruns                        #ml flows experiments

```

## Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/username/kickstarter-prediction.git
cd kickstarter-prediction
pip install -r requirements.txt
```
---

## Dataset

The dataset contains information about 15,215 Kickstarter projects with 46 features, including:
- Project metadata (name, goal, pledged amount, category, etc.)
- Temporal information (launch date, deadline, etc.)
- Campaign characteristics (staff pick status, video presence, etc.)

### Data Preprocessing & Feature Engineering 

1. **Leakage Prevention**  
   - Dropped post-outcome fields (pledged amount, backers, timestamps)  
   - Logged all removals via MLflow  
2. **Missing Data**  
   - Numeric: KNN imputation (k=5)  
   - Categorical: Mode imputation for `main_category`  
3. **Datetime Features**  
   - Parsed `created_at`, `launched_at`, `deadline`  
   - Engineered duration, preparation time, weekend flag, cyclic time features  
4. **Goal-Related Features**  
   - Goal/day, log-goal, goal percentile bins, USD-adjusted(goal)  
5. **Encoding**  
   - One-hot for low-cardinality (<10 values)  
   - Smoothed target encoding for high-cardinality (category, subcategory, country)  
   - Entity embeddings (dim = min(50, √unique_vals)) for selected categorical features  
6. **Dimensionality Reduction**  
   - PCA on standardized numeric features  
   - Retained 95% variance → reduced 85 → 52 components  
7. **Final Cleanup**  
   - Converted datetime to days since 2010-01-01  
   - Dropped original category columns  
   - Median imputation for remaining missing bins  

---

## Model Training

The model training phase transforms our engineered features into predictive insights by (1) splitting the data into training and test sets, (2) selecting the most relevant features, (3) fitting several baseline algorithms, and (4) tuning hyperparameters to maximize generalization performance. We leverage stratified sampling, standard scaling, and both linear and tree-based methods to build a robust suite of models, ultimately selecting the optimized XGBoost as our production-ready solution. 

### Detailed Steps

1. **Train/Test Split**  
   - Perform an 80/20 stratified split on the full dataset (`random_state=42`) to preserve class balance.  
   - Apply `StandardScaler` to numeric features, fitting on the training set and transforming both splits. 

2. **Feature Selection**  
   - **Univariate Ranking**: Use `SelectKBest` with ANOVA F-test to score features by class separation.  
   - **Recursive Elimination**: Apply `RFECV` (recursive feature elimination with cross-validation) to identify an optimal subset.  
   - **Tree-Based Validation**: Compute Random Forest feature importances to confirm and refine the selection.

3. **Baseline Model Training**  
   - Fit four core algorithms on the selected features:  
     - Logistic Regression  
     - Random Forest  
     - XGBoost  
     - Multi-Layer Perceptron (MLP)  
   - Record training time and key metrics (Accuracy, Precision, Recall, F1, ROC AUC, Average Precision). 

4. **Hyperparameter Optimization**  
   - Use Optuna with a TPE sampler, running 20 trials per model to optimize ROC AUC.  
   - Define search spaces for each algorithm (e.g., tree depths, learning rates, regularization penalties).  
   - Automatically log all trials and best parameters to MLflow for reproducibility.
     
5. **Optimized Model Evaluation**  
   - Retrain each algorithm with its best hyperparameters on the full training set.  
   - Evaluate on the held-out test set, emphasizing ROC AUC alongside Precision/Recall trade-offs.  
   - Identify the top performer (optimized XGBoost with ROC AUC ≈ 0.91) for deployment.
    
6. **Model Persistence**  
   - Serialize the chosen model pipeline (`best_pipeline.pkl`) and the list of selected features (`selected_features.txt`).  
   - Store artifacts in MLflow under the “ModelTraining” experiment for versioning and auditability. 

---

## Model Evaluation

Model evaluation quantifies how well our algorithms generalize to unseen data and informs model selection for deployment. We assess both baseline and hyperparameter-optimized versions of Logistic Regression, Random Forest, XGBoost, and MLP using multiple metrics—Accuracy, Precision, Recall, F1 Score, ROC AUC, and Average Precision—on the held-out test set. In-depth comparisons highlight the benefits of tuning and guide our choice of the final model. :contentReference[oaicite:0]{index=0}&#8203;:contentReference[oaicite:1]{index=1}

### Evaluation Metrics

- **Accuracy**: Overall fraction of correct predictions  
- **Precision**: Fraction of positive predictions that are true positives  
- **Recall**: Fraction of actual positives correctly identified  
- **F1 Score**: Harmonic mean of Precision and Recall  
- **ROC AUC**: Area under the Receiver Operating Characteristic curve—measures separability  
- **Average Precision**: Area under the Precision–Recall curve—sensitive to class imbalance :contentReference[oaicite:2]{index=2}&#8203;:contentReference[oaicite:3]{index=3}

### Baseline vs. Optimized Performance

- **Logistic Regression**  
  - Baseline ROC AUC: 0.9074, F1: 0.8382  
  - Optimized ROC AUC: 0.9077 (+0.0003), F1: 0.8393 :contentReference[oaicite:4]{index=4}&#8203;:contentReference[oaicite:5]{index=5}  

- **Random Forest**  
  - Baseline ROC AUC: 0.9059, F1: 0.8375  
  - Optimized ROC AUC: 0.9087 (+0.0028), F1: 0.8396 :contentReference[oaicite:6]{index=6}&#8203;:contentReference[oaicite:7]{index=7}  

- **XGBoost**  
  - Baseline ROC AUC: 0.8986, F1: 0.8284  
  - Optimized ROC AUC: 0.9105 (+0.0119), F1: 0.8335 :contentReference[oaicite:8]{index=8}&#8203;:contentReference[oaicite:9]{index=9}  

- **MLP Classifier**  
  - Baseline ROC AUC: 0.8804, F1: 0.8153  
  - Optimized ROC AUC: 0.9069 (+0.0265), F1: 0.8338 :contentReference[oaicite:10]{index=10}&#8203;:contentReference[oaicite:11]{index=11}  

### Model Selection

The optimized XGBoost model achieves the highest ROC AUC (0.9105) and demonstrates balanced Precision (0.8456) and Recall (0.8217), making it the preferred candidate for production. Its substantial AUC gain from hyperparameter tuning underscores its ability to capture non-linear feature interactions effectively. :contentReference[oaicite:12]{index=12}&#8203;:contentReference[oaicite:13]{index=13}

### Calibration & Robustness

- **Calibration**: Reliability diagrams confirm well-calibrated probability outputs for the selected XGBoost model, ensuring reliable confidence estimates.  
- **Cross-Validation Stability**: Low variance in ROC AUC across folds (< 0.005) indicates consistent performance.  

---
### Ethical AI & Fairness
To ensure our final XGBoost model operates fairly across different user groups, we conducted a fairness analysis using country_encoded as the protected attribute. Leveraging Fairlearn, we computed group-specific metrics such as accuracy and selection rate, alongside global fairness metrics including Demographic Parity Difference and Equalized Odds Difference. The results indicated no substantial performance disparities across country groups, with fairness metrics falling within commonly accepted thresholds (≤ 0.1). These findings suggest that the model does not systematically favor or disadvantage any specific country group, supporting its ethical application in the Kickstarter prediction context.

### Explainability & Fairness

- **SHAP Analysis**  
  - Top predictors: `usd_goal_real`, `campaign_duration`, `launch_month`  
  - Dependence plots reveal non-linear effects (e.g. optimal campaign length ≈ 30–45 days)  
- **Fairness Evaluation**  
  - Accuracy gap: staff-picked vs. non-staff-picked campaigns (~13%)  
  - Goal-size disparity: Demographic Parity Δ up to 0.74  
  - Category disparity: Equalized Odds Δ ≈ 0.95  
  - Mitigation: reweighting, fairness-aware post-processing  
- **Additional Methods**  
  - LIME & ELI5 Permutation Importance confirms feature consistency  

---


### MLOps & Deployment

To ensure reliable, repeatable, and scalable delivery of our Kickstarter Success Prediction pipeline, we operationalize every stage—from model training to serving and monitoring—using containerization, CI/CD, and automated drift detection. This MLOps framework guarantees reproducibility, rapid iteration, and proactive maintenance in production. :contentReference[oaicite:0]{index=0}&#8203;:contentReference[oaicite:1]{index=1}

#### 1. Dockerized Pipelines  
- **Training Image** (`docker/train/Dockerfile`)  
  - **Base**: `python:3.10-slim`  
  - **Dependencies**: system packages + `requirements.txt`  
  - **Copy**: `src/` directory, config files, and preprocessed CSVs  
  - **Entrypoint**:  
    ```bash
    python src/pipeline.py --config config/train.yaml
    ```  
  - **Outputs**:  
    - `models/best_pipeline.pkl`  
    - `selected_features.txt`  
    These artifacts are written to a mounted host volume and logged to MLflow. :contentReference[oaicite:2]{index=2}&#8203;:contentReference[oaicite:3]{index=3}  

- **Inference Image** (`docker/infer/Dockerfile`)  
  - **Base**: `python:3.10-slim`  
  - **Dependencies**: system packages + `requirements.txt`  
  - **Copy**:  
    - `src/inference.py` (FastAPI service)  
    - Saved model (`best_pipeline.pkl`)  
    - Feature list (`selected_features.txt`)  
  - **Environment Variables**:  
    - `MODEL_PATH=/app/models/best_pipeline.pkl`  
    - `FEATURES_PATH=/app/models/selected_features.txt`  
  - **Expose**: port `8000`  
  - **Command**:  
    ```bash
    uvicorn src.inference:app --host 0.0.0.0 --port 8000
    ```  
  - **Health Check**:  
    - Endpoint: `GET /healthz` returns `200 OK` when service is ready. :contentReference[oaicite:4]{index=4}&#8203;:contentReference[oaicite:5]{index=5}  

#### 2. CI/CD with GitHub Actions  
- **Triggers**:  
  - `push` and `pull_request` on the `main` branch  
  - `workflow_dispatch` for manual runs  
- **Jobs**:  
  1. **Lint**  
     - Tool: `flake8 src/`  
     - Enforces PEP 8 compliance and style rules  
  2. **Tests**  
     - Frameworks: `pytest`, `HTTPX`  
     - **Smoke Tests**: Basic import checks and `/predict` validation  
     - **Demo Tests**: End-to-end inference with sample payloads  
  3. **Build & Publish Docker Images**  
     - Build `train` and `infer` images  
     - Tag with Git commit SHA (`${{ github.sha }}`)  
     - Push to Docker Hub or GitHub Container Registry  
  4. **(Optional) Deployment**  
     - Deploy `infer` image to Kubernetes via Helm  
     - Supports Canary or Blue/Green rollout  
- **Secrets & Config**:  
  - `DOCKERHUB_USERNAME`, `DOCKERHUB_TOKEN`  
  - `KUBE_CONFIG` for cluster access  
- **Artifacts**:  
  - Test reports (JUnit XML)  
  - Coverage reports  
  - Built Docker image digests :contentReference[oaicite:6]{index=6}&#8203;:contentReference[oaicite:7]{index=7}  

#### 3. Model Drift Detection & Monitoring  
- **Drift Metrics** (Evidently.ai)  
  - **Numerical Drift**: KS-statistic, Population Stability Index (PSI)  
  - **Categorical Drift**: Chi-square test on frequency shifts  
  - **Prediction Drift**: Distribution changes in model outputs vs. ground truth  
- **Pipeline**:  
  1. **Load** reference (training) & current (production) datasets  
  2. **Configure** `ColumnMapping` (features, target, predictions)  
  3. **Run** `Report(metrics=[…])` → generate HTML report  
  4. **Log** HTML artifact under MLflow “DriftDetection” experiment  
- **Scheduling & Automation**:  
  - **Cron**: Daily at 02:00 AM ET via Airflow / system cron  
  - **MLflow**: Scheduled job option  
- **Alerting & Remediation**:  
  - **Slack**: Post summary with link to detailed report in `#model-alerts`  
  - **Email**: Send PDF/HTML report to data-science@company.com  
  - **Automated Retraining**: Trigger `docker/train` pipeline when drift exceeds thresholds  
  - **Rollback**: Redeploy last stable inference image on critical degradation  
- **Dashboarding**:  
  - Visualize drift trends, feature importance shifts, and model performance over time. :contentReference[oaicite:8]{index=8}&#8203;:contentReference[oaicite:9]{index=9}  


### API Service




## Team

- Team Name: [Group 7]
- Team Members:
  - [Claudia Ni] - GitHub: [@Claudia-Ni]
  - [Elina Gu] - GitHub: [@ElinaGu]
  - [Hannah Wang] - GitHub: [@hannah0406xy]
  - [Lincoln Lyu] - GitHub: [@Lincolnlyu]
  - [Ricardo Lu] - GitHub: [@rickyy-ming]

## License

This project is licensed under the MIT License - see the LICENSE file for details.
