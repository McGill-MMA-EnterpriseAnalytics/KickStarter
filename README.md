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

Our analysis leverages a comprehensive dataset of **15,215 Kickstarter campaigns** spanning **2011–2021**, covering **22 countries**, **15 main categories** (158 subcategories), and **46 original variables**. Projects range from \$1 to \$100 M goals (median \$5 000) and exhibit a 57.7% success rate. This rich, multi-dimensional data enables robust modeling of success drivers across diverse campaign types. :contentReference[oaicite:0]{index=0}&#8203;:contentReference[oaicite:1]{index=1}

Key feature groups include:  
- **Project metadata**: name length, funding goal, category, country, staff pick status, video presence, etc.  
- **Temporal information**: creation date, launch date, deadline, campaign duration, weekday/weekend flags.  
- **Campaign characteristics**: blurb length, reward tiers, staff picks, spotlight flags.  
- **Engagement indicators**: historical subcategory success rates, country-level success rates.  

---

### Data Preprocessing & Feature Engineering

To ensure data integrity, prevent leakage, and extract meaningful predictors, we apply a multi-step preprocessing pipeline, with all actions logged via MLflow for transparency and reproducibility. :contentReference[oaicite:2]{index=2}&#8203;

1. **Leakage Prevention**  
   - **Target Definition**: Create binary label (`success` vs. `failure`) based solely on project state at prediction time.  
   - **Dynamic Removal**: Identify and drop post-outcome fields (e.g., `pledged_amount`, `backers`, `state_change_timestamps`, `spotlight`).  
   - **Audit Trail**: Log removed features and counts in MLflow under the “DataPreprocessing” experiment.  

2. **Missing Data Imputation**  
   - **Numeric Columns**: KNN imputation (`k=5`) on standardized values, then invert scaling to original units.  
   - **Categorical Columns**: Mode imputation for `main_category` (only ~1.9% missing).  
   - **Logging**: Record imputation counts and methods in MLflow.

3. **Datetime Feature Extraction**  
   - **Parsing**: Convert `created_at`, `launched_at`, and `deadline` to `datetime64`.  
   - **Derived Features**:  
     - `campaign_duration` = days between launch and deadline  
     - `prep_time` = days between creation and launch  
     - `launch_weekend` flag  
     - Cyclic encodings for `launch_month`, `launch_day_of_week`, `launch_hour` (sine & cosine transforms). 

4. **Goal-Related Features**  
   - **Daily Goal Rate**: `goal_per_day` = goal / campaign_duration  
   - **Log Transform**: `goal_log` = ln(goal + 1) to reduce skewness  
   - **Percentile Bins**: Bin `goal_log` into deciles to capture non-linear effects  
   - **USD Adjustment**: Normalize goal by historical currency exchange rates for multi-country campaigns  

5. **Categorical Encoding & Embeddings**  
   - **One-Hot Encoding**: For low-cardinality features (< 10 unique values) such as `launch_weekend`.  
   - **Smoothed Target Encoding**: For high-cardinality features (`category`, `subcategory`, `country`), using training-only aggregates to avoid leakage.  
   - **Entity Embeddings**: For selected high-cardinality categories (unique values ≥ 10), with embedding dimension = `min(50, √(n_unique))`. 

6. **Dimensionality Reduction**  
   - **Standardization**: Scale numeric features to zero mean and unit variance.  
   - **PCA**: Retain 95% of variance, reducing from 85 numeric dimensions to 52 principal components.  
   - **Integration**: Append PCA components to feature matrix and log explained variance ratio. 

7. **Final Cleanup**  
   - **Datetime Numeric Conversion**: Drop original timestamp columns; convert key dates to numeric days since `2010-01-01`.  
   - **Drop Originals**: Remove raw categorical columns after encoding and embedding.  
   - **Residual Imputation**: Median-impute any remaining missing values (e.g., missing percentile bins).  
   - **Audit**: Log final feature count and any dropped columns for reproducibility. 

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

Model evaluation quantifies how well our algorithms generalize to unseen data and informs model selection for deployment. We assess both baseline and hyperparameter-optimized versions of Logistic Regression, Random Forest, XGBoost, and MLP using multiple metrics—Accuracy, Precision, Recall, F1 Score, ROC AUC, and Average Precision—on the held-out test set. In-depth comparisons highlight the benefits of tuning and guide our choice of the final model. 

### Evaluation Metrics

- **Accuracy**: Overall fraction of correct predictions  
- **Precision**: Fraction of positive predictions that are true positives  
- **Recall**: Fraction of actual positives correctly identified  
- **F1 Score**: Harmonic mean of Precision and Recall  
- **ROC AUC**: Area under the Receiver Operating Characteristic curve—measures separability  
- **Average Precision**: Area under the Precision–Recall curve—sensitive to class imbalance 

### Baseline vs. Optimized Performance

- **Logistic Regression**  
  - Baseline ROC AUC: 0.9074, F1: 0.8382  
  - Optimized ROC AUC: 0.9077 (+0.0003), F1: 0.8393 

- **Random Forest**  
  - Baseline ROC AUC: 0.9059, F1: 0.8375  
  - Optimized ROC AUC: 0.9087 (+0.0028), F1: 0.8396 

- **XGBoost**  
  - Baseline ROC AUC: 0.8986, F1: 0.8284  
  - Optimized ROC AUC: 0.9105 (+0.0119), F1: 0.8335 

- **MLP Classifier**  
  - Baseline ROC AUC: 0.8804, F1: 0.8153  
  - Optimized ROC AUC: 0.9069 (+0.0265), F1: 0.8338

### Model Selection

The optimized XGBoost model achieves the highest ROC AUC (0.9105) and demonstrates balanced Precision (0.8456) and Recall (0.8217), making it the preferred candidate for production. Its substantial AUC gain from hyperparameter tuning underscores its ability to capture non-linear feature interactions effectively. 

### Calibration & Robustness

- **Calibration**: Reliability diagrams confirm well-calibrated probability outputs for the selected XGBoost model, ensuring reliable confidence estimates.  
- **Cross-Validation Stability**: Low variance in ROC AUC across folds (< 0.005) indicates consistent performance.  

---

## Ethical AI & Fairness
To ensure our final XGBoost model operates fairly across different user groups, we conducted a fairness analysis using country_encoded as the protected attribute. Leveraging Fairlearn, we computed group-specific metrics such as accuracy and selection rate, alongside global fairness metrics including Demographic Parity Difference and Equalized Odds Difference. The results indicated no substantial performance disparities across country groups, with fairness metrics falling within commonly accepted thresholds (≤ 0.1). These findings suggest that the model does not systematically favor or disadvantage any specific country group, supporting its ethical application in the Kickstarter prediction context.

### Explainability & Fairness

To build trust and identify potential biases, we apply both global and local explainability techniques alongside rigorous fairness audits.  

#### 1. SHAP Analysis  
- **Global Importance**  
  - **Top Features** (by mean absolute SHAP value):  
    1. `usd_goal_real` (higher goals decrease success probability)  
    2. `campaign_duration` (optimal around 30–45 days)  
    3. `launch_month` (seasonality effects; mid-year launches perform best)  
    4. `subcategory_success_rate` (historical success in subcategory boosts prediction)  
    5. `staff_pick` flag  
  - **Beeswarm Plot**: Displays each feature’s distribution of impacts on the model output, revealing nonlinear and interaction effects. 
- **Dependence Plots**  
  - **`usd_goal_real`**: Success probability drops sharply as goal rises above the 25th percentile, with diminishing marginal effects at very high goals. 
  - **`campaign_duration`**: Peak success around 30–45 days; very short (<15 days) or very long (>60 days) campaigns show reduced probability. 
  - **`launch_month`**: Warm-season months (May–August) have ~5–8% higher success probability compared to winter months.
    
- **Local Explanations**  
  - Use **SHAP Force Plots** to interpret individual predictions, e.g., why a campaign with a \$2,500 goal and 35-day duration was predicted as successful.  

#### 2. Fairness Evaluation  
We evaluate fairness across three protected attributes: **staff_pick**, **goal_percentile_bin**, and **main_category_encoded**.  

| Attribute               | Metric                    | Group A                   | Group B                      | Disparity       |
|-------------------------|---------------------------|---------------------------|------------------------------|-----------------|
| **Staff Pick**          | Accuracy                  | 91.6%                     | 78.6%                        | Δ = 0.130       |
|                         | Selection Rate (TPR)      | 96.5%                     | 49.8%                        | Δ = 0.4671      |
|                         | Equalized Odds Difference | —                         | —                            | 0.6133          |
| **Goal Percentile Bin** | Selection Rate (TPR)      | Lowest 50%: 82%           | Highest 10%: 8%              | Δ = 0.7401      |
|                         | Equalized Odds Difference | —                         | —                            | 0.6023          |
| **Category Encoded**    | Selection Rate (TPR)      | Best categories: >80%     | Worst categories: ~25%       | Δ = 0.6552      |
|                         | Equalized Odds Difference | —                         | —                            | 0.9464          |

- **Key Findings**  
  - Staff-picked campaigns are far more likely to be correctly classified (ΔAccuracy ≈13%).  
  - High-goal campaigns suffer from dramatically lower true positive rates (ΔTPR ≈74%).  
  - Certain project categories are underrepresented, with TPR disparities up to ~65%. 
- **Mitigation Strategies**  
  1. **Reweighting**: Apply sample weights inversely proportional to group prevalence during training.  
  2. **Fairness-Aware Algorithms**: Integrate constraints (e.g., Equalized Odds) via fairlearn or a post-processing wrapper.  
  3. **Threshold Adjustment**: Calibrate decision thresholds per subgroup to equalize TPRs.  
  4. **Data Augmentation**: Oversample underperforming categories or goal bins to balance training data.  

#### 3. Additional Explainability Methods  
- **LIME (Local Interpretable Model-Agnostic Explanations)**  
  - Generates local surrogate linear models for individual predictions, confirming that `staff_pick`, `subcategory_success_rate`, and `usd_goal_real` consistently dominate.  
- **Permutation Importance (ELI5)**  
  - **Random Forest**: `staff_pick` (~0.037), `subcategory_success_rate`, followed by PCA components.  
  - **Logistic Regression**: `category_success_rate` (~0.23), `main_category_encoded`, `staff_pick`.  
  - Confirms global SHAP rankings and highlights model-specific nuances. 

By combining SHAP, LIME, and permutation importance, we achieve both global and local transparency. The fairness audit, with concrete disparity metrics and mitigation plans, ensures we proactively address biases before deployment.  


---


## MLOps & Deployment

To ensure reliable, repeatable, and scalable delivery of our Kickstarter Success Prediction pipeline, we operationalize every stage—from model training to serving and monitoring—using containerization, CI/CD, and automated drift detection. This MLOps framework guarantees reproducibility, rapid iteration, and proactive maintenance in production.

### 1. Dockerized Pipelines  
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
    These artifacts are written to a mounted host volume and logged to MLflow. 

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
    - Endpoint: `GET /healthz` returns `200 OK` when service is ready. 

### 2. CI/CD with GitHub Actions  
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
- **Secrets & Config**:  
  - `DOCKERHUB_USERNAME`, `DOCKERHUB_TOKEN`  
  - `KUBE_CONFIG` for cluster access  
- **Artifacts**:  
  - Test reports (JUnit XML)  
  - Coverage reports  
  - Built Docker image digests   

### 3. Model Drift Detection & Monitoring  
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
  - Visualize drift trends, feature importance shifts, and model performance over time. 

---
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
