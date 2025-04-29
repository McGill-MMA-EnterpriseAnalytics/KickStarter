# Kickstarter Success Prediction

A data science project to predict Kickstarter campaign success using advanced machine learning techniques.

## Project Overview

This project analyzes Kickstarter campaign data to develop models that predict campaign success. The insights and models can help creators optimize their campaigns for better funding outcomes.

## Dataset

The dataset contains information about 15,215 Kickstarter projects with 46 features including:
- Project metadata (name, goal, pledged amount, category, etc.)
- Temporal information (launch date, deadline, etc.)
- Campaign characteristics (staff pick status, video presence, etc.)

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

## Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/username/kickstarter-prediction.git
cd kickstarter-prediction
pip install -r requirements.txt
```

## Usage

### Data Preparation

```bash
python src/data/process_data.py
```

### Model Training

```bash
python src/models/train_model.py
```

### Ethical AI & Fairness
To ensure our final XGBoost model operates fairly across different user groups, we conducted a fairness analysis using country_encoded as the protected attribute. Leveraging Fairlearn, we computed group-specific metrics such as accuracy and selection rate, alongside global fairness metrics including Demographic Parity Difference and Equalized Odds Difference. The results indicated no substantial performance disparities across country groups, with fairness metrics falling within commonly accepted thresholds (≤ 0.1). These findings suggest that the model does not systematically favor or disadvantage any specific country group, supporting its ethical application in the Kickstarter prediction context.

### API Service

```bash
python api/app.py
```

## Key Features

- Data preprocessing and advanced feature engineering
- Machine learning model training with hyperparameter optimization
- Model interpretation and explainability analysis
- MLOps implementation with CI/CD, Docker, and MLflow
- API for model serving and predictions

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
