# Kickstarter Success Prediction

A data science project to predict Kickstarter campaign success using machine learning techniques.

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

- Team Name: [Your Team Name]
- Team Members:
  - [Claudia Ni] - GitHub: [@github_id1]
  - [Elina Gu] - GitHub: [@github_id2]
  - [Hannah Wang] - GitHub: [@github_id3]
  - [Lincoln Lyu] - GitHub: [@github_id3]
  - [Ricardo Lu] - GitHub: [@github_id3]

## License

This project is licensed under the MIT License - see the LICENSE file for details.
