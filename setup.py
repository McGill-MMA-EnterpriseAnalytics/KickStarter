from setuptools import setup, find_packages

setup(
    name="kickstarter_ml_pipeline",
    version="0.1.0",
    description="End-to-end pipeline for Kickstarter success prediction",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.8",
    install_requires=[
        "pandas",
        "numpy",
        "scikit-learn",
        "xgboost",
        "tensorflow",
        "mlflow",
        "optuna",
        "shap",
        "fairlearn",
        "lime",
        "seaborn",
        "fastapi",
        "uvicorn"
    ],
    entry_points={
        "console_scripts": [
            # Run the full pipeline: python -m kickstarter_ml_pipeline or `kickstarter-pipeline`
            "kickstarter-pipeline=src.pipeline:main",
            # Drift detection CLI
            "kickstarter-drift=src.drift_detection:detect_drift"
        ]
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
)