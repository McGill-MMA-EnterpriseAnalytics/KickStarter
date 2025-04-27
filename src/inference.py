'''
This script demonstrates how to load and use the saved pipeline and features.
'''
import joblib
import pandas as pd


def load_pipeline(pipeline_path: str = 'best_kickstarter_model_pipeline.pkl') -> object:
    """
    Load the scikit-learn pipeline from disk.
    """
    pipeline = joblib.load(pipeline_path)
    return pipeline


def load_selected_features(features_path: str = 'selected_features.txt') -> list:
    """
    Load the list of selected feature names from a text file.
    """
    with open(features_path, 'r') as f:
        features = [line.strip() for line in f if line.strip()]
    return features


def demo_prediction(pipeline, features):
    """
    Create a dummy sample (zeros) to demonstrate prediction and probability.
    """
    # Create a DataFrame with a single sample (all zeros)
    sample = pd.DataFrame([[0] * len(features)], columns=features)
    prediction = pipeline.predict(sample)
    probability = pipeline.predict_proba(sample)[:, 1]
    print('Demo sample prediction:', prediction)
    print('Demo sample probability of success:', probability)


def main():
    # Load artifacts
    pipeline = load_pipeline()
    features = load_selected_features()

    print('Loaded pipeline steps:')
    print(pipeline)
    print('\nSelected features:')
    print(features)

    # Demo prediction
    demo_prediction(pipeline, features)


if __name__ == '__main__':
    main()
