import pytest
import pandas as pd
import numpy as np
from src.data_prep import load_data, preprocess_data, create_features, prepare_for_training
from src.model import HeartDiseaseModel

@pytest.fixture
def sample_data():
    """Create sample data for testing"""
    return pd.DataFrame({
        'Age': [54, 65, 47],
        'Sex': ['M', 'F', 'M'],
        'ChestPainType': ['ATA', 'NAP', 'ASY'],
        'RestingBP': [140, 160, 130],
        'Cholesterol': [239, 180, 204],
        'FastingBS': [0, 1, 0],
        'RestingECG': ['Normal', 'ST', 'Normal'],
        'MaxHR': [160, 120, 172],
        'ExerciseAngina': ['N', 'Y', 'N'],
        'Oldpeak': [1.2, 2.0, 0.5],
        'ST_Slope': ['Flat', 'Down', 'Up'],
        'HeartDisease': [0, 1, 0]
    })

def test_data_loading(tmp_path):
    """Test data loading functionality"""
    # Create a temporary CSV file
    df = sample_data()
    csv_path = tmp_path / "test_heart.csv"
    df.to_csv(csv_path, index=False)
    
    # Test loading
    loaded_df = load_data(csv_path)
    assert len(loaded_df) == len(df)
    assert all(loaded_df.columns == df.columns)

def test_preprocessing(sample_data):
    """Test data preprocessing"""
    processed_df = preprocess_data(sample_data)
    
    # Check if new features are created
    expected_new_features = ['AgeGroup', 'BPCategory', 'HRCategory', 
                           'Age_BP_Interaction', 'Age_HR_Interaction']
    for feature in expected_new_features:
        assert feature in processed_df.columns
    
    # Check if categorical variables are encoded
    categorical_cols = ['Sex', 'ChestPainType', 'RestingECG', 
                       'ExerciseAngina', 'ST_Slope']
    for col in categorical_cols:
        assert processed_df[col].dtype in ['int32', 'int64']

def test_feature_creation(sample_data):
    """Test feature engineering"""
    featured_df = create_features(sample_data)
    
    # Check interaction features
    assert 'Age_BP_Interaction' in featured_df.columns
    assert 'Age_HR_Interaction' in featured_df.columns
    
    # Check categorical features
    assert 'AgeGroup' in featured_df.columns
    assert 'BPCategory' in featured_df.columns
    assert 'HRCategory' in featured_df.columns

def test_training_preparation(sample_data):
    """Test preparation of training data"""
    processed_df = preprocess_data(sample_data)
    X_train, X_test, y_train, y_test, scaler = prepare_for_training(processed_df)
    
    # Check shapes
    assert len(X_train) + len(X_test) == len(sample_data)
    assert len(y_train) + len(y_test) == len(sample_data)
    
    # Check if target is not in features
    assert 'HeartDisease' not in X_train.columns
    
    # Check if scaler is fitted
    assert hasattr(scaler, 'mean_')

def test_model_training(sample_data):
    """Test model training and prediction"""
    # Prepare data
    processed_df = preprocess_data(sample_data)
    X_train, X_test, y_train, y_test, scaler = prepare_for_training(processed_df)
    
    # Initialize and train model
    model = HeartDiseaseModel(n_estimators=10)
    model.train(X_train, y_train, "Test Experiment")
    
    # Test prediction
    predictions = model.predict(X_test)
    assert len(predictions) == len(y_test)
    assert all(isinstance(pred, (np.int64, int)) for pred in predictions)
    
    # Test evaluation
    metrics = model.evaluate(X_test, y_test)
    assert 'accuracy' in metrics
    assert 'confusion_matrix' in metrics
    assert 'classification_report' in metrics

if __name__ == '__main__':
    pytest.main([__file__])