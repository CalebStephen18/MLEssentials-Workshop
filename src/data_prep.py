import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import logging

# Set up logging
logger = logging.getLogger(__name__)

def load_data(filepath: str) -> pd.DataFrame:
    """
    Load the heart disease dataset.
    
    Args:
        filepath: Path to the CSV file
    Returns:
        DataFrame containing the heart disease data
    """
    logger.info(f"Loading data from {filepath}")
    df = pd.read_csv(filepath)
    logger.info(f"Loaded {len(df)} rows and {len(df.columns)} columns")
    return df

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess the heart disease data.
    
    Args:
        df: Raw DataFrame
    Returns:
        Preprocessed DataFrame
    """
    logger.info("Starting data preprocessing")
    df_processed = df.copy()
    
    # Encode categorical variables
    categorical_cols = ['Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope']
    label_encoders = {}
    
    for col in categorical_cols:
        logger.info(f"Encoding categorical column: {col}")
        label_encoders[col] = LabelEncoder()
        df_processed[col] = label_encoders[col].fit_transform(df_processed[col])
    
    # Handle any missing values
    logger.info("Checking for missing values")
    missing_values = df_processed.isnull().sum()
    if missing_values.any():
        logger.warning(f"Found missing values:\n{missing_values[missing_values > 0]}")
        df_processed = handle_missing_values(df_processed)
    
    # Feature engineering
    logger.info("Creating new features")
    df_processed = create_features(df_processed)
    
    logger.info("Preprocessing completed")
    return df_processed

def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Handle missing values in the dataset.
    
    Args:
        df: DataFrame with missing values
    Returns:
        DataFrame with handled missing values
    """
    logger.info("Handling missing values")
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    
    # Fill numeric columns with median
    for col in numeric_cols:
        if df[col].isnull().any():
            logger.info(f"Filling missing values in {col} with median")
            df[col] = df[col].fillna(df[col].median())
    
    return df

def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create new features for the model.
    
    Args:
        df: Input DataFrame
    Returns:
        DataFrame with new features
    """
    logger.info("Starting feature engineering")
    df = df.copy()
    
    # Create age groups
    logger.info("Creating age groups")
    df['AgeGroup'] = pd.cut(df['Age'], 
                           bins=[0, 40, 50, 60, 100],
                           labels=['<40', '40-50', '50-60', '>60'])
    
    # Create BP categories based on medical standards
    logger.info("Creating blood pressure categories")
    df['BPCategory'] = pd.cut(df['RestingBP'],
                             bins=[0, 120, 140, 180, 300],
                             labels=['normal', 'prehypertension', 'hypertension', 'severe'])
    
    # Create HR categories
    logger.info("Creating heart rate categories")
    df['HRCategory'] = pd.cut(df['MaxHR'],
                             bins=[0, 100, 140, 170, 220],
                             labels=['low', 'normal', 'elevated', 'high'])
    
    # Create interaction features
    logger.info("Creating interaction features")
    df['Age_BP_Interaction'] = df['Age'] * df['RestingBP'] / 100
    df['Age_HR_Interaction'] = df['Age'] * df['MaxHR'] / 100
    
    # Encode the new categorical features
    logger.info("Encoding new categorical features")
    new_categorical_cols = ['AgeGroup', 'BPCategory', 'HRCategory']
    for col in new_categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
    
    logger.info("Feature engineering completed")
    return df

def prepare_for_training(df: pd.DataFrame, target_col: str = 'HeartDisease'):
    """
    Prepare data for model training.
    
    Args:
        df: Preprocessed DataFrame
        target_col: Name of the target column
    Returns:
        X_train, X_test, y_train, y_test, scaler
    """
    logger.info("Preparing data for training")
    
    # Separate features and target
    y = df[target_col]
    X = df.drop(columns=[target_col])
    
    # Select numeric columns for scaling
    numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns
    
    # Create scaler
    logger.info("Scaling numeric features")
    scaler = StandardScaler()
    
    # Scale numeric features
    X_scaled = X.copy()
    X_scaled[numeric_cols] = scaler.fit_transform(X[numeric_cols])
    
    # Split the data
    logger.info("Splitting data into train and test sets")
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )
    
    logger.info(f"Training set size: {len(X_train)}, Test set size: {len(X_test)}")
    
    return X_train, X_test, y_train, y_test, scaler