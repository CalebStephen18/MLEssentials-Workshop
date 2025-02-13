import argparse
import os
import sys
from src.data_prep import load_data, preprocess_data, prepare_for_training
from src.model import HeartDiseaseModel
import mlflow
import logging
from datetime import datetime
import pandas as pd

def setup_logging(log_level=logging.INFO):
    """Set up logging configuration"""
    # Create logs directory if it doesn't exist
    if not os.path.exists('logs'):
        os.makedirs('logs')
    
    # Create a unique log file name with timestamp
    log_filename = f"logs/heart_disease_prediction_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    # Configure logging
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    return logging.getLogger(__name__)

def setup_argparse():
    parser = argparse.ArgumentParser(description='Heart Disease Prediction Pipeline')
    parser.add_argument('--data-path', type=str, default='data/heart.csv',
                       help='Path to the heart disease dataset')
    parser.add_argument('--experiment-name', type=str, default='Heart Disease Prediction',
                       help='Name of the MLflow experiment')
    parser.add_argument('--n-estimators', type=int, default=100,
                       help='Number of trees in Random Forest')
    parser.add_argument('--max-depth', type=int, default=None,
                       help='Maximum depth of trees')
    parser.add_argument('--log-level', type=str, default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                       help='Set the logging level')
    parser.add_argument('--model-output', type=str, default='models',
                       help='Directory to save the trained model')
    return parser

def create_directories(dirs):
    """Create necessary directories if they don't exist"""
    for dir_path in dirs:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
            logging.info(f"Created directory: {dir_path}")

def print_dataset_info(df):
    """Print dataset information"""
    info = {
        "Total Samples": len(df),
        "Features": list(df.columns),
        "Missing Values": df.isnull().sum().sum(),
        "Target Distribution": df['HeartDisease'].value_counts().to_dict()
    }
    return info

def main():
    # Set up argument parser
    parser = setup_argparse()
    args = parser.parse_args()
    
    # Set up logging
    logger = setup_logging(getattr(logging, args.log_level))
    logger.info("Starting Heart Disease Prediction Pipeline")
    logger.info(f"Arguments: {args}")
    
    # Create necessary directories
    create_directories(['logs', 'models', 'mlruns'])
    
    # Check if data file exists
    if not os.path.exists(args.data_path):
        logger.error(f"Data file not found at {args.data_path}")
        logger.info(
            "Please download the dataset from Kaggle using:\n"
            "kaggle datasets download -d fedesoriano/heart-failure-prediction\n"
            "Then unzip it to the data directory"
        )
        return
    
    try:
        # Load and preprocess data
        logger.info("Starting data pipeline")
        df = load_data(args.data_path)
        
        # Log dataset information
        dataset_info = print_dataset_info(df)
        logger.info("Dataset Information:")
        for key, value in dataset_info.items():
            logger.info(f"{key}: {value}")
        
        # Preprocess data
        logger.info("Preprocessing data...")
        df_processed = preprocess_data(df)
        
        # Prepare data for training
        logger.info("Preparing data for training...")
        X_train, X_test, y_train, y_test, scaler = prepare_for_training(df_processed)
        
        # Initialize and train model
        logger.info("Initializing model...")
        model = HeartDiseaseModel(
            n_estimators=args.n_estimators,
            max_depth=args.max_depth
        )
        
        # Train model
        logger.info("Training model...")
        model.train(X_train, y_train, args.experiment_name)
        
        # Evaluate model
        logger.info("Evaluating model...")
        metrics = model.evaluate(X_test, y_test)
        
        # Save model
        model_path = os.path.join(args.model_output, 
                                 f"heart_disease_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        model.save_model(model_path)
        logger.info(f"Model saved to {model_path}")
        
        # Print final results
        logger.info("\n" + "="*50)
        logger.info("Training completed successfully!")
        logger.info(f"Model accuracy: {metrics['accuracy']:.4f}")
        logger.info("\nClassification Report:")
        logger.info(metrics['classification_report'])
        
        # MLflow UI access information
        logger.info("\nTo view the experiment in MLflow UI:")
        logger.info("1. Run: mlflow ui")
        logger.info("2. Open: http://localhost:5000")
        logger.info("="*50)
        
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}", exc_info=True)
        raise
    finally:
        logger.info("Pipeline execution completed")

if __name__ == "__main__":
    main()