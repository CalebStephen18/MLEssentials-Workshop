from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import logging
import json
from datetime import datetime

logger = logging.getLogger(__name__)

class HeartDiseaseModel:
    def __init__(self, n_estimators=100, max_depth=None, random_state=42):
        """
        Initialize the heart disease prediction model.
        
        Args:
            n_estimators: Number of trees in the forest
            max_depth: Maximum depth of the trees
            random_state: Random state for reproducibility
        """
        logger.info(f"Initializing model with {n_estimators} trees")
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state
        )
        
    def train(self, X_train, y_train, experiment_name="Heart Disease Prediction"):
        """
        Train the model and log metrics with MLflow.
        
        Args:
            X_train: Training features
            y_train: Training target
            experiment_name: Name for MLflow experiment
        """
        logger.info(f"Setting up MLflow experiment: {experiment_name}")
        mlflow.set_experiment(experiment_name)
        
        with mlflow.start_run(run_name=f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
            logger.info("Starting model training")
            
            # Train the model
            self.model.fit(X_train, y_train)
            logger.info("Model training completed")
            
            # Log model parameters
            logger.info("Logging model parameters")
            params = {
                "n_estimators": self.model.n_estimators,
                "max_depth": self.model.max_depth,
                "criterion": self.model.criterion,
                "min_samples_split": self.model.min_samples_split
            }
            mlflow.log_params(params)
            
            # Log the model
            logger.info("Logging model to MLflow")
            mlflow.sklearn.log_model(self.model, "model")
            
            # Log feature importance
            logger.info("Calculating and logging feature importance")
            feature_importance = pd.DataFrame({
                'feature': X_train.columns,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            # Log feature importance as JSON
            mlflow.log_dict(
                feature_importance.to_dict(orient='records'),
                "feature_importance.json"
            )
            
            # Log top features plot artifact
            top_features = feature_importance.head(10).to_dict(orient='records')
            logger.info("Top 10 important features:")
            for feature in top_features:
                logger.info(f"{feature['feature']}: {feature['importance']:.4f}")
    
    def evaluate(self, X_test, y_test):
        """
        Evaluate the model and log metrics with MLflow.
        
        Args:
            X_test: Test features
            y_test: Test target
        Returns:
            Dictionary of evaluation metrics
        """
        logger.info("Starting model evaluation")
        
        with mlflow.start_run(nested=True):
            # Make predictions
            logger.info("Making predictions on test set")
            y_pred = self.model.predict(X_test)
            
            # Calculate metrics
            logger.info("Calculating evaluation metrics")
            accuracy = accuracy_score(y_test, y_pred)
            conf_matrix = confusion_matrix(y_test, y_pred)
            class_report = classification_report(y_test, y_pred)
            
            metrics = {
                'accuracy': accuracy,
                'confusion_matrix': conf_matrix.tolist(),
                'classification_report': class_report
            }
            
            # Log metrics
            logger.info(f"Model accuracy: {accuracy:.4f}")
            mlflow.log_metric("accuracy", accuracy)
            
            # Log confusion matrix as JSON
            mlflow.log_dict(
                {"confusion_matrix": metrics['confusion_matrix']},
                "confusion_matrix.json"
            )
            
            # Log detailed metrics
            logger.info("Classification Report:")
            logger.info(f"\n{class_report}")
            
            return metrics
    
    def predict(self, X):
        """
        Make predictions on new data.
        
        Args:
            X: Features to predict on
        Returns:
            Array of predictions
        """
        logger.info(f"Making predictions on {len(X)} samples")
        predictions = self.model.predict(X)
        logger.info("Predictions completed")
        return predictions
        
    def save_model(self, filepath):
        """
        Save the trained model to a file.
        
        Args:
            filepath: Path to save the model
        """
        logger.info(f"Saving model to {filepath}")
        mlflow.sklearn.save_model(self.model, filepath)
        logger.info("Model saved successfully")