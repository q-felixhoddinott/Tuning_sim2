"""
Modeling Pipeline Module

This module provides a minimal ModelingPipeline class that creates scorecards,
partitions data, trains logistic regression and XGBoost models, and provides
comprehensive evaluation metrics and scoring functions.
"""

import numpy as np
import pandas as pd
from typing import Optional, Union, List, Dict, Any
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve
import xgboost as xgb

from data_simulation import Scorecard


class ModelingPipeline:
    """
    A complete modeling pipeline that creates scorecards, trains models, and provides evaluation.
    
    This class handles the entire workflow from data generation through model training
    and evaluation, providing both logistic regression and XGBoost models trained on
    the untransformed total_score to predict binary outcomes.
    
    Attributes:
        scorecard (Scorecard): Generated scorecard with features and outcomes
        X_train (pd.Series): Training features (total_score)
        X_val (pd.Series): Validation features (total_score)
        y_train (pd.Series): Training targets (binary_outcome)
        y_val (pd.Series): Validation targets (binary_outcome)
        logreg_model (LogisticRegression): Trained logistic regression model
        xgb_model (xgb.XGBClassifier): Trained XGBoost model
        logreg_auc (float): AUC score for logistic regression
        xgb_auc (float): AUC score for XGBoost
        logreg_metrics (Dict): Metrics dict for logistic regression
        xgb_metrics (Dict): Metrics dict for XGBoost
    """
    
    def __init__(self, n_rows: int = 1000, n_features: int = 8, binary_prevalence: float = 0.1, 
                 beta: float = 2.0, weights_override: Optional[Union[float, List[float]]] = None,
                 random_state: Optional[int] = None, test_size: float = 0.3):
        """
        Initialize the modeling pipeline with all components.
        
        Args:
            n_rows: Number of rows (samples) to generate
            n_features: Number of features (columns) to generate  
            binary_prevalence: Desired prevalence for binary outcome (default 0.1 = 10%)
            beta: Parameter controlling correlation between total_score and binary outcome
            weights_override: Optional weights override for scorecard features
            random_state: Random seed for reproducible results
            test_size: Proportion of data to use for validation (default 0.3)
        """
        # Store parameters
        self.random_state = random_state
        self.test_size = test_size
        
        # Create scorecard
        self.scorecard = Scorecard(
            n_rows=n_rows,
            n_features=n_features, 
            binary_prevalence=binary_prevalence,
            beta=beta,
            weights_override=weights_override,
            random_state=random_state
        )
        
        # Partition data
        self._partition_data()
        
        # Train models
        self._train_models()
        
        # Calculate evaluation metrics
        self._calculate_metrics()
    
    def _partition_data(self):
        """Partition the data into training and validation sets."""
        # Extract features and targets
        X = self.scorecard.total_scores['total_score']
        y = self.scorecard.binary_outcome
        
        # Split the data
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state, stratify=y
        )
    
    def _train_models(self):
        """Train both logistic regression and XGBoost models."""
        # Reshape features for sklearn (needs 2D array)
        X_train_reshaped = self.X_train.values.reshape(-1, 1)
        X_val_reshaped = self.X_val.values.reshape(-1, 1)
        
        # Train logistic regression
        self.logreg_model = LogisticRegression(random_state=self.random_state)
        self.logreg_model.fit(X_train_reshaped, self.y_train)
        
        # Train XGBoost
        self.xgb_model = xgb.XGBClassifier(
            random_state=self.random_state,
            eval_metric='logloss'  # Suppress warning
        )
        self.xgb_model.fit(X_train_reshaped, self.y_train)
    
    def _calculate_metrics(self):
        """Calculate AUC and precision/recall metrics at 50% population threshold."""
        # Reshape validation features
        X_val_reshaped = self.X_val.values.reshape(-1, 1)
        
        # Get predictions (probabilities)
        logreg_probs = self.logreg_model.predict_proba(X_val_reshaped)[:, 1]
        xgb_probs = self.xgb_model.predict_proba(X_val_reshaped)[:, 1]
        
        # Calculate AUC scores
        self.logreg_auc = roc_auc_score(self.y_val, logreg_probs)
        self.xgb_auc = roc_auc_score(self.y_val, xgb_probs)
        
        # Calculate metrics at 50% population threshold
        self.logreg_metrics = self._calculate_threshold_metrics(logreg_probs, self.y_val)
        self.xgb_metrics = self._calculate_threshold_metrics(xgb_probs, self.y_val)
        
        # Add AUC to metrics dictionaries
        self.logreg_metrics['auc'] = self.logreg_auc
        self.xgb_metrics['auc'] = self.xgb_auc
    
    def _calculate_threshold_metrics(self, probabilities: np.ndarray, true_labels: pd.Series) -> Dict[str, float]:
        """
        Calculate precision and recall at the threshold that selects 50% of population.
        
        Args:
            probabilities: Model predicted probabilities
            true_labels: True binary labels
            
        Returns:
            Dictionary with precision, recall, and threshold at 50% population
        """
        # Find threshold at 50th percentile (top 50% of scored population)
        threshold_50pct = np.percentile(probabilities, 50)
        
        # Make predictions at this threshold
        predictions = (probabilities >= threshold_50pct).astype(int)
        
        # Calculate precision and recall manually
        true_positives = np.sum((predictions == 1) & (true_labels == 1))
        false_positives = np.sum((predictions == 1) & (true_labels == 0))
        false_negatives = np.sum((predictions == 0) & (true_labels == 1))
        
        # Handle edge cases
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
        
        # Store private threshold for scoring functions
        if hasattr(self, '_logreg_threshold_50'):
            self._xgb_threshold_50 = threshold_50pct
        else:
            self._logreg_threshold_50 = threshold_50pct
        
        return {
            'precision_at_50pct': precision,
            'recall_at_50pct': recall,
            'threshold_50pct': threshold_50pct
        }
    
    def score_logreg(self, total_score: Union[float, List[float], np.ndarray]) -> Union[float, np.ndarray]:
        """
        Score new examples using the trained logistic regression model.
        
        Args:
            total_score: Single value, list, or array of total_score values
            
        Returns:
            Predicted probabilities for the positive class
        """
        # Convert to numpy array and reshape for sklearn
        scores = np.array(total_score)
        if scores.ndim == 0:
            scores = scores.reshape(1, -1)
        else:
            scores = scores.reshape(-1, 1)
        
        # Get probabilities
        probabilities = self.logreg_model.predict_proba(scores)[:, 1]
        
        # Return single value if input was single value
        if len(probabilities) == 1:
            return float(probabilities[0])
        return probabilities
    
    def score_xgb(self, total_score: Union[float, List[float], np.ndarray]) -> Union[float, np.ndarray]:
        """
        Score new examples using the trained XGBoost model.
        
        Args:
            total_score: Single value, list, or array of total_score values
            
        Returns:
            Predicted probabilities for the positive class
        """
        # Convert to numpy array and reshape for sklearn
        scores = np.array(total_score)
        if scores.ndim == 0:
            scores = scores.reshape(1, -1)
        else:
            scores = scores.reshape(-1, 1)
        
        # Get probabilities
        probabilities = self.xgb_model.predict_proba(scores)[:, 1]
        
        # Return single value if input was single value
        if len(probabilities) == 1:
            return float(probabilities[0])
        return probabilities
