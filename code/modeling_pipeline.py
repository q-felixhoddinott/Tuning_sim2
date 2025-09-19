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
    A complete modeling pipeline that creates or accepts scorecards, trains models, and provides evaluation.
    
    This class handles the entire workflow from data generation through model training
    and evaluation, providing both logistic regression and XGBoost models trained on
    the raw scorecard features to predict binary outcomes. Also evaluates the original
    scorecard total_score as a third comparison "model".
    
    The pipeline can either create a new scorecard with specified parameters or accept
    an existing Scorecard object for reuse across different modeling experiments.
    
    Attributes:
        scorecard (Scorecard): Scorecard with features and outcomes (created or provided)
        X_train (pd.DataFrame): Training features (raw scorecard features)
        X_val (pd.DataFrame): Validation features (raw scorecard features)
        y_train (pd.Series): Training targets (binary_outcome)
        y_val (pd.Series): Validation targets (binary_outcome)
        total_scores_train (pd.Series): Training total_scores from scorecard
        total_scores_val (pd.Series): Validation total_scores from scorecard
        logreg_model (LogisticRegression): Trained logistic regression model
        xgb_model (xgb.XGBClassifier): Trained XGBoost model
        logreg_auc (float): AUC score for logistic regression
        xgb_auc (float): AUC score for XGBoost
        total_score_auc (float): AUC score for total_score
        logreg_metrics (Dict): Metrics dict for logistic regression
        xgb_metrics (Dict): Metrics dict for XGBoost
        total_score_metrics (Dict): Metrics dict for total_score
    """
    
    def __init__(self, n_rows: int = 1000, n_features: int = 8, binary_prevalence: float = 0.1, 
                 beta: float = 2.0, weights_override: Optional[Union[float, List[float]]] = None,
                 random_state: Optional[int] = None, test_size: float = 0.3, 
                 scorecard: Optional[Scorecard] = None):
        """
        Initialize the modeling pipeline with all components.
        
        Args:
            n_rows: Number of rows (samples) to generate (ignored if scorecard provided)
            n_features: Number of features (columns) to generate (ignored if scorecard provided)
            binary_prevalence: Desired prevalence for binary outcome (ignored if scorecard provided)
            beta: Parameter controlling correlation between scorecard total_score and binary outcome (ignored if scorecard provided)
            weights_override: Optional weights override for scorecard features (ignored if scorecard provided)
            random_state: Random seed for reproducible results
            test_size: Proportion of data to use for validation (default 0.3)
            scorecard: Optional existing Scorecard object to use instead of creating a new one
        """
        # Store parameters
        self.random_state = random_state
        self.test_size = test_size
        
        # Use existing scorecard or create new one
        if scorecard is not None:
            if not isinstance(scorecard, Scorecard):
                raise TypeError("scorecard must be an instance of Scorecard class")
            self.scorecard = scorecard
        else:
            # Create new scorecard
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
        # Extract features, targets, and total_scores
        X = self.scorecard.features
        y = self.scorecard.binary_outcome
        total_scores = self.scorecard.total_scores['total_score']
        
        # Split the data consistently
        self.X_train, self.X_val, self.y_train, self.y_val, self.total_scores_train, self.total_scores_val = train_test_split(
            X, y, total_scores, test_size=self.test_size, random_state=self.random_state, stratify=y
        )
    
    def _train_models(self):
        """Train both logistic regression and XGBoost models."""
        # Train logistic regression on raw features
        self.logreg_model = LogisticRegression(random_state=self.random_state)
        self.logreg_model.fit(self.X_train, self.y_train)
        
        # Train XGBoost on raw features
        self.xgb_model = xgb.XGBClassifier(
            random_state=self.random_state,
            eval_metric='logloss'  # Suppress warning
        )
        self.xgb_model.fit(self.X_train, self.y_train)
    
    def _calculate_metrics(self):
        """Calculate AUC and precision/recall metrics at 50% population threshold for all models."""
        # Get predictions (probabilities) using raw features
        logreg_probs = self.logreg_model.predict_proba(self.X_val)[:, 1]
        xgb_probs = self.xgb_model.predict_proba(self.X_val)[:, 1]
        
        # Calculate AUC scores for ML models
        self.logreg_auc = roc_auc_score(self.y_val, logreg_probs)
        self.xgb_auc = roc_auc_score(self.y_val, xgb_probs)
        
        # Calculate AUC for total_score (use total_score directly as "predictions")
        self.total_score_auc = roc_auc_score(self.y_val, self.total_scores_val)
        
        # Calculate metrics at 50% population threshold for all models
        self.logreg_metrics = self._calculate_threshold_metrics(logreg_probs, self.y_val)
        self.xgb_metrics = self._calculate_threshold_metrics(xgb_probs, self.y_val)
        self.total_score_metrics = self._calculate_threshold_metrics(self.total_scores_val.values, self.y_val)
        
        # Add AUC to metrics dictionaries
        self.logreg_metrics['auc'] = self.logreg_auc
        self.xgb_metrics['auc'] = self.xgb_auc
        self.total_score_metrics['auc'] = self.total_score_auc
    
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
            'precision_at_50pct': float(precision),
            'recall_at_50pct': float(recall),
            'threshold_50pct': float(threshold_50pct)
        }
    
    def score_logreg(self, features: Union[pd.DataFrame, np.ndarray, List[List[float]]]) -> Union[float, np.ndarray]:
        """
        Score new examples using the trained logistic regression model.
        
        Args:
            features: Feature values as DataFrame, array, or list of lists.
                     Should have same number of columns as training features.
            
        Returns:
            Predicted probabilities for the positive class
        """
        # Convert to DataFrame if needed
        if isinstance(features, list):
            features = np.array(features)
        if isinstance(features, np.ndarray):
            # If 1D array (single sample), reshape to 2D
            if features.ndim == 1:
                features = features.reshape(1, -1)
            features = pd.DataFrame(features, columns=self.scorecard.features.columns)
        
        # Get probabilities
        probabilities = self.logreg_model.predict_proba(features)[:, 1]
        
        # Return single value if input was single sample
        if len(probabilities) == 1:
            return float(probabilities[0])
        return probabilities
    
    def score_xgb(self, features: Union[pd.DataFrame, np.ndarray, List[List[float]]]) -> Union[float, np.ndarray]:
        """
        Score new examples using the trained XGBoost model.
        
        Args:
            features: Feature values as DataFrame, array, or list of lists.
                     Should have same number of columns as training features.
            
        Returns:
            Predicted probabilities for the positive class
        """
        # Convert to DataFrame if needed
        if isinstance(features, list):
            features = np.array(features)
        if isinstance(features, np.ndarray):
            # If 1D array (single sample), reshape to 2D
            if features.ndim == 1:
                features = features.reshape(1, -1)
            features = pd.DataFrame(features, columns=self.scorecard.features.columns)
        
        # Get probabilities
        probabilities = self.xgb_model.predict_proba(features)[:, 1]
        
        # Return single value if input was single sample
        if len(probabilities) == 1:
            return float(probabilities[0])
        return probabilities
