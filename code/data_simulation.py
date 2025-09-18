"""
Data Simulation Module

This module provides a clean, minimal implementation for generating simulated data
for the tuning_sim project. It creates pandas DataFrames with random values and
implements a Quantexa scorecard simulation with thresholds, binary scores, and weights.
"""

import numpy as np
import pandas as pd
from typing import Optional, Union, List
from scipy import stats


class Scorecard:
    """
    A class to simulate a Quantexa scorecard with features, thresholds, weights, and scores.
    
    This class generates random feature data, applies thresholds to create binary scores,
    and calculates weighted total scores. Binary scores are calculated on-the-fly and not stored.
    
    Attributes:
        features (pd.DataFrame): DataFrame with feature values (0-1 range)
        thresholds (pd.Series): Series with threshold values for each feature
        weights (pd.Series): Series with weight values for each feature
        total_scores (pd.DataFrame): DataFrame with total weighted scores
        binary_outcome (pd.Series): Binary outcome (0 or 1) correlated with total_score
    """
    
    def __init__(self, n_rows: int = 1000, n_features: int = 10, binary_prevalence: float = 0.1, beta = 1,
                 weights_override: Optional[Union[float, List[float]]] = None, random_state: Optional[int] = None):
        """
        Initialize a Scorecard with simulated data.
        
        Args:
            n_rows: Number of rows (samples) to generate
            n_features: Number of features (columns) to generate
            binary_prevalence: Desired prevalence for binary outcome (default 0.1 = 10%)
            beta: Parameter related to the power of the total_score to predict binary outcomes, 
            weights_override: Optional weights override. Can be:
                - float: Use this value as weight for all features
                - List[float]: Use these values as weights (must be n_features long)
                - None: Generate random weights (default behavior)
            random_state: Random seed for reproducible results
            
        Raises:
            ValueError: If binary_prevalence not in (0,1) or weights_override list length doesn't match n_features
        """
        if not 0 < binary_prevalence < 1:
            raise ValueError("Binary prevalence must be between 0 and 1")
            
        if random_state is not None:
            np.random.seed(random_state)
        
        # Validate weights_override if provided
        if weights_override is not None:
            if isinstance(weights_override, list):
                if len(weights_override) != n_features:
                    raise ValueError(f"weights_override list length ({len(weights_override)}) must match n_features ({n_features})")
            elif not isinstance(weights_override, (int, float)):
                raise ValueError("weights_override must be a number or a list of numbers")
        
        # Generate all components
        self.features = self._generate_features(n_rows, n_features)
        self.thresholds = self._generate_thresholds(n_features)
        self.weights = self._generate_weights(n_features, weights_override)
        self.total_scores = self._calculate_total_scores()
        
        # Generate binary outcome by default
        self.generate_binary_outcome(binary_prevalence, random_state, beta=beta)
    
    def _generate_features(self, n_rows: int, n_features: int) -> pd.DataFrame:
        """Generate random feature data between 0 and 1."""
        data = np.random.random(size=(n_rows, n_features))
        columns = [f'feature_{i+1}' for i in range(n_features)]
        return pd.DataFrame(data, columns=columns)
    
    def _generate_thresholds(self, n_features: int) -> pd.Series:
        """Generate random thresholds for each feature."""
        thresholds = np.random.random(size=n_features)
        index = [f'feature_{i+1}' for i in range(n_features)]
        return pd.Series(thresholds, index=index, name='threshold')
    
    def _generate_weights(self, n_features: int, weights_override: Optional[Union[float, List[float]]] = None) -> pd.Series:
        """
        Generate weights for each feature.
        
        Args:
            n_features: Number of features
            weights_override: Optional weights override. Can be:
                - float: Use this value as weight for all features
                - List[float]: Use these values as weights (must be n_features long)
                - None: Generate random weights (default behavior)
        
        Returns:
            pd.Series: Weights for each feature
        """
        index = [f'feature_{i+1}' for i in range(n_features)]
        
        if weights_override is None:
            # Default behavior: generate random weights
            weights = np.random.random(size=n_features)
        elif isinstance(weights_override, list):
            # Use provided list of weights
            weights = np.array(weights_override)
        else:
            # Use single weight value for all features
            weights = np.full(n_features, weights_override)
        
        return pd.Series(weights, index=index, name='weight')
    
    def _calculate_total_scores(self) -> pd.DataFrame:
        """Calculate weighted total scores by computing binary scores on-the-fly."""
        total_score = pd.Series(0.0, index=self.features.index)
        
        for col in self.features.columns:
            # Calculate binary score on-the-fly: 1 if feature > threshold, 0 otherwise
            binary_score = (self.features[col] > self.thresholds[col]).astype(int)
            total_score += binary_score * self.weights[col]
        
        return pd.DataFrame({'total_score': total_score})
    
    def generate_binary_outcome(self, prevalence: float = 0.1, random_state: Optional[int] = None, beta: float = 1.0) -> pd.Series:
        """
        Generate a binary outcome where log odds are proportional to total_score.
        
        This method creates a binary feature where:
        - Log odds = α + β * total_score
        - The overall prevalence (mean) equals the specified value
        - Higher total_scores have higher probability of positive outcome
        
        Args:
            prevalence: Desired overall prevalence (proportion of 1s), default 0.1 (10%)
            random_state: Random seed for reproducible results
            beta: Slope coefficient for the logistic regression (default 2.0).
                  Higher values create stronger correlation between total_score and outcome.
            
        Returns:
            pd.Series: Binary outcome (0 or 1) for each row
            
        Raises:
            ValueError: If prevalence is not between 0 and 1
        """
        if not 0 < prevalence < 1:
            raise ValueError("Prevalence must be between 0 and 1")
        
        if random_state is not None:
            np.random.seed(random_state)
        
        # Standardize total scores to have mean 0 and std 1
        total_scores_series = self.total_scores['total_score']
        total_scores_array = np.array(total_scores_series)
        standardized_scores = (total_scores_array - np.mean(total_scores_array)) / np.std(total_scores_array)
        
        # Use logistic regression approach to find intercept that gives desired prevalence
        # Use the provided beta parameter as the slope coefficient
        
        # Find intercept (alpha) that gives the desired prevalence
        # We need to solve: mean(1 / (1 + exp(-(alpha + beta * standardized_scores)))) = prevalence
        def prevalence_error(alpha):
            log_odds = alpha + beta * standardized_scores
            probabilities = 1 / (1 + np.exp(-log_odds))
            return np.mean(probabilities) - prevalence
        
        # Find alpha using root finding (search between -10 and 10)
        try:
            from scipy.optimize import brentq
            alpha = brentq(prevalence_error, -10, 10)
        except (ValueError, ImportError):
            # If root finding fails, use a simpler approach
            alpha = np.log(prevalence / (1 - prevalence))
        
        # Calculate final probabilities
        log_odds = alpha + beta * standardized_scores
        probabilities = 1 / (1 + np.exp(-log_odds))
        
        # Generate binary outcomes
        binary_outcome = np.random.binomial(1, probabilities)
        
        # Store the result and return
        self.binary_outcome = pd.Series(binary_outcome, index=self.total_scores.index, name='binary_outcome')
        return self.binary_outcome
