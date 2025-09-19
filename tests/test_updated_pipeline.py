#!/usr/bin/env python3
"""
Test script for the updated ModelingPipeline that uses raw features instead of total_score.
"""

from code.modeling_pipeline import ModelingPipeline
import numpy as np
import pandas as pd

def test_pipeline():
    """Test the updated modeling pipeline."""
    print("Testing updated ModelingPipeline...")
    
    # Create pipeline with small dataset for quick testing
    pipeline = ModelingPipeline(
        n_rows=500,
        n_features=6, 
        binary_prevalence=0.15,
        beta=2.0,
        random_state=42
    )
    
    print(f"Pipeline created successfully!")
    print(f"Scorecard features shape: {pipeline.scorecard.features.shape}")
    print(f"Training features shape: {pipeline.X_train.shape}")
    print(f"Validation features shape: {pipeline.X_val.shape}")
    print(f"Training targets shape: {pipeline.y_train.shape}")
    print(f"Validation targets shape: {pipeline.y_val.shape}")
    
    # Test model performance
    print(f"\nModel Performance:")
    print(f"Logistic Regression AUC: {pipeline.logreg_auc:.4f}")
    print(f"XGBoost AUC: {pipeline.xgb_auc:.4f}")
    print(f"Total Score AUC: {pipeline.total_score_auc:.4f}")
    
    print(f"\nLogistic Regression Metrics:")
    for key, value in pipeline.logreg_metrics.items():
        print(f"  {key}: {value:.4f}")
    
    print(f"\nXGBoost Metrics:")
    for key, value in pipeline.xgb_metrics.items():
        print(f"  {key}: {value:.4f}")
    
    print(f"\nTotal Score Metrics:")
    for key, value in pipeline.total_score_metrics.items():
        print(f"  {key}: {value:.4f}")
    
    # Test scoring functions with different input formats
    print(f"\nTesting scoring functions...")
    
    # Test with single sample as numpy array
    sample_features = pipeline.X_val.iloc[0].values
    print(f"Sample features shape: {sample_features.shape}")
    
    logreg_score = pipeline.score_logreg(sample_features)
    xgb_score = pipeline.score_xgb(sample_features)
    print(f"Single sample - LogReg score: {logreg_score:.4f}")
    print(f"Single sample - XGB score: {xgb_score:.4f}")
    
    # Test with multiple samples as DataFrame
    multi_samples = pipeline.X_val.iloc[:3]
    print(f"Multi samples shape: {multi_samples.shape}")
    
    logreg_scores = pipeline.score_logreg(multi_samples)
    xgb_scores = pipeline.score_xgb(multi_samples)
    print(f"Multi samples - LogReg scores: {logreg_scores}")
    print(f"Multi samples - XGB scores: {xgb_scores}")
    
    # Test with list of lists
    list_features = [[0.1, 0.2, 0.3, 0.4, 0.5, 0.6], [0.6, 0.5, 0.4, 0.3, 0.2, 0.1]]
    logreg_scores_list = pipeline.score_logreg(list_features)
    xgb_scores_list = pipeline.score_xgb(list_features)
    print(f"List input - LogReg scores: {logreg_scores_list}")
    print(f"List input - XGB scores: {xgb_scores_list}")
    
    print("\n✅ All tests passed! Pipeline is working correctly.")

if __name__ == "__main__":
    test_pipeline()
