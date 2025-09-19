#!/usr/bin/env python3
"""
Test script to verify the beta parameter functionality in the binary outcome generator.
"""

import numpy as np
from sklearn.metrics import roc_auc_score
from data_simulation import Scorecard

def test_beta_parameter():
    """Test different beta values and their effect on the correlation between score and outcome."""
    print("Testing beta parameter in binary outcome generator...")
    print("=" * 60)
    
    # Test with different beta values
    beta_values = [0.5, 1.0, 2.0, 4.0]
    results = []
    
    for beta in beta_values:
        print(f"\nTesting beta = {beta}")
        print("-" * 30)
        
        # Create scorecard with fixed random state for reproducibility
        scorecard = Scorecard(n_rows=5000, n_features=8, binary_prevalence=0.1, random_state=42)
        
        # Generate binary outcome with specific beta
        outcome = scorecard.generate_binary_outcome(prevalence=0.1, random_state=42, beta=beta)
        
        # Calculate correlation and AUC
        total_scores = scorecard.total_scores['total_score']
        correlation = np.corrcoef(total_scores, outcome)[0, 1]
        auc = roc_auc_score(outcome, total_scores)
        
        # Check actual prevalence
        actual_prevalence = outcome.mean()
        
        print(f"  Actual prevalence: {actual_prevalence:.3f}")
        print(f"  Correlation: {correlation:.3f}")
        print(f"  AUC: {auc:.3f}")
        
        results.append({
            'beta': beta,
            'prevalence': actual_prevalence,
            'correlation': correlation,
            'auc': auc
        })
    
    print("\n" + "=" * 60)
    print("Summary:")
    print("Beta  | Prevalence | Correlation | AUC")
    print("-" * 40)
    for result in results:
        print(f"{result['beta']:4.1f}  |    {result['prevalence']:.3f}    |    {result['correlation']:.3f}     | {result['auc']:.3f}")
    
    print("\nExpected behavior:")
    print("- Higher beta values should lead to stronger correlation")
    print("- Higher beta values should lead to higher AUC")
    print("- Prevalence should remain close to 0.1 for all beta values")

if __name__ == "__main__":
    test_beta_parameter()
