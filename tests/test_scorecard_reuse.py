#!/usr/bin/env python3
"""
Test script to verify that ModelingPipeline can accept existing scorecard objects.
Tests both the original functionality (creating new scorecard) and the new functionality (reusing existing scorecard).
"""

from data_simulation import Scorecard
from modeling_pipeline import ModelingPipeline

def test_new_scorecard_creation():
    """Test original functionality: pipeline creates new scorecard."""
    print("Testing original functionality (new scorecard creation)...")
    
    # Create pipeline with new scorecard
    pipeline1 = ModelingPipeline(
        n_rows=500,
        n_features=6,
        binary_prevalence=0.15,
        beta=1.5,
        random_state=42
    )
    
    print(f"Pipeline 1 - Scorecard shape: {pipeline1.scorecard.features.shape}")
    print(f"Pipeline 1 - Binary prevalence: {pipeline1.scorecard.binary_outcome.mean():.3f}")
    print(f"Pipeline 1 - LogReg AUC: {pipeline1.logreg_auc:.3f}")
    print(f"Pipeline 1 - XGBoost AUC: {pipeline1.xgb_auc:.3f}")
    print(f"Pipeline 1 - Total Score AUC: {pipeline1.total_score_auc:.3f}")
    print()
    
    return pipeline1

def test_existing_scorecard_reuse():
    """Test new functionality: pipeline accepts existing scorecard."""
    print("Testing new functionality (existing scorecard reuse)...")
    
    # Create a scorecard separately
    existing_scorecard = Scorecard(
        n_rows=800,
        n_features=5,
        binary_prevalence=0.2,
        beta=3.0,
        weights_override=0.5,  # All features have weight 0.5
        random_state=123
    )
    
    print(f"Created standalone scorecard - Shape: {existing_scorecard.features.shape}")
    print(f"Created standalone scorecard - Binary prevalence: {existing_scorecard.binary_outcome.mean():.3f}")
    print(f"Created standalone scorecard - Total score range: {existing_scorecard.total_scores['total_score'].min():.2f} - {existing_scorecard.total_scores['total_score'].max():.2f}")
    
    # Create pipeline with existing scorecard
    pipeline2 = ModelingPipeline(
        scorecard=existing_scorecard,
        test_size=0.25,  # Different test size
        random_state=456  # Different random state for train/val split
    )
    
    print(f"Pipeline 2 - Scorecard shape: {pipeline2.scorecard.features.shape}")
    print(f"Pipeline 2 - Binary prevalence: {pipeline2.scorecard.binary_outcome.mean():.3f}")
    print(f"Pipeline 2 - LogReg AUC: {pipeline2.logreg_auc:.3f}")
    print(f"Pipeline 2 - XGBoost AUC: {pipeline2.xgb_auc:.3f}")
    print(f"Pipeline 2 - Total Score AUC: {pipeline2.total_score_auc:.3f}")
    print()
    
    # Verify the scorecard object is the same
    assert pipeline2.scorecard is existing_scorecard, "Pipeline should use the same scorecard object"
    print("✓ Confirmed: Pipeline uses the same scorecard object")
    
    return pipeline2, existing_scorecard

def test_multiple_pipelines_same_scorecard():
    """Test that multiple pipelines can use the same scorecard with different configurations."""
    print("Testing multiple pipelines with same scorecard...")
    
    # Create one scorecard
    shared_scorecard = Scorecard(
        n_rows=1000,
        n_features=4,
        binary_prevalence=0.1,
        beta=2.0,
        random_state=789
    )
    
    # Create two pipelines with different train/validation splits
    pipeline_a = ModelingPipeline(
        scorecard=shared_scorecard,
        test_size=0.2,
        random_state=100
    )
    
    pipeline_b = ModelingPipeline(
        scorecard=shared_scorecard,
        test_size=0.4,
        random_state=200
    )
    
    print(f"Pipeline A - Training size: {len(pipeline_a.X_train)}, Validation size: {len(pipeline_a.X_val)}")
    print(f"Pipeline B - Training size: {len(pipeline_b.X_train)}, Validation size: {len(pipeline_b.X_val)}")
    print(f"Pipeline A - LogReg AUC: {pipeline_a.logreg_auc:.3f}")
    print(f"Pipeline B - LogReg AUC: {pipeline_b.logreg_auc:.3f}")
    
    # Verify both use the same scorecard
    assert pipeline_a.scorecard is shared_scorecard, "Pipeline A should use the shared scorecard"
    assert pipeline_b.scorecard is shared_scorecard, "Pipeline B should use the shared scorecard"
    assert pipeline_a.scorecard is pipeline_b.scorecard, "Both pipelines should use the same scorecard"
    print("✓ Confirmed: Both pipelines share the same scorecard object")
    print()

def test_error_handling():
    """Test error handling for invalid scorecard parameter."""
    print("Testing error handling...")
    
    try:
        # This should raise TypeError
        pipeline = ModelingPipeline(scorecard="not_a_scorecard")
        assert False, "Should have raised TypeError"
    except TypeError as e:
        print(f"✓ Correctly caught TypeError: {e}")
    
    try:
        # This should also raise TypeError
        pipeline = ModelingPipeline(scorecard=123)
        assert False, "Should have raised TypeError"
    except TypeError as e:
        print(f"✓ Correctly caught TypeError: {e}")
    
    print()

def test_backward_compatibility():
    """Test that existing code still works unchanged."""
    print("Testing backward compatibility...")
    
    # This is how the pipeline was used before (should still work)
    old_style_pipeline = ModelingPipeline(
        n_rows=300,
        n_features=3,
        binary_prevalence=0.12,
        beta=1.8
    )
    
    print(f"Backward compatibility test - Scorecard shape: {old_style_pipeline.scorecard.features.shape}")
    print(f"Backward compatibility test - AUC: {old_style_pipeline.logreg_auc:.3f}")
    print("✓ Backward compatibility confirmed")
    print()

if __name__ == "__main__":
    print("=" * 60)
    print("TESTING MODELINGPIPELINE SCORECARD REUSE FUNCTIONALITY")
    print("=" * 60)
    print()
    
    # Run all tests
    pipeline1 = test_new_scorecard_creation()
    pipeline2, scorecard = test_existing_scorecard_reuse()
    test_multiple_pipelines_same_scorecard()
    test_error_handling()
    test_backward_compatibility()
    
    print("=" * 60)
    print("ALL TESTS COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    print()
    
    # Summary
    print("Summary:")
    print(f"• New scorecard creation: ✓ Works (AUC: {pipeline1.logreg_auc:.3f})")
    print(f"• Existing scorecard reuse: ✓ Works (AUC: {pipeline2.logreg_auc:.3f})")
    print("• Multiple pipelines sharing scorecard: ✓ Works")
    print("• Error handling: ✓ Works")
    print("• Backward compatibility: ✓ Works")
