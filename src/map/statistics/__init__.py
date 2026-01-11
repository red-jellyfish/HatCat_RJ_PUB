"""
Statistical Estimation for Mechanistic Interpretability.

This module provides statistical rigor for concept detection and steering,
treating interpretability methods as statistical estimators with variance
and robustness properties.

Based on: "Mechanistic Interpretability as Statistical Estimation" (Méloux et al., 2025)

Key concepts:
- Variance: Sensitivity to data resampling (bootstrap)
- Robustness: Stability under hyperparameter/method changes
- Confidence intervals for detection scores
- Jaccard similarity for concept activation stability

Usage:
    from src.map.statistics import ActivationDistribution, BootstrapEstimator

    # Track activations across samples
    dist = ActivationDistribution()
    for sample in samples:
        dist.add_observation(concept_activations)

    # Get confidence interval
    ci = dist.confidence_interval("Deception", confidence=0.95)

    # Measure concept stability
    stability = dist.concept_jaccard_similarity()
"""

from .estimator import (
    # Core statistical primitives
    compute_jaccard_similarity,
    compute_coefficient_of_variation,

    # Stability metrics
    StabilityMetrics,
    ConceptObservation,

    # Activation tracking
    ActivationDistribution,
    BootstrapEstimator,

    # Calibration integration
    CalibrationConfidence,
    CalibrationDistribution,
    compute_calibration_confidence,

    # Evaluation aggregation
    BehaviorStatistics,
    aggregate_sample_results,
)

__all__ = [
    # Core statistical primitives
    'compute_jaccard_similarity',
    'compute_coefficient_of_variation',

    # Stability metrics
    'StabilityMetrics',
    'ConceptObservation',

    # Activation tracking
    'ActivationDistribution',
    'BootstrapEstimator',

    # Calibration integration
    'CalibrationConfidence',
    'CalibrationDistribution',
    'compute_calibration_confidence',

    # Evaluation aggregation
    'BehaviorStatistics',
    'aggregate_sample_results',
]
