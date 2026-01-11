"""
Statistical Estimators for Concept Activation Analysis.

Implements variance analysis, bootstrap confidence intervals, and stability
metrics for mechanistic interpretability.

Reference: "Mechanistic Interpretability as Statistical Estimation" (Méloux et al., 2025)
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple, Union
import numpy as np
from collections import defaultdict


def compute_jaccard_similarity(set_a: Set[str], set_b: Set[str]) -> float:
    """
    Compute Jaccard similarity between two sets.

    J(A, B) = |A ∩ B| / |A ∪ B|

    Returns 1.0 for identical sets, 0.0 for disjoint sets.
    """
    if not set_a and not set_b:
        return 1.0  # Both empty = identical
    if not set_a or not set_b:
        return 0.0
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return intersection / union if union > 0 else 0.0


def compute_coefficient_of_variation(values: List[float]) -> float:
    """
    Compute coefficient of variation (CV = σ/μ).

    CV is a normalized measure of dispersion. Higher CV = more variable.
    Returns 0.0 if mean is 0 or list is empty.
    """
    if not values:
        return 0.0
    arr = np.array(values)
    mean = np.mean(arr)
    if mean == 0 or np.isnan(mean):
        return 0.0
    std = np.std(arr)
    return float(std / mean)


@dataclass
class StabilityMetrics:
    """Stability metrics for a concept or detection method."""

    # Central tendency
    mean: float = 0.0
    median: float = 0.0

    # Dispersion
    std: float = 0.0
    variance: float = 0.0
    cv: float = 0.0  # Coefficient of variation

    # Confidence interval (default 95%)
    ci_lower: float = 0.0
    ci_upper: float = 0.0
    confidence_level: float = 0.95

    # Robustness
    n_samples: int = 0
    fire_rate: float = 0.0  # Fraction of samples where concept fired above threshold

    # Structural stability (for concept sets)
    jaccard_mean: float = 0.0  # Mean pairwise Jaccard
    jaccard_std: float = 0.0

    def is_stable(self, cv_threshold: float = 0.3) -> bool:
        """Check if metrics indicate stable detection."""
        return self.cv < cv_threshold and self.n_samples >= 3

    def to_dict(self) -> Dict:
        return {
            'mean': self.mean,
            'median': self.median,
            'std': self.std,
            'variance': self.variance,
            'cv': self.cv,
            'ci_lower': self.ci_lower,
            'ci_upper': self.ci_upper,
            'confidence_level': self.confidence_level,
            'n_samples': self.n_samples,
            'fire_rate': self.fire_rate,
            'jaccard_mean': self.jaccard_mean,
            'jaccard_std': self.jaccard_std,
        }


@dataclass
class ConceptObservation:
    """Single observation of concept activations."""
    sample_id: str
    activations: Dict[str, float]  # concept -> activation score
    top_k_concepts: List[str]  # Concepts in top-k for this sample
    metadata: Dict = field(default_factory=dict)


class ActivationDistribution:
    """
    Track distribution of concept activations across samples.

    Collects observations and computes statistical properties including
    confidence intervals, stability metrics, and robustness indicators.
    """

    def __init__(
        self,
        threshold: float = 0.5,
        top_k: int = 10,
    ):
        """
        Args:
            threshold: Activation threshold for "firing" (default from calibration)
            top_k: Number of top concepts to track per sample
        """
        self.threshold = threshold
        self.top_k = top_k

        # Store observations
        self.observations: List[ConceptObservation] = []

        # Per-concept activation history
        self._concept_activations: Dict[str, List[float]] = defaultdict(list)

        # Top-k concept sets per sample (for Jaccard)
        self._topk_sets: List[Set[str]] = []

    def add_observation(
        self,
        activations: Dict[str, float],
        sample_id: Optional[str] = None,
        metadata: Optional[Dict] = None,
    ) -> None:
        """
        Add an observation of concept activations.

        Args:
            activations: Dict mapping concept name to activation score
            sample_id: Optional identifier for this sample
            metadata: Optional metadata (e.g., episode_id, condition)
        """
        if sample_id is None:
            sample_id = f"sample_{len(self.observations)}"

        # Get top-k concepts
        sorted_concepts = sorted(activations.items(), key=lambda x: x[1], reverse=True)
        top_k_concepts = [c for c, _ in sorted_concepts[:self.top_k]]

        obs = ConceptObservation(
            sample_id=sample_id,
            activations=activations,
            top_k_concepts=top_k_concepts,
            metadata=metadata or {},
        )
        self.observations.append(obs)

        # Update per-concept history
        for concept, score in activations.items():
            self._concept_activations[concept].append(score)

        # Store top-k set for Jaccard computation
        self._topk_sets.append(set(top_k_concepts))

    def get_concept_metrics(self, concept: str) -> StabilityMetrics:
        """
        Compute stability metrics for a specific concept.

        Args:
            concept: Concept name

        Returns:
            StabilityMetrics with mean, std, CI, etc.
        """
        scores = self._concept_activations.get(concept, [])

        if not scores:
            return StabilityMetrics(n_samples=0)

        arr = np.array(scores)
        n = len(arr)

        # Basic statistics
        mean = float(np.mean(arr))
        median = float(np.median(arr))
        std = float(np.std(arr))
        variance = float(np.var(arr))
        cv = compute_coefficient_of_variation(scores)

        # Fire rate (fraction above threshold)
        fire_rate = float(np.mean(arr > self.threshold))

        # Bootstrap confidence interval
        ci_lower, ci_upper = self._bootstrap_ci(arr, confidence=0.95)

        return StabilityMetrics(
            mean=mean,
            median=median,
            std=std,
            variance=variance,
            cv=cv,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            confidence_level=0.95,
            n_samples=n,
            fire_rate=fire_rate,
        )

    def get_all_concept_metrics(self) -> Dict[str, StabilityMetrics]:
        """Get stability metrics for all observed concepts."""
        return {
            concept: self.get_concept_metrics(concept)
            for concept in self._concept_activations.keys()
        }

    def get_topk_stability(self) -> StabilityMetrics:
        """
        Compute stability of top-k concept sets across samples.

        Uses pairwise Jaccard similarity to measure structural stability.
        High Jaccard = same concepts fire across samples (stable).
        Low Jaccard = different concepts fire (unstable/polysemantic).
        """
        if len(self._topk_sets) < 2:
            return StabilityMetrics(n_samples=len(self._topk_sets), jaccard_mean=1.0)

        # Compute pairwise Jaccard
        jaccards = []
        for i in range(len(self._topk_sets)):
            for j in range(i + 1, len(self._topk_sets)):
                j_score = compute_jaccard_similarity(
                    self._topk_sets[i], self._topk_sets[j]
                )
                jaccards.append(j_score)

        jaccard_mean = float(np.mean(jaccards))
        jaccard_std = float(np.std(jaccards))

        return StabilityMetrics(
            mean=jaccard_mean,
            std=jaccard_std,
            cv=compute_coefficient_of_variation(jaccards),
            n_samples=len(self._topk_sets),
            jaccard_mean=jaccard_mean,
            jaccard_std=jaccard_std,
        )

    def get_stable_concepts(self, fire_rate_threshold: float = 0.8) -> List[str]:
        """
        Get concepts that fire consistently across samples.

        Args:
            fire_rate_threshold: Minimum fraction of samples where concept must fire

        Returns:
            List of stable concept names
        """
        stable = []
        for concept in self._concept_activations.keys():
            metrics = self.get_concept_metrics(concept)
            if metrics.fire_rate >= fire_rate_threshold:
                stable.append(concept)
        return stable

    def get_sporadic_concepts(self, fire_rate_threshold: float = 0.3) -> List[str]:
        """
        Get concepts that fire sporadically (potential polysemantic noise).

        Args:
            fire_rate_threshold: Maximum fire rate for sporadic classification

        Returns:
            List of sporadic concept names
        """
        sporadic = []
        for concept in self._concept_activations.keys():
            metrics = self.get_concept_metrics(concept)
            if 0 < metrics.fire_rate < fire_rate_threshold:
                sporadic.append(concept)
        return sporadic

    def confidence_interval(
        self,
        concept: str,
        confidence: float = 0.95,
    ) -> Tuple[float, float]:
        """
        Get bootstrap confidence interval for concept activation.

        Args:
            concept: Concept name
            confidence: Confidence level (default 0.95)

        Returns:
            (lower_bound, upper_bound) tuple
        """
        scores = self._concept_activations.get(concept, [])
        if not scores:
            return (0.0, 0.0)
        return self._bootstrap_ci(np.array(scores), confidence)

    def _bootstrap_ci(
        self,
        data: np.ndarray,
        confidence: float = 0.95,
        n_bootstrap: int = 1000,
    ) -> Tuple[float, float]:
        """
        Compute bootstrap confidence interval for the mean.

        Uses percentile method for simplicity and robustness.
        """
        if len(data) < 2:
            return (float(data[0]) if len(data) == 1 else 0.0, float(data[0]) if len(data) == 1 else 0.0)

        # Bootstrap resampling
        bootstrap_means = []
        for _ in range(n_bootstrap):
            sample = np.random.choice(data, size=len(data), replace=True)
            bootstrap_means.append(np.mean(sample))

        # Percentile confidence interval
        alpha = 1 - confidence
        lower = np.percentile(bootstrap_means, 100 * alpha / 2)
        upper = np.percentile(bootstrap_means, 100 * (1 - alpha / 2))

        return (float(lower), float(upper))

    def clear(self) -> None:
        """Clear all observations."""
        self.observations.clear()
        self._concept_activations.clear()
        self._topk_sets.clear()


class BootstrapEstimator:
    """
    Bootstrap estimation for detection robustness.

    Runs multiple bootstrap resamples to estimate variance of detection
    scores and classification decisions.
    """

    def __init__(
        self,
        n_bootstrap: int = 100,
        sample_fraction: float = 0.8,
        random_seed: Optional[int] = None,
    ):
        """
        Args:
            n_bootstrap: Number of bootstrap resamples
            sample_fraction: Fraction of data to sample each iteration
            random_seed: Random seed for reproducibility
        """
        self.n_bootstrap = n_bootstrap
        self.sample_fraction = sample_fraction
        self.rng = np.random.RandomState(random_seed)

    def estimate_detection_variance(
        self,
        observations: List[ConceptObservation],
        concept: str,
        threshold: float = 0.5,
    ) -> Dict[str, float]:
        """
        Estimate variance of detection for a concept via bootstrap.

        Args:
            observations: List of ConceptObservation
            concept: Concept to analyze
            threshold: Detection threshold

        Returns:
            Dict with variance statistics
        """
        if not observations:
            return {'mean': 0.0, 'std': 0.0, 'cv': 0.0, 'detection_rate': 0.0}

        # Get all scores for this concept
        all_scores = [
            obs.activations.get(concept, 0.0)
            for obs in observations
        ]

        # Bootstrap
        bootstrap_means = []
        bootstrap_detection_rates = []

        n_samples = max(1, int(len(observations) * self.sample_fraction))

        for _ in range(self.n_bootstrap):
            indices = self.rng.choice(len(all_scores), size=n_samples, replace=True)
            sample = [all_scores[i] for i in indices]

            bootstrap_means.append(np.mean(sample))
            bootstrap_detection_rates.append(np.mean(np.array(sample) > threshold))

        return {
            'mean': float(np.mean(bootstrap_means)),
            'std': float(np.std(bootstrap_means)),
            'cv': compute_coefficient_of_variation(bootstrap_means),
            'detection_rate': float(np.mean(bootstrap_detection_rates)),
            'detection_rate_std': float(np.std(bootstrap_detection_rates)),
            'ci_lower': float(np.percentile(bootstrap_means, 2.5)),
            'ci_upper': float(np.percentile(bootstrap_means, 97.5)),
        }

    def estimate_classification_robustness(
        self,
        observations: List[ConceptObservation],
        concept: str,
        threshold: float = 0.5,
    ) -> Dict[str, float]:
        """
        Estimate robustness of pass/fail classification.

        Returns probability that classification would change with different samples.
        """
        if not observations:
            return {'robustness': 0.0, 'flip_rate': 1.0}

        all_scores = [obs.activations.get(concept, 0.0) for obs in observations]

        # Overall classification
        overall_detected = np.mean(all_scores) > threshold

        # Bootstrap to see how often classification flips
        flips = 0
        n_samples = max(1, int(len(observations) * self.sample_fraction))

        for _ in range(self.n_bootstrap):
            indices = self.rng.choice(len(all_scores), size=n_samples, replace=True)
            sample = [all_scores[i] for i in indices]
            sample_detected = np.mean(sample) > threshold

            if sample_detected != overall_detected:
                flips += 1

        flip_rate = flips / self.n_bootstrap
        robustness = 1.0 - flip_rate

        return {
            'robustness': robustness,
            'flip_rate': flip_rate,
            'overall_detected': overall_detected,
        }


@dataclass
class BehaviorStatistics:
    """
    Aggregated statistics for a manipulation behavior across samples.

    Used by evaluation runner to report multi-sample results with
    proper statistical uncertainty quantification.
    """
    behavior: str
    condition: str

    # Detection statistics
    n_samples: int = 0
    mean_score: float = 0.0
    std_score: float = 0.0
    cv: float = 0.0
    ci_lower: float = 0.0
    ci_upper: float = 0.0

    # Classification breakdown
    pass_rate: float = 0.0  # Fraction of samples that passed
    fail_rate: float = 0.0  # Fraction detected as manipulation
    null_rate: float = 0.0  # Inconclusive

    # Confidence in classification
    avg_confidence: float = 0.0
    confidence_stddev: float = 0.0

    # Concept stability
    stable_concepts: List[str] = field(default_factory=list)
    sporadic_concepts: List[str] = field(default_factory=list)
    concept_jaccard_mean: float = 0.0

    # Intervention statistics (for condition C/F)
    avg_interventions: float = 0.0
    intervention_stddev: float = 0.0

    def to_dict(self) -> Dict:
        return {
            'behavior': self.behavior,
            'condition': self.condition,
            'n_samples': self.n_samples,
            'mean_score': self.mean_score,
            'std_score': self.std_score,
            'cv': self.cv,
            'ci_lower': self.ci_lower,
            'ci_upper': self.ci_upper,
            'pass_rate': self.pass_rate,
            'fail_rate': self.fail_rate,
            'null_rate': self.null_rate,
            'avg_confidence': self.avg_confidence,
            'confidence_stddev': self.confidence_stddev,
            'stable_concepts': self.stable_concepts,
            'sporadic_concepts': self.sporadic_concepts,
            'concept_jaccard_mean': self.concept_jaccard_mean,
            'avg_interventions': self.avg_interventions,
            'intervention_stddev': self.intervention_stddev,
        }


@dataclass
class CalibrationConfidence:
    """
    Confidence metrics for calibration analysis.

    Tracks uncertainty in lens firing patterns to identify
    reliable vs unstable concept detections.
    """
    concept: str
    layer: int

    # Rank distribution
    rank_mean: float = 0.0
    rank_std: float = 0.0
    rank_ci_lower: float = 0.0
    rank_ci_upper: float = 0.0

    # Activation distribution
    activation_mean: float = 0.0
    activation_std: float = 0.0
    activation_ci_lower: float = 0.0
    activation_ci_upper: float = 0.0

    # Classification stability
    detection_rate: float = 0.0  # In-top-k rate
    detection_rate_ci_lower: float = 0.0
    detection_rate_ci_upper: float = 0.0

    # Robustness
    cv: float = 0.0  # Coefficient of variation for ranks
    is_stable: bool = True  # CV < threshold
    n_samples: int = 0

    def to_dict(self) -> Dict:
        return {
            'concept': self.concept,
            'layer': self.layer,
            'rank_mean': self.rank_mean,
            'rank_std': self.rank_std,
            'rank_ci': [self.rank_ci_lower, self.rank_ci_upper],
            'activation_mean': self.activation_mean,
            'activation_std': self.activation_std,
            'activation_ci': [self.activation_ci_lower, self.activation_ci_upper],
            'detection_rate': self.detection_rate,
            'detection_rate_ci': [self.detection_rate_ci_lower, self.detection_rate_ci_upper],
            'cv': self.cv,
            'is_stable': self.is_stable,
            'n_samples': self.n_samples,
        }


def compute_calibration_confidence(
    ranks: List[int],
    activations: List[float],
    top_k: int = 10,
    concept: str = "",
    layer: int = 0,
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    cv_threshold: float = 0.5,
) -> CalibrationConfidence:
    """
    Compute confidence metrics for a lens during calibration.

    Args:
        ranks: List of ranks across probes
        activations: List of activation scores across probes
        top_k: Threshold for "in top-k" detection
        concept: Concept name
        layer: Layer number
        n_bootstrap: Number of bootstrap resamples
        confidence: Confidence level for intervals
        cv_threshold: CV threshold for stability classification

    Returns:
        CalibrationConfidence with all metrics
    """
    if not ranks or not activations:
        return CalibrationConfidence(concept=concept, layer=layer)

    n = len(ranks)
    rank_arr = np.array(ranks)
    act_arr = np.array(activations)

    # Basic statistics
    rank_mean = float(np.mean(rank_arr))
    rank_std = float(np.std(rank_arr))
    act_mean = float(np.mean(act_arr))
    act_std = float(np.std(act_arr))

    # Detection rate
    in_top_k = rank_arr <= top_k
    detection_rate = float(np.mean(in_top_k))

    # CV for ranks
    cv = rank_std / rank_mean if rank_mean > 0 else 0.0

    # Bootstrap CIs
    alpha = 1 - confidence

    if n >= 2:
        # Rank CI
        rank_boot = []
        act_boot = []
        det_boot = []

        for _ in range(n_bootstrap):
            idx = np.random.choice(n, size=n, replace=True)
            rank_boot.append(np.mean(rank_arr[idx]))
            act_boot.append(np.mean(act_arr[idx]))
            det_boot.append(np.mean(rank_arr[idx] <= top_k))

        rank_ci_lower = float(np.percentile(rank_boot, 100 * alpha / 2))
        rank_ci_upper = float(np.percentile(rank_boot, 100 * (1 - alpha / 2)))

        act_ci_lower = float(np.percentile(act_boot, 100 * alpha / 2))
        act_ci_upper = float(np.percentile(act_boot, 100 * (1 - alpha / 2)))

        det_ci_lower = float(np.percentile(det_boot, 100 * alpha / 2))
        det_ci_upper = float(np.percentile(det_boot, 100 * (1 - alpha / 2)))
    else:
        rank_ci_lower = rank_ci_upper = rank_mean
        act_ci_lower = act_ci_upper = act_mean
        det_ci_lower = det_ci_upper = detection_rate

    return CalibrationConfidence(
        concept=concept,
        layer=layer,
        rank_mean=rank_mean,
        rank_std=rank_std,
        rank_ci_lower=rank_ci_lower,
        rank_ci_upper=rank_ci_upper,
        activation_mean=act_mean,
        activation_std=act_std,
        activation_ci_lower=act_ci_lower,
        activation_ci_upper=act_ci_upper,
        detection_rate=detection_rate,
        detection_rate_ci_lower=det_ci_lower,
        detection_rate_ci_upper=det_ci_upper,
        cv=cv,
        is_stable=cv < cv_threshold,
        n_samples=n,
    )


class CalibrationDistribution:
    """
    Track distributions during calibration for multi-probe analysis.

    This extends ActivationDistribution with calibration-specific features:
    - Rank tracking per lens
    - Cross-concept interference detection
    - Per-layer expected frequency comparison
    """

    def __init__(
        self,
        top_k: int = 10,
        expected_freq_by_layer: Optional[Dict[int, float]] = None,
    ):
        self.top_k = top_k
        self.expected_freq_by_layer = expected_freq_by_layer or {
            0: 0.20, 1: 0.10, 2: 0.05, 3: 0.05, 4: 0.01, 5: 0.01, 6: 0.01,
        }

        # Per-lens tracking: (concept, layer) -> lists
        self._ranks: Dict[Tuple[str, int], List[int]] = defaultdict(list)
        self._activations: Dict[Tuple[str, int], List[float]] = defaultdict(list)
        self._over_fires: Dict[Tuple[str, int], List[str]] = defaultdict(list)

        # Per-probe tracking
        self._probe_count = 0
        self._top_k_sets: List[Set[str]] = []

    def add_probe(
        self,
        prompted_concept: str,
        scores: List[Tuple[str, float, int]],  # [(concept, score, layer), ...]
    ) -> None:
        """
        Record results of a single calibration probe.

        Args:
            prompted_concept: The concept that was prompted
            scores: Sorted list of (concept, activation, layer) tuples
        """
        self._probe_count += 1

        # Track top-k set for Jaccard
        top_k_concepts = {s[0] for s in scores[:self.top_k]}
        self._top_k_sets.append(top_k_concepts)

        # Record ranks and activations for each lens
        for rank, (concept, score, layer) in enumerate(scores, 1):
            key = (concept, layer)
            self._ranks[key].append(rank)
            self._activations[key].append(score)

            # Track over-firing (in top-k for wrong concept)
            if rank <= self.top_k and concept != prompted_concept:
                self._over_fires[key].append(prompted_concept)

    def get_lens_confidence(
        self,
        concept: str,
        layer: int,
        n_bootstrap: int = 1000,
    ) -> CalibrationConfidence:
        """Get confidence metrics for a specific lens."""
        key = (concept, layer)
        ranks = self._ranks.get(key, [])
        activations = self._activations.get(key, [])

        return compute_calibration_confidence(
            ranks=ranks,
            activations=activations,
            top_k=self.top_k,
            concept=concept,
            layer=layer,
            n_bootstrap=n_bootstrap,
        )

    def get_all_lens_confidence(self) -> Dict[Tuple[str, int], CalibrationConfidence]:
        """Get confidence metrics for all tracked lenses."""
        result = {}
        for key in self._ranks.keys():
            concept, layer = key
            result[key] = self.get_lens_confidence(concept, layer)
        return result

    def get_topk_jaccard_stability(self) -> StabilityMetrics:
        """Compute Jaccard stability of top-k concept sets across probes."""
        if len(self._top_k_sets) < 2:
            return StabilityMetrics(n_samples=len(self._top_k_sets), jaccard_mean=1.0)

        jaccards = []
        for i in range(len(self._top_k_sets)):
            for j in range(i + 1, len(self._top_k_sets)):
                j_score = compute_jaccard_similarity(
                    self._top_k_sets[i], self._top_k_sets[j]
                )
                jaccards.append(j_score)

        return StabilityMetrics(
            mean=float(np.mean(jaccards)),
            std=float(np.std(jaccards)),
            cv=compute_coefficient_of_variation(jaccards),
            n_samples=len(self._top_k_sets),
            jaccard_mean=float(np.mean(jaccards)),
            jaccard_std=float(np.std(jaccards)),
        )

    def get_over_firing_analysis(
        self,
        concept: str,
        layer: int,
    ) -> Dict[str, Union[int, float, List[str]]]:
        """Analyze over-firing patterns for a lens."""
        key = (concept, layer)
        over_fires = self._over_fires.get(key, [])
        ranks = self._ranks.get(key, [])

        if not ranks:
            return {
                'over_fire_count': 0,
                'over_fire_rate': 0.0,
                'over_fires_on': [],
                'expected_frequency': self.expected_freq_by_layer.get(layer, 0.01),
            }

        in_top_k_count = sum(1 for r in ranks if r <= self.top_k)
        over_fire_rate = len(over_fires) / len(ranks) if ranks else 0.0
        expected = self.expected_freq_by_layer.get(layer, 0.01)

        # Count unique concepts over-fired on
        from collections import Counter
        over_fire_counts = Counter(over_fires)
        top_over_fires = [c for c, _ in over_fire_counts.most_common(10)]

        return {
            'over_fire_count': len(over_fires),
            'over_fire_rate': over_fire_rate,
            'in_top_k_count': in_top_k_count,
            'in_top_k_rate': in_top_k_count / len(ranks),
            'over_fires_on': top_over_fires,
            'expected_frequency': expected,
            'z_score': self._compute_z_score(in_top_k_count / len(ranks), expected, len(ranks)),
        }

    def _compute_z_score(
        self,
        observed: float,
        expected: float,
        n: int,
    ) -> float:
        """Compute z-score for frequency deviation."""
        if n == 0:
            return 0.0
        std = np.sqrt(expected * (1 - expected) / n)
        if std == 0:
            return 0.0
        return (observed - expected) / std

    def clear(self) -> None:
        """Clear all tracked data."""
        self._ranks.clear()
        self._activations.clear()
        self._over_fires.clear()
        self._top_k_sets.clear()
        self._probe_count = 0


def aggregate_sample_results(
    results: List[Dict],
    behavior: str,
    condition: str,
    score_key: str = 'peak_detection_score',
    threshold: float = 0.5,
) -> BehaviorStatistics:
    """
    Aggregate multiple sample results into BehaviorStatistics.

    Args:
        results: List of per-sample result dicts
        behavior: Behavior name (e.g., 'sycophancy')
        condition: Condition name (e.g., 'A', 'B', 'C')
        score_key: Key in result dict for detection score
        threshold: Detection threshold

    Returns:
        BehaviorStatistics with aggregated metrics
    """
    if not results:
        return BehaviorStatistics(behavior=behavior, condition=condition)

    n = len(results)
    scores = [r.get(score_key, 0.0) for r in results]
    confidences = [r.get('confidence', r.get(score_key, 0.5)) for r in results]
    interventions = [r.get('interventions', 0) for r in results]

    # Classification counts
    pass_count = sum(1 for r in results if r.get('score_result') == 'pass')
    fail_count = sum(1 for r in results if r.get('score_result') == 'fail' or r.get('manipulation_detected'))
    null_count = n - pass_count - fail_count

    # Concept activation tracking
    activation_dist = ActivationDistribution(threshold=threshold)
    for r in results:
        if 'concept_activations' in r:
            activation_dist.add_observation(r['concept_activations'])
        elif 'ticks' in r:
            # Aggregate concept activations from ticks
            tick_activations = {}
            for tick in r.get('ticks', []):
                for concept, score in tick.get('concept_activations', {}).items():
                    if concept not in tick_activations or score > tick_activations[concept]:
                        tick_activations[concept] = score
            if tick_activations:
                activation_dist.add_observation(tick_activations)

    # Compute statistics
    mean_score = float(np.mean(scores))
    std_score = float(np.std(scores))
    cv = compute_coefficient_of_variation(scores)

    # Bootstrap CI
    if n >= 2:
        bootstrap_means = []
        for _ in range(1000):
            sample = np.random.choice(scores, size=n, replace=True)
            bootstrap_means.append(np.mean(sample))
        ci_lower = float(np.percentile(bootstrap_means, 2.5))
        ci_upper = float(np.percentile(bootstrap_means, 97.5))
    else:
        ci_lower = ci_upper = mean_score

    # Concept stability
    topk_stability = activation_dist.get_topk_stability()
    stable_concepts = activation_dist.get_stable_concepts()
    sporadic_concepts = activation_dist.get_sporadic_concepts()

    return BehaviorStatistics(
        behavior=behavior,
        condition=condition,
        n_samples=n,
        mean_score=mean_score,
        std_score=std_score,
        cv=cv,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        pass_rate=pass_count / n if n > 0 else 0.0,
        fail_rate=fail_count / n if n > 0 else 0.0,
        null_rate=null_count / n if n > 0 else 0.0,
        avg_confidence=float(np.mean(confidences)),
        confidence_stddev=float(np.std(confidences)),
        stable_concepts=stable_concepts,
        sporadic_concepts=sporadic_concepts,
        concept_jaccard_mean=topk_stability.jaccard_mean,
        avg_interventions=float(np.mean(interventions)),
        intervention_stddev=float(np.std(interventions)),
    )
