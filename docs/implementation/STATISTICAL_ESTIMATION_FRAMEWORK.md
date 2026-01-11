# Statistical Estimation Framework for Mechanistic Interpretability

Based on: "Mechanistic Interpretability as Statistical Estimation" (Méloux et al., 2025)

## Overview

This document describes the statistical estimation framework integrated into HatCat core to provide proper uncertainty quantification for concept detection and steering. The framework treats interpretability methods as statistical estimators with variance and robustness properties.

## Key Findings from Méloux et al.

The paper demonstrates that mechanistic interpretability results exhibit significant statistical variance:

1. **Non-identifiability**: Multiple distinct circuits can satisfy the same evaluation criteria
2. **Bootstrap variance**: Circuits discovered via bootstrap resampling show Jaccard similarity ~0.56 (not 1.0)
3. **Multimodal distributions**: Detection scores often show bimodal or multimodal distributions
4. **Hyperparameter sensitivity**: Results vary significantly with method choices

### Implications for HatCat

- Single-run detection scores are insufficient - we need confidence intervals
- Top-k concept sets may vary across samples - we need Jaccard stability metrics
- Calibration must account for variance, not just point estimates

## Implementation

### Module Location

```
src/map/statistics/
├── __init__.py
└── estimator.py
```

### Core Components

#### 1. ActivationDistribution

Tracks concept activations across samples with bootstrap confidence intervals:

```python
from src.map.statistics import ActivationDistribution

dist = ActivationDistribution(threshold=0.5, top_k=10)

# Add observations from multiple samples
for sample in samples:
    dist.add_observation(concept_activations, sample_id=sample.id)

# Get stability metrics for a concept
metrics = dist.get_concept_metrics("Deception")
# Returns: mean, std, cv, ci_lower, ci_upper, fire_rate

# Get top-k structural stability
stability = dist.get_topk_stability()
# Returns: jaccard_mean, jaccard_std (how consistent is top-k across samples)

# Get reliably firing concepts
stable = dist.get_stable_concepts(fire_rate_threshold=0.8)
```

#### 2. CalibrationConfidence

Calibration-specific metrics with rank and activation confidence intervals:

```python
from src.map.statistics import compute_calibration_confidence

conf = compute_calibration_confidence(
    ranks=[3, 5, 2, 4, 3],          # Ranks across probes
    activations=[0.8, 0.7, 0.9, 0.75, 0.82],
    top_k=10,
    concept="Deception",
    layer=3,
    n_bootstrap=1000,
)

# Access confidence metrics
print(f"Rank: {conf.rank_mean:.1f} [{conf.rank_ci_lower:.1f}, {conf.rank_ci_upper:.1f}]")
print(f"Detection rate: {conf.detection_rate:.1%} CI: [{conf.detection_rate_ci_lower:.1%}, {conf.detection_rate_ci_upper:.1%}]")
print(f"Stable: {conf.is_stable} (CV={conf.cv:.2f})")
```

#### 3. CalibrationDistribution

Tracks distributions during calibration for multi-probe analysis:

```python
from src.map.statistics import CalibrationDistribution

calib_dist = CalibrationDistribution(top_k=10)

# Record calibration probes
for probe in probes:
    calib_dist.add_probe(
        prompted_concept=probe.concept,
        scores=probe.all_lens_scores,  # [(concept, score, layer), ...]
    )

# Get lens-level confidence
conf = calib_dist.get_lens_confidence("Deception", layer=3)

# Get over-firing analysis
analysis = calib_dist.get_over_firing_analysis("Deception", layer=3)
```

#### 4. BehaviorStatistics

Aggregated statistics for multi-sample evaluation:

```python
from src.map.statistics import aggregate_sample_results

stats = aggregate_sample_results(
    results=episode_results,  # List of per-sample result dicts
    behavior="sycophancy",
    condition="C",
    score_key="peak_detection_score",
    threshold=0.5,
)

# Access aggregated metrics
print(f"Mean score: {stats.mean_score:.3f} +/- {stats.std_score:.3f}")
print(f"95% CI: [{stats.ci_lower:.3f}, {stats.ci_upper:.3f}]")
print(f"CV: {stats.cv:.2f}")
print(f"Pass rate: {stats.pass_rate:.1%}")
print(f"Stable concepts: {stats.stable_concepts}")
print(f"Concept Jaccard: {stats.concept_jaccard_mean:.3f}")
```

## Integration Points

### Calibration Matrix (src/map/training/calibration/matrix.py)

The `CalibrationMatrixBuilder` now computes:

- **Per-lens confidence intervals**: rank_ci, activation_ci, detection_rate_ci
- **Coefficient of variation**: Stability measure for each lens
- **Top-k Jaccard stability**: Structural consistency across probes
- **Stable/unstable lens counts**: Summary of reliable vs unreliable lenses

New fields in `LensStatistics`:
```python
rank_ci_lower: float
rank_ci_upper: float
activation_ci_lower: float
activation_ci_upper: float
detection_rate_ci_lower: float
detection_rate_ci_upper: float
cv: float  # Coefficient of variation
is_stable: bool  # CV < 0.5
```

New fields in `CalibrationMatrix`:
```python
topk_jaccard_mean: float  # Structural stability
topk_jaccard_std: float
stable_lens_count: int
unstable_lens_count: int
```

### Detection (src/hush/hush_integration.py)

The `WorldTick` dataclass includes:
```python
concept_confidence: Dict[str, Tuple[float, float]]  # concept -> (ci_lower, ci_upper)
detection_cv: float  # Coefficient of variation
```

New function for post-hoc aggregation:
```python
from src.hush.hush_integration import aggregate_ticks_with_confidence

result = aggregate_ticks_with_confidence(
    tick_lists=[run1_ticks, run2_ticks, run3_ticks],
    confidence=0.95,
    n_bootstrap=1000,
)

# Access aggregated results
print(f"Safety: {result['safety_stats']['mean']:.3f} CI: [{result['safety_stats']['ci_lower']:.3f}, {result['safety_stats']['ci_upper']:.3f}]")
print(f"Top-k Jaccard: {result['stability']['topk_jaccard_mean']:.3f}")
```

## Time Estimates

### First-Light Lens Pack (gemma-3-4b_first-light-v1)

Pack characteristics:
- **7,947 lenses** (.pt files)
- **7,947 concepts** across 7 layers (0-6)
- Layer distribution: L0=5, L1=23, L2=241, L3=1283, L4=2603, L5=2687, L6=1105

### Full Calibration Matrix (N×N)

Building the complete calibration matrix requires:
- N² lens evaluations = 7,947² = 63.2M evaluations
- Per-probe: ~50ms model forward pass + ~10ms lens scoring
- Total: **~35-40 hours on single GPU**

Recommended: Use sampling (`sample_rate=0.1`) for initial analysis:
- 10% sample: ~795 probes × 7,947 lenses = 6.3M evaluations
- Time: **~3-4 hours**

### Statistical Estimation Add-on

The statistical estimation adds minimal overhead:
- Bootstrap CI (500 resamples): ~5ms per lens
- Total for 7,947 lenses: ~40 seconds
- Jaccard computation (100 sampled pairs): ~100ms

**Net impact: <1% additional time**

### Per-Lens Calibration Analysis

If running `calibration.analysis` (not full matrix):
- 1 probe per concept = 7,947 probes
- Each probe scores all lenses: 7,947 × 7,947 = 63.2M scores
- But using batched inference: ~0.1ms per lens batch
- Total: **~20-30 minutes**

### Recommended Workflow

1. **Quick validation** (5-10 min): Sample 100 concepts, fast mode
   ```bash
   python -m training.calibration.matrix \
       --lens-pack lens_packs/gemma-3-4b_first-light-v1 \
       --concept-pack concept_packs/first-light \
       --model google/gemma-3-4b-pt \
       --fast-mode --max-concepts 100
   ```

2. **Layer-specific analysis** (30-60 min): Focus on layers 3-5
   ```bash
   python -m training.calibration.matrix \
       --lens-pack lens_packs/gemma-3-4b_first-light-v1 \
       --concept-pack concept_packs/first-light \
       --model google/gemma-3-4b-pt \
       --fast-mode --layers 3 4 5
   ```

3. **Full calibration** (3-4 hours): 10% sample with CIs
   ```bash
   python -m training.calibration.matrix \
       --lens-pack lens_packs/gemma-3-4b_first-light-v1 \
       --concept-pack concept_packs/first-light \
       --model google/gemma-3-4b-pt \
       --fast-mode --sample-rate 0.1
   ```

## Interpreting Results

### Stability Thresholds

| Metric | Stable | Marginal | Unstable |
|--------|--------|----------|----------|
| CV (Coefficient of Variation) | < 0.3 | 0.3-0.5 | > 0.5 |
| Top-k Jaccard | > 0.7 | 0.5-0.7 | < 0.5 |
| Fire Rate | > 0.8 | 0.5-0.8 | < 0.5 |
| CI Width (relative) | < 20% | 20-40% | > 40% |

### Expected Values (from Méloux et al.)

- Bootstrap Jaccard for circuits: ~0.56 (significant variance is normal)
- Expect 10-30% of lenses to be "unstable" by CV metric
- Over-firing lenses typically have z-score > 2.0

### Actionable Insights

1. **Unstable lenses** (high CV): May need more training data or indicate polysemantic concepts
2. **Low Jaccard**: Top-k concepts vary across samples - consider reporting top-k with confidence
3. **Wide CIs**: Insufficient samples - run more probes before drawing conclusions
4. **High over-fire rate**: Lens is too general - consider hierarchical refinement

## Relationship to Calibration Pipeline

Statistical estimation provides confidence intervals and variance tracking that complements the calibration pipeline:

- **Calibration produces** CFR, GFR, self_mean, cross_mean per lens
- **Statistical estimation adds** confidence intervals, CV, Jaccard stability
- **Together they enable** reliable HUSH threshold configuration

The key insight is that calibration metrics (like CFR) have their own variance across samples. A lens with CFR=15% might actually be anywhere from 10-20% depending on sample variance. Statistical estimation quantifies this uncertainty.

For full calibration pipeline documentation, see: `docs/approach/CALIBRATION.md`

## References

- Méloux, M., et al. (2025). "Mechanistic Interpretability as Statistical Estimation"
- HatCat calibration documentation: `docs/approach/CALIBRATION.md`
- Significance scoring: `docs/implementation/HUSH_SIGNIFICANCE_SCORING.md`
- Runtime normalization: `src/hat/monitoring/deployment_manifest.py` (ConceptCalibration)
