# Pack-Level Lens Calibration

## Overview

After training individual lenses, pack-level calibration identifies and corrects lenses that fire inappropriately relative to other lenses in the ensemble. This document describes the calibration framework, metrics, and tuning approach.

## Triple-Criteria Calibration (Current Approach)

The current calibration system uses three criteria to determine if a lens is well-calibrated:

### 1. Ancestor Criterion

**Test**: When we prompt with concept A's name, does lens A score higher than all of A's ancestor lenses?

**Pass condition**: A is rank 0 (highest score) among A + ancestors

**Rationale**: A concept should always fire more strongly than its parents when prompted directly. If "Walk" doesn't beat "Mathematics" and "Actions" when we prompt "Walk", the hierarchy is broken.

### 2. Random Criterion

**Test**: When we prompt with concept A's name, does lens A appear in the top 5 among random unrelated concepts?

**Pass condition**: A is in top 5 of A + 4 random concepts

**Rationale**: A lens should fire consistently on its own concept name regardless of what other concepts are loaded. This catches lenses that are generally weak.

### 3. Sibling Criterion

**Test**: When we prompt with concept A's name, does lens A score higher than all siblings (concepts sharing the same parent)?

**Pass condition**: A is rank 0 (highest score) among A + siblings

**Rationale**: Siblings compete for the same semantic space. If "Dog" can't beat "Cat" when prompted with "Dog", the lenses aren't discriminative enough.

### Passing All Three

A concept is considered **well-calibrated** when it passes all three criteria (at >=80% success rate across multiple tests). This approach replaces the separate sibling refinement process by integrating sibling competition into the main calibration loop.

## Contrastive Fine-Tuning

When a lens fails calibration, we use contrastive training to fix it:

### For Ancestor/Sibling Failures

```python
# Boost target lens
target_score → 1.0

# Suppress competitors that beat us
for competitor in competitors_that_beat_target:
    if competitor_score > target_score - 0.2:
        competitor_score → target_score - 0.3
```

This is more effective than just boosting the target because many lenses saturate at 1.0. We need to push competitors DOWN, not just push the target UP.

### For Random Criterion Failures

Simple boost training: increase target activation on its own concept name.

## Hierarchical Calibration Strategy

The key insight is that calibration should follow the hierarchy:

1. **Fix parents first**: If a parent concept doesn't fire, its children can never expand
2. **Layer-by-layer**: Calibrate layer 0, then layer 1, etc.
3. **Siblings compete locally**: Only need to beat siblings, not all 8k concepts

This reduces the search space from N×N to roughly N×depth for ancestors and N×siblings for sibling competition.

## The Calibration Matrix (Reference)

### Idealized Full Calibration

The complete calibration produces an N×N matrix where:
- **Rows**: Prompted concepts (what we ask the model about)
- **Columns**: Lens concepts (which lenses fire)
- **Cells**: Rank of the column concept when the row concept is prompted

```
                    Lens Concepts
                    A    B    C    D    ...
Prompted    A      [1]   45   230  12   ...
Concepts    B       23  [1]   89   456  ...
            C       12   34  [1]   78   ...
            D       5    67   123 [1]   ...
```

The diagonal (bracketed) represents each concept's rank when its own name is prompted - ideally rank 1, but natural variance means this won't always be the case.

### Expected Patterns

1. **Diagonal dominance**: Each concept should rank highly when prompted directly
2. **Hierarchical correlation**: Parent concepts rank higher when children are prompted
3. **Semantic clustering**: Related concepts (siblings, synonyms) show correlated activations
4. **Layer-weighted distribution**: Higher-layer (more abstract) concepts appear more frequently across all prompts

### Natural Variance Sources

- **Polysemy**: "Bank" activates both financial and geographic concepts
- **Abstraction level**: "Entity" fires broadly; "LeftMetatarsal2Bone" fires narrowly
- **Training data**: Some concepts have richer training hints than others
- **Prompt sensitivity**: Same concept shows different ranks with different phrasings

## Running Calibration

### Analysis Phase

```bash
python -m src.map.training.calibration.batched_analysis \
    --lens-pack lens_packs/your-pack \
    --concept-pack concept_packs/your-pack \
    --model your-model \
    --dual-criteria \
    --output lens_packs/your-pack/calibration_analysis.json
```

Output shows:
- Ancestor criterion pass rate
- Random criterion pass rate
- Sibling criterion pass rate
- All three criteria pass rate
- List of failing concepts with details

### Fine-Tuning Phase

```bash
python -m src.map.training.calibration.finetune \
    --lens-pack lens_packs/your-pack \
    --concept-pack concept_packs/your-pack \
    --model your-model \
    --analysis lens_packs/your-pack/calibration_analysis.json \
    --dual-criteria \
    --output lens_packs/your-pack/finetune_report.json
```

This will:
1. Process ancestor competition failures (contrastive training)
2. Process sibling competition failures (contrastive training)
3. Process random criterion failures (boost training)

### Calibration Cycle

For full iterative calibration including cross-activation measurement and validation:

```bash
python -m src.map.training.calibration.cycle \
    --lens-pack lens_packs/your-pack \
    --concept-pack concept_packs/your-pack \
    --model your-model \
    --max-cycles 10 \
    --convergence-threshold 0.05
```

The cycle runs these phases:

1. **Analysis → Finetune cycles**: Iterate until convergence or max cycles
2. **Cross-activation calibration**: Measure per-concept noise floors for normalized scoring
3. **Validation**: Compute quality metrics (diagonal rank, Jaccard stability)

Output files:
- `calibration_analysis_cycle{N}.json` - Analysis results per cycle
- `calibration_finetune_cycle{N}.json` - Fine-tune report per cycle
- `calibration.json` - Cross-activation data with validation metrics merged
- `validation.json` - Standalone validation results
- `calibration_summary.json` - Overall cycle summary

CLI options:
- `--no-cross-activation`: Skip cross-activation calibration
- `--no-validation`: Skip validation step
- `--fast-mode`: Use prompt-only analysis (faster but less accurate)
- `--production`: Test against full DynamicLensManager population
- `--cross-activation-samples N`: Samples per concept (default: 5)

## Cross-Activation Calibration

After the analysis/finetune cycles complete, cross-activation calibration measures how each lens responds to prompts for OTHER concepts. This produces per-concept statistics:

- **self_mean/self_std**: Mean/std activation on the concept's own prompts
- **cross_mean/cross_std**: Mean/std activation on other concepts' prompts
- **cross_fire_rate**: Fraction of cross-concept probes where lens fires > threshold
- **gen_fire_rate**: Fraction of generation probes where lens fires

These enable **normalized scoring** at inference: raw activations are transformed so 1.0 = true signal, 0.5 = noise floor, 0.0 = below floor. Concepts without calibration data get a conservative default (confidence=0) that pulls scores toward 0.5.

## Validation Metrics

The validation step computes quality metrics stored in `calibration.json`:

- **diagonal_in_top_k_rate**: When probing concept X, how often does lens X appear in top-k?
- **avg_diagonal_rank**: Average rank of the target lens when probing its concept
- **topk_jaccard_mean/std**: Jaccard similarity between top-k sets across probes (stability)
- **stable_lens_count/unstable_lens_count**: Lenses classified by coefficient of variation

## Convergence Criteria

Calibration is complete when:
- All three criteria pass rates > 90%
- Improvement between cycles < convergence threshold (default 5%)
- No concepts failing all three criteria

## Output Format

### Analysis Result

```json
{
  "mode": "triple_criteria",
  "total_concepts": 7946,
  "lens_reports": {
    "ConceptName": {
      "ancestor_rank_0_rate": 0.8,
      "random_top5_rate": 1.0,
      "sibling_rank_0_rate": 0.6,
      "passes_ancestor_criterion": true,
      "passes_random_criterion": true,
      "passes_sibling_criterion": false,
      "failed_ancestors": ["Parent1", "Parent2"],
      "failed_siblings": ["Sibling1", "Sibling2"]
    }
  },
  "under_firing": ["concept1", "concept2"],
  "over_firing": ["concept3"],
  "well_calibrated": ["concept4", "concept5"]
}
```

### Fine-Tune Report

```json
{
  "total_lenses_processed": 150,
  "lenses_boosted": 140,
  "improvement_rate": 0.93,
  "results": [
    {
      "concept": "ConceptName",
      "action": "ancestor_competition",
      "before_in_top_k_rate": 0.0,
      "after_in_top_k_rate": 1.0,
      "improved": true
    }
  ]
}
```

## Implementation Files

- `src/map/training/calibration/batched_analysis.py` - Analysis with triple criteria
- `src/map/training/calibration/finetune.py` - Contrastive fine-tuning
- `src/map/training/calibration/cycle.py` - Iterative calibration cycles
- `src/map/training/calibration/cross_activation.py` - Cross-activation calibration
- `src/map/training/calibration/validation.py` - Quality metrics validation
- `src/map/training/calibration/activation_cache.py` - Activation caching for efficiency

## Future Work

- **Adaptive thresholds**: Learn pass rates from data rather than fixed 80%
- **Cross-pack calibration**: Ensure consistency when multiple packs are loaded
- **Online calibration**: Adjust based on production activation patterns
- **Hierarchical batching**: Process by layer for more efficient calibration
