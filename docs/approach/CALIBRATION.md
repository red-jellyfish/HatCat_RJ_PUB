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

For iterative calibration until convergence:

```bash
python -m src.map.training.calibration.cycle \
    --lens-pack lens_packs/your-pack \
    --concept-pack concept_packs/your-pack \
    --model your-model \
    --mode full \
    --max-cycles 10 \
    --threshold 1.0
```

This runs analyze → finetune → analyze cycles until the improvement rate drops below threshold.

## Convergence Criteria

Calibration is complete when:
- All three criteria pass rates > 90%
- Improvement between cycles < 1%
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

- `src/training/calibration/batched_analysis.py` - Analysis with triple criteria
- `src/training/calibration/finetune.py` - Contrastive fine-tuning
- `src/training/calibration/cycle.py` - Iterative calibration cycles

## Full Calibration Pipeline Architecture

The complete calibration pipeline connects training-time calibration to runtime normalization and significance scoring. Understanding this end-to-end flow is critical for proper HUSH configuration.

### Pipeline Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        CALIBRATION PIPELINE                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐          │
│  │  TRAINING-TIME  │    │   GENERATION    │    │    RUNTIME      │          │
│  │   CALIBRATION   │ →  │   CALIBRATION   │ →  │  NORMALIZATION  │          │
│  └─────────────────┘    └─────────────────┘    └─────────────────┘          │
│         ↓                      ↓                       ↓                     │
│  Triple-Criteria        cross_fire_rate         ConceptCalibration          │
│  (ancestor/sibling/     gen_fire_rate           .normalize()                │
│   random)               self_mean/cross_mean    → [0,1] confidence          │
│                                                                              │
│                                ↓                                             │
│                    ┌─────────────────────┐                                   │
│                    │ SIGNIFICANCE SCORING │                                  │
│                    └─────────────────────┘                                   │
│                             ↓                                                │
│                    Distinguish decision                                      │
│                    tokens from filler                                        │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Phase 1: Training-Time Calibration

**Purpose**: Ensure lenses fire correctly on their target concepts.

**Process**: `src/map/training/calibration/cycle.py`
1. **Analysis**: Score all concepts, check ancestor/sibling/random criteria
2. **Fine-tune**: Contrastive training to fix failures
3. **Repeat**: Until >90% pass rate on all criteria

**Output**: Corrected lens weights (.pt files)

### Phase 2: Generation Calibration

**Purpose**: Measure how lenses behave during normal text generation (not concept prompts).

**Process**: `scripts/tools/calibrate_cross_activation.py --mode generation`

For each lens, measures:
- **self_mean**: Average activation when own concept is prompted
- **cross_mean**: Average activation on other concepts' prompts (where it fired)
- **cross_fire_rate** (CFR): Fraction of OTHER concept prompts it fired on
- **gen_fire_rate** (GFR): Fraction of GENERATION tokens it fired on

**Interpretation**:
| Metric | Good | Problematic |
|--------|------|-------------|
| CFR | <20% | >50% = "over-firer" |
| GFR | <20% | >30% = noisy on generation |
| Gap (self_mean - cross_mean) | >0.2 | <0.1 = poor discrimination |

**Output**: `calibration.json` in lens pack directory

### Phase 3: Runtime Normalization

**Purpose**: Transform raw activation probabilities to confidence-weighted scores.

**Implementation**: `ConceptCalibration.normalize()` in `deployment_manifest.py`

Three-stage transformation:

**Stage 1: Range Transformation**
```
Maps: [0, cross_mean, self_mean, 1.0] → [0, 0.5, 1.0, beyond]

If raw_prob >= cross_mean:
    range_transformed = 0.5 + 0.5 * (raw_prob - cross_mean) / gap
Else:
    range_transformed = 0.5 * raw_prob / cross_mean
```

**Stage 2: Confidence Weighting**
```
confidence = (1 - CFR) × (1 - GFR) × gap_confidence

Where gap_confidence = min(1.0, gap / 0.2)

Result = 0.5 + (range_transformed - 0.5) × confidence
```

Critical insight: **Over-firers (CFR=100%) get confidence=0**, meaning their normalized score is stuck at ~0.5 regardless of raw activation. This automatically disables unreliable lenses.

**Stage 3: Sigmoid Compression**
```
output = sigmoid(steepness × (x - 0.5))

With steepness=6: maps approximately [0,1] → [0.05, 0.95]
```

### Phase 4: Significance Scoring

**Purpose**: Distinguish "decision" tokens from "filler" tokens.

**Implementation**: `src/hush/prod_sig.py`

Uses three signals:
1. **Activation Delta**: How much hidden state changed between layers
2. **Entropy over Top-K**: Concentrated (low entropy) = decision point
3. **Max Above Noise Floor**: Clear signal above calibrated baseline

**Calibrated Defaults** (from gemma-3-4b_first-light-v1):
- default_noise_floor: 0.60 (median gen_mean across concepts)
- entropy_thresh: 2.0 (~log(8) for top-8)
- max_above_thresh: 0.05 (5% above noise floor)

### How This Affects HUSH Thresholds

HUSH constraints use thresholds in **normalized space**:

```yaml
hush:
  honest_steering:
    constraints:
      - simplex_term: "Deception"
        max_deviation: 0.7  # In normalized space!
```

**Key relationships**:
- Threshold 0.5 = fires at noise floor (always triggers)
- Threshold 0.7 = requires ~40% above noise floor in raw space
- Threshold 0.9 = requires strong, high-confidence signal

For a lens with:
- self_mean=0.87, cross_mean=0.58, CFR=2%, GFR=0%
- gap=0.29, gap_confidence=1.0, confidence=0.98

A raw activation of 0.67 would normalize to:
```
range_transformed = 0.5 + 0.5 × (0.67 - 0.58) / 0.29 = 0.655
confidence_weighted = 0.5 + (0.655 - 0.5) × 0.98 = 0.652
after_sigmoid ≈ 0.71
```

So threshold 0.7 would trigger at raw~0.67.

### Hierarchy and Dynamic Loading

The concept hierarchy affects lens loading:

**Always-loaded layers**: L0, L1, L2 (base coverage)
- All base layer concepts have lenses
- These lenses are kept in memory permanently

**Dynamic loading** (L3+):
- When parent confidence exceeds threshold → load children
- Requires parent to have a lens AND fire above threshold
- Full parent chain must fire: L0→L1→L2→...→target

**Example loading path for HonestDefaults (L5)**:
```
MindsAndAgents(L0) → AgentAction(L1) → CreativeActivities(L2) →
PersuasiveCommunication(L3) → EthicalDesign(L4) → HonestDefaults(L5)
```

Each ancestor must fire above its threshold to load the next layer's children.

### Pack Usability Analysis

From first-light-v1 calibration (7,696 concepts):

| Category | Count | Percentage |
|----------|-------|------------|
| Reliable (CFR<20%, GFR<20%) | ~27.5% | Usable for detection |
| Over-firers (CFR>50%) | ~51% | Auto-disabled by normalization |
| High GFR (GFR>20%) | ~15% | Noisy on generation |
| Mixed issues | ~6.5% | Case-by-case evaluation |

### Cross-References

- **Statistical framework**: `docs/implementation/STATISTICAL_ESTIMATION_FRAMEWORK.md`
- **Significance scoring**: `docs/implementation/HUSH_SIGNIFICANCE_SCORING.md`
- **HUSH safety harness**: `docs/specification/HUSH/HUSH_SAFETY_HARNESS.md`
- **Dynamic loading**: `docs/approach/dynamic_lens_loading.md`

## Future Work

- **Adaptive thresholds**: Learn pass rates from data rather than fixed 80%
- **Cross-pack calibration**: Ensure consistency when multiple packs are loaded
- **Online calibration**: Adjust based on production activation patterns
- **Hierarchical batching**: Process by layer for more efficient calibration
- **Per-concept noise floors**: Use calibrated gen_mean per concept in significance scoring
