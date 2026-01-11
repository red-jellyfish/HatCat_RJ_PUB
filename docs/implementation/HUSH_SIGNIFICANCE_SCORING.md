# HUSH Significance Scoring

## Overview

Production significance scoring distinguishes "decision" tokens from "filler" tokens during model generation. This addresses a key weakness in interpretability methods identified in "The Dead Salmons of AI Interpretability" paper - that concept activations on filler tokens (AND, THE, THERE) are often meaningless noise.

**Key insight**: Model decisions happen in "spurts and runs" - bursts of activity around punctuation and complex concepts, not uniformly across all tokens.

## The Problem

During generation, every token gets concept activations from the lens system. However:
- Filler tokens (articles, conjunctions, common words) show noisy, diffuse activations
- These aren't meaningful signals - the model isn't "deciding" anything significant
- High fire rates on safety concepts (e.g., `SelfDeceptionSignal` at 58%) may include filler noise
- This creates false positives in safety monitoring

## The Solution: Production Significance Scoring

Located in `src/hush/prod_sig.py`

### Three Independent Signals

1. **Activation Delta** (`delta`)
   - Measures: `||h_late - h_ref||` - how much hidden state changed between layers
   - High delta = active computation happening
   - Low delta = token just passing through

2. **Entropy over Top-K Concepts** (`entropy`)
   - Measures: Shannon entropy of softmax(top-k concept scores)
   - Low entropy = concentrated on specific concepts = decision point
   - High entropy = diffuse/uncertain activations = filler

3. **Max Above Noise Floor** (`max_above`)
   - Measures: How much the top concept exceeds the calibrated noise floor
   - High = clear signal above baseline
   - Low = within normal noise range

### Combining Signals

```python
logits = w_delta * z_delta - w_entropy * z_H + w_max_above * z_max
significance = sigmoid(logits)  # 0-1 score
```

The z-values use robust normalization (median/MAD) for outlier resistance.

### Layer-Aware Computation

Uses HatCat's existing early/mid/late layer structure:
- **Early** (layer ~L/3): Definitional representations
- **Mid** (layer ~L/2): Behavioral processing
- **Late** (layer ~2L/3): Output preparation

Decision tokens show **entropy drops** across the cascade:
- Entropy decreases early→mid as concept crystallizes
- Further decrease mid→late as output commits

Filler tokens show **flat entropy** across all layers.

## Calibrated Defaults

From `gemma-3-4b_first-light-v1` lens pack calibration:

| Parameter | Value | Source |
|-----------|-------|--------|
| `default_noise_floor` | 0.60 | Median gen_mean across 7696 concepts |
| `entropy_thresh` | 2.0 | ~log(8) for top-8 concepts |
| `max_above_thresh` | 0.05 | 5% above noise floor |
| `delta_thresh` | 0.1 | Conservative default |

### Calibration Analysis

Run `src/hush/calibrate_significance.py` to analyze a lens pack:

```bash
python src/hush/calibrate_significance.py /path/to/lens_pack
```

Outputs:
- Noise floor statistics (median, percentiles)
- Per-layer distribution
- Safety concept fire rates

## Integration with WorldTick

`WorldTick` now includes significance fields:

```python
@dataclass
class WorldTick:
    # ... existing fields ...

    # Significance scoring
    significance: float = 0.0      # 0-1, high = decision point
    entropy_by_layer: Dict[str, float] = field(default_factory=dict)
    activation_delta: float = 0.0
    is_filler: bool = False        # Hard classification
```

## Usage in Safety Monitoring

### Weighting Safety Intensity

```python
# Reduce alert intensity on filler tokens
if significance > 0:
    safety_intensity = safety_intensity * (0.5 + 0.5 * significance)
```

### Filtering Alerts

Only trigger alerts on high-significance tokens:
```python
if significance > 0.7 and safety_score > threshold:
    trigger_alert()
```

### Audit Log Enrichment

Include significance in ASK audit entries to distinguish:
- High-confidence detections (high significance + high safety score)
- Potential false positives (low significance + high safety score)

## API

### ProdSigConfig

```python
@dataclass
class ProdSigConfig:
    temp: float = 1.0              # Softmax temperature
    eps: float = 1e-8              # Numerical stability

    # Feature weights
    w_delta: float = 1.0
    w_entropy: float = 1.0
    w_max_above: float = 1.0
    w_entropy_drop: float = 0.5    # Layer cascade weight

    # Thresholds for hard filler mask
    delta_thresh: float = 0.1
    entropy_thresh: float = 2.0
    max_above_thresh: float = 0.05
    default_noise_floor: float = 0.60

    # Layer settings
    use_layer_cascade: bool = True
    use_seq_robust_norm: bool = True
```

### Core Functions

```python
def compute_sig_fast(
    h_last: torch.Tensor,           # (T, D) late layer hidden states
    h_ref: torch.Tensor,            # (T, D) reference layer hidden states
    topk_concept_ids: torch.Tensor, # (T, k) concept indices
    topk_scores: torch.Tensor,      # (T, k) concept scores
    noise_floor: Optional[torch.Tensor] = None,  # (C,) per-concept
    cfg: Optional[ProdSigConfig] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Returns:
        sig: (T,) significance scores in [0,1]
        weighted_topk: (T,k) concept scores weighted by significance
        filler_mask: (T,) boolean mask
    """
```

```python
def compute_sig_with_layer_cascade(
    h_early: torch.Tensor,
    h_mid: torch.Tensor,
    h_late: torch.Tensor,
    topk_concept_ids: torch.Tensor,
    topk_scores: torch.Tensor,
    concept_vectors: Optional[torch.Tensor] = None,
    noise_floor: Optional[torch.Tensor] = None,
    cfg: Optional[ProdSigConfig] = None,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
    """
    Layer-aware significance with entropy cascade tracking.
    """
```

### SignificanceScorer Class

Stateful wrapper for per-token scoring during generation:

```python
scorer = SignificanceScorer(lens_manager, cfg)
scorer.reset()  # New generation

for token in generation:
    sig, entropy_by_layer, is_filler = scorer.score_tick(
        hidden_states={'early': h_e, 'mid': h_m, 'late': h_l},
        topk_concept_ids=ids,
        topk_scores=scores,
    )
```

## Performance

- **O(T × k)** complexity per sequence (T tokens, k top concepts)
- No full vocabulary/concept dictionary scans
- Robust normalization is O(T) with median operations
- Layer cascade adds minimal overhead when projections already computed

## Future Work

1. **Full calibration run**: Compute significance across all training samples, correlate with positive/negative labels, fit optimal thresholds via ROC analysis

2. **Per-concept noise floors**: Use calibrated gen_mean per concept instead of global median

3. **Temporal patterns**: Track significance patterns across sequences to identify manipulation "build-up"

4. **Integration with steering**: Weight steering strength by inverse significance (stronger correction on decision tokens)

## Relationship to Calibration Pipeline

Significance scoring is Phase 4 of the full calibration pipeline:

1. **Training-time calibration** → ensures lenses fire on correct concepts
2. **Generation calibration** → measures CFR/GFR noise characteristics
3. **Runtime normalization** → applies confidence weighting to raw scores
4. **Significance scoring** → distinguishes decision vs filler tokens

The `default_noise_floor` (0.60) comes from generation calibration's median `gen_mean`. This connects significance scoring to the same calibration data that drives runtime normalization.

For full pipeline documentation, see: `docs/approach/CALIBRATION.md`

## References

- "The Dead Salmons of AI Interpretability" - Motivating paper on interpretability uncertainty
- `src/hush/prod_sig.py` - Implementation
- `src/hush/calibrate_significance.py` - Calibration analysis tool
- `src/hush/hush_integration.py` - WorldTick integration
- `docs/approach/CALIBRATION.md` - Full calibration pipeline documentation
- `docs/implementation/STATISTICAL_ESTIMATION_FRAMEWORK.md` - Confidence intervals and variance
