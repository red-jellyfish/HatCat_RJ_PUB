# Hierarchy Blocker Remediation Plan

## Problem Statement

Analysis of the first-light lens pack revealed that **86% of concepts are blocked** from dynamic loading due to over-firing parent concepts in the hierarchy. However, using coverage-adjusted metrics shows the problem is concentrated in a small number of high-impact concepts.

### Root Cause

Parent concepts in L1-L3 have Cross-Fire Rates (CFR) far exceeding their expected baseline (tree coverage). This causes them to be classified as "over-firers" by the normalization system, which assigns them confidence=0 and effectively disables them.

**Key insight**: A concept covering 18% of the hierarchy (like Artifact with 1,446 descendants) SHOULD fire ~18% of the time. Firing 71% of the time means 53% "excess" - still problematic but not as broken as raw CFR suggests.

### Impact Analysis (Coverage-Adjusted)

```
Retraining  1 concept  → 25% of blocked concepts unblocked
Retraining  3 concepts → 50% unblocked
Retraining  5 concepts → 75% unblocked
Retraining  8 concepts → 95% unblocked
```

The top 10 impactful blockers account for ~5,244 descendants (66% of all blocked concepts).

## Remediation Strategy

### Phase 1: Normalization Consistency (Immediate)

**Problem**: Normalization is currently applied only at display output, not for internal decisions.

**Solution**: Move normalization earlier in the pipeline so ALL consumers see normalized scores:
- Dynamic child loading decisions
- HUSH threshold evaluation
- Cache scoring
- Any other internal consumers

**Files to modify**:
- `src/hat/monitoring/lens_manager.py` - Apply normalization in `detect_and_expand()` before storing scores
- `src/hush/hush_controller.py` - Verify uses normalized scores (currently uses raw cache scores)

### Phase 2: MELD Creation (Short-term)

Create high-quality MELDs for the 30 highest-impact blockers. These concepts lack proper descriptions and positive/negative examples.

**Priority MELD List** (excess CFR > 20%, sorted by descendants blocked):

| # | Concept | Layer | CFR | Excess | Descendants |
|---|---------|-------|-----|--------|-------------|
| 1 | Artifact | L1 | 70.8% | 52.6% | 1,446 |
| 2 | Device | L2 | 49.7% | 40.6% | 729 |
| 3 | PhysicalMedia | L1 | 33.8% | 24.8% | 713 |
| 4 | Proposition | L1 | 57.6% | 49.8% | 618 |
| 5 | Motion | L1 | 55.0% | 48.4% | 528 |
| 6 | CognitiveAgent | L3 | 100.0% | 95.3% | 372 |
| 7 | Quantity | L1 | 45.0% | 40.8% | 336 |
| 8 | Food | L2 | 48.0% | 43.8% | 336 |
| 9 | Product | L2 | 64.0% | 59.7% | 335 |
| 10 | Group | L2 | 31.4% | 27.4% | 320 |
| 11 | Organization | L1 | 90.0% | 86.4% | 289 |
| 12 | BiologicallyActiveSubstance | L3 | 100.0% | 96.8% | 255 |
| 13 | Region | L1 | 71.3% | 68.1% | 248 |
| 14 | Mixture | L3 | 100.0% | 97.3% | 215 |
| 15 | Communication | L2 | 72.1% | 69.4% | 210 |
| 16 | Text | L2 | 76.7% | 74.2% | 202 |
| 17 | IntentionalPsychologicalProcess | L3 | 100.0% | 97.5% | 199 |
| 18 | FoodIngredient | L3 | 100.0% | 97.5% | 195 |
| 19 | Work | L1 | 89.9% | 87.6% | 185 |
| 20 | Electronics | L3 | 99.2% | 97.1% | 165 |
| 21 | OrganismProcess | L3 | 100.0% | 98.0% | 158 |
| 22 | Transfer | L3 | 100.0% | 98.1% | 153 |
| 23 | BodySubstance | L3 | 100.0% | 98.1% | 148 |
| 24 | PreparedFood | L3 | 100.0% | 98.2% | 142 |
| 25 | GroupOfPeople | L3 | 100.0% | 98.3% | 138 |
| 26 | Animal | L2 | 47.0% | 45.4% | 134 |
| 27 | Vehicle | L3 | 98.2% | 96.6% | 128 |
| 28 | LinguisticCommunication | L3 | 100.0% | 98.5% | 123 |
| 29 | Appliances | L3 | 100.0% | 98.5% | 122 |
| 30 | BeliefGroup | L4 | 100.0% | 98.6% | 115 |

### Phase 3: Retraining (Medium-term)

After MELDs are created:
1. Apply MELDs to concept definitions
2. Retrain affected lenses with improved definitions
3. Re-run calibration cycle
4. Per standing policy: retrain parent concepts when children change

**Blast radius**: Limited - all 30 concepts are in L1-L4, with most in L1-L3. Children inherit improved parent behavior.

### Phase 4: Pack Upload (Medium-term)

Upload retrained lens pack with:
- Updated lens weights
- New calibration.json with improved CFR metrics
- Documentation of changes

## Short-Term Workaround (Hackathon)

For immediate testing, accept that L0-L2 concepts are always active:

1. **Standard config**: L0-L1 are always loaded anyway (`always_load_layers: [0, 1]`)
2. **Behavior**: Over-firer parents fire constantly but children still load and fire correctly
3. **Display filtering**: Don't show L0-L2 in top-k display (they're not interesting)
4. **Trade-offs**:
   - Higher VRAM (more lenses loaded)
   - Slightly higher latency
   - But: Correct child detection works

**Implementation**: No code changes needed - just accept current behavior and filter display output.

## Normalization Architecture (Reference)

### Current Flow (Problematic)
```
lens(hidden_state) → raw_prob
    ↓
cache.lens_scores[key] = raw_prob  ← RAW stored
    ↓
detect_and_expand() uses raw for child loading
    ↓
HUSH uses cache.lens_scores (RAW) for thresholds
    ↓
normalize() applied ONLY for display output
```

### Target Flow (Consistent)
```
lens(hidden_state) → raw_prob
    ↓
normalized = calibration.normalize(raw_prob)
    ↓
cache.lens_scores[key] = normalized  ← NORMALIZED stored
    ↓
All consumers see normalized scores
```

## Analysis Tools

- `scripts/tools/analyze_hierarchy_blockers.py` - Coverage-adjusted blocker analysis
- `scripts/tools/calibrate_cross_activation.py` - Calibration data generation
- `src/hush/calibrate_significance.py` - Significance threshold analysis

## Metrics to Track

After remediation:
- Valid parent chains: Target >80% (currently 14%)
- Impactful over-firers: Target <50 (currently 749)
- Top-10 excess CFR: Target <30% average (currently ~70%)

## Timeline

| Phase | Effort | Impact |
|-------|--------|--------|
| Phase 1 (Normalization) | 2-4 hours | Consistent behavior |
| Phase 2 (MELDs) | 4-8 hours | Definition quality |
| Phase 3 (Retraining) | 2-4 hours compute | Fixed calibration |
| Phase 4 (Upload) | 30 min | Deployment |

## References

- `docs/approach/CALIBRATION.md` - Full calibration pipeline
- `docs/implementation/HUSH_SIGNIFICANCE_SCORING.md` - Significance scoring
- `src/hat/monitoring/deployment_manifest.py` - ConceptCalibration.normalize()
