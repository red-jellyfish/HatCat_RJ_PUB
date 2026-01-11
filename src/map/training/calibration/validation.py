#!/usr/bin/env python3
"""
Calibration Validation - Final step of calibration cycle.

Validates lens pack quality by probing concepts and measuring:
1. Diagonal rank: When probing concept X, where does lens X rank?
2. Diagonal in top-k rate: % of probes where target appears in top-k
3. Top-k Jaccard stability: How similar are top-k sets across probes?
4. Per-lens statistics with confidence intervals

This is the final step after cross-activation calibration, providing
quality metrics that should be stored with the lens pack.

Usage:
    # As part of calibration cycle (called automatically)
    # Or standalone:
    python -m src.map.training.calibration.validation \
        --lens-pack lens_packs/gemma-3-4b_first-light-v1-bf16 \
        --concept-pack concept_packs/first-light \
        --model google/gemma-3-4b-pt
"""

import json
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Any

import numpy as np
from tqdm import tqdm


@dataclass
class ValidationResults:
    """Results from calibration validation."""
    timestamp: str
    lens_pack: str
    concepts_probed: int
    total_lenses: int

    # Diagonal metrics (does concept X rank highly when probing X?)
    avg_diagonal_rank: float = 0.0
    median_diagonal_rank: float = 0.0
    diagonal_in_top_k_rate: float = 0.0

    # Stability metrics (how consistent are top-k across probes?)
    topk_jaccard_mean: float = 0.0
    topk_jaccard_std: float = 0.0

    # Lens quality classification
    stable_lens_count: int = 0
    unstable_lens_count: int = 0

    # Per-lens detailed statistics
    lens_stats: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    # Top over-firers (concepts that appear in top-k when they shouldn't)
    over_firing: List[str] = field(default_factory=list)


def run_validation(
    lens_pack_dir: Path,
    concept_pack_dir: Path,
    model,
    tokenizer,
    device: str,
    layers: Optional[List[int]] = None,
    top_k: int = 10,
    max_concepts: Optional[int] = None,
    layer_idx: int = -1,  # Use last layer to match HushedGenerator inference
    compute_per_lens_stats: bool = True,
    max_lens_stats: int = 500,
) -> ValidationResults:
    """
    Run calibration validation on a lens pack.

    Args:
        lens_pack_dir: Path to lens pack
        concept_pack_dir: Path to concept pack with hierarchy
        model: Loaded model for inference
        tokenizer: Tokenizer for prompts
        device: Device for inference
        layers: Layers to include (None = auto-detect)
        top_k: Top-k for ranking metrics
        max_concepts: Limit concepts to probe (None = all)
        layer_idx: Model layer for hidden state extraction
        compute_per_lens_stats: Whether to compute detailed per-lens stats
        max_lens_stats: Max lenses to compute detailed stats for

    Returns:
        ValidationResults with all metrics
    """
    import torch
    from src.hat.monitoring.lens_batched import BatchedLensBank
    from src.hat.monitoring.lens_types import create_lens_from_state_dict

    print(f"\n{'='*60}")
    print("CALIBRATION VALIDATION")
    print(f"{'='*60}")
    print(f"  Lens pack: {lens_pack_dir}")
    print(f"  Concept pack: {concept_pack_dir}")
    print(f"  Top-k: {top_k}")

    # Determine layers
    if layers is None:
        layers = []
        for layer_dir in lens_pack_dir.glob("layer*"):
            if layer_dir.is_dir():
                try:
                    layer_num = int(layer_dir.name.replace("layer", ""))
                    layers.append(layer_num)
                except ValueError:
                    pass
        layers.sort()
    print(f"  Layers: {layers}")

    # Load all lenses into BatchedLensBank
    print("\nLoading lenses...")
    lens_bank = BatchedLensBank(device=device)
    all_lenses = {}
    lens_layer_map = {}

    # Detect hidden dimension from first lens
    hidden_dim = None
    for layer in layers:
        layer_dir = lens_pack_dir / f"layer{layer}"
        if not layer_dir.exists():
            continue
        for lens_file in layer_dir.glob("*.pt"):
            state_dict = torch.load(lens_file, map_location='cpu', weights_only=True)
            if 'fc1.weight' in state_dict:
                hidden_dim = state_dict['fc1.weight'].shape[1]
            elif 'weight' in state_dict:
                hidden_dim = state_dict['weight'].shape[1]
            break
        if hidden_dim:
            break

    if hidden_dim is None:
        raise ValueError("Could not detect hidden dimension from lenses")
    print(f"  Hidden dimension: {hidden_dim}")

    for layer in layers:
        layer_dir = lens_pack_dir / f"layer{layer}"
        if not layer_dir.exists():
            continue

        for lens_file in layer_dir.glob("*.pt"):
            concept_name = lens_file.stem.replace('_classifier', '')
            try:
                state_dict = torch.load(lens_file, map_location='cpu', weights_only=True)
                lens = create_lens_from_state_dict(state_dict, hidden_dim=hidden_dim, device='cpu')
                all_lenses[concept_name] = lens
                lens_layer_map[concept_name] = layer
            except Exception as e:
                print(f"  Warning: Failed to load {lens_file}: {e}")

    print(f"  Loaded {len(all_lenses)} lenses")
    lens_bank.add_lenses(all_lenses)
    lens_bank.to(device)
    lens_bank.eval()

    # Load concepts from hierarchy
    print("\nLoading concepts...")
    concepts = {}
    for layer in layers:
        layer_file = concept_pack_dir / "hierarchy" / f"layer{layer}.json"
        if not layer_file.exists():
            continue

        with open(layer_file) as f:
            layer_data = json.load(f)

        concept_list = layer_data.get('concepts', layer_data if isinstance(layer_data, list) else [])
        for concept in concept_list:
            term = concept.get('sumo_term') or concept.get('term')
            if term and term in all_lenses:
                # Use term as prompt (fast mode)
                concepts[term] = {
                    'layer': layer,
                    'prompt': term,
                }

    print(f"  Found {len(concepts)} concepts with lenses")

    # Sample concepts if limited
    concept_names = list(concepts.keys())
    if max_concepts and len(concept_names) > max_concepts:
        import random
        concept_names = random.sample(concept_names, max_concepts)

    print(f"\nValidating {len(concept_names)} concepts...")

    # Track statistics
    lens_ranks: Dict[str, List[int]] = defaultdict(list)
    lens_activations: Dict[str, List[float]] = defaultdict(list)
    over_fire_counts: Dict[str, int] = defaultdict(int)
    diagonal_ranks = []
    topk_sets = []

    for concept_name in tqdm(concept_names, desc="Probing concepts"):
        concept_data = concepts[concept_name]
        prompt = concept_data['prompt']

        try:
            # Get activation from model
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = model(**inputs, output_hidden_states=True)
                hidden_states = outputs.hidden_states[layer_idx]
                activation = hidden_states[0, -1, :].float()

            # Score all lenses
            scores = lens_bank(activation)

            # Sort by score
            sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)

            # Track ranks and activations
            for rank, (lens_concept, score) in enumerate(sorted_scores, 1):
                lens_ranks[lens_concept].append(rank)
                lens_activations[lens_concept].append(score)

                # Track over-firing (in top-k but not the target)
                if rank <= top_k and lens_concept != concept_name:
                    over_fire_counts[lens_concept] += 1

                # Track diagonal rank (where does target concept rank?)
                if lens_concept == concept_name:
                    diagonal_ranks.append(rank)

            # Track top-k set for Jaccard stability
            topk_set = frozenset(s[0] for s in sorted_scores[:top_k])
            topk_sets.append(topk_set)

        except Exception as e:
            print(f"  Error probing {concept_name}: {e}")
            continue

    print("\nComputing metrics...")

    # Diagonal metrics
    avg_diagonal_rank = float(np.mean(diagonal_ranks)) if diagonal_ranks else 0.0
    median_diagonal_rank = float(np.median(diagonal_ranks)) if diagonal_ranks else 0.0
    diagonal_in_top_k = sum(1 for r in diagonal_ranks if r <= top_k) / len(diagonal_ranks) if diagonal_ranks else 0.0

    # Jaccard stability
    jaccard_mean = 0.0
    jaccard_std = 0.0
    if len(topk_sets) >= 2:
        jaccards = []
        sample_size = min(100, len(topk_sets))
        import random
        sampled = random.sample(topk_sets, sample_size)

        for i in range(len(sampled)):
            for j in range(i + 1, len(sampled)):
                intersection = len(sampled[i] & sampled[j])
                union = len(sampled[i] | sampled[j])
                if union > 0:
                    jaccards.append(intersection / union)

        if jaccards:
            jaccard_mean = float(np.mean(jaccards))
            jaccard_std = float(np.std(jaccards))

    # Per-lens statistics
    lens_stats = {}
    stable_count = 0
    unstable_count = 0

    if compute_per_lens_stats:
        lens_concepts = list(lens_ranks.keys())[:max_lens_stats]
        for lens_concept in tqdm(lens_concepts, desc="Computing lens stats"):
            ranks = lens_ranks[lens_concept]
            activations = lens_activations[lens_concept]

            if len(ranks) < 2:
                continue

            mean_rank = float(np.mean(ranks))
            std_rank = float(np.std(ranks))
            cv = std_rank / mean_rank if mean_rank > 0 else 0.0
            is_stable = cv < 0.5

            if is_stable:
                stable_count += 1
            else:
                unstable_count += 1

            in_topk_rate = sum(1 for r in ranks if r <= top_k) / len(ranks)

            lens_stats[lens_concept] = {
                'mean_rank': round(mean_rank, 2),
                'std_rank': round(std_rank, 2),
                'cv': round(cv, 3),
                'is_stable': is_stable,
                'detection_rate': round(in_topk_rate, 4),
                'n_samples': len(ranks),
            }

    # Top over-firers
    over_firing = sorted(over_fire_counts.items(), key=lambda x: x[1], reverse=True)[:20]
    over_firing_concepts = [f"{c}_L{lens_layer_map.get(c, '?')}" for c, _ in over_firing]

    results = ValidationResults(
        timestamp=datetime.now(timezone.utc).isoformat(),
        lens_pack=lens_pack_dir.name,
        concepts_probed=len(concept_names),
        total_lenses=len(all_lenses),
        avg_diagonal_rank=avg_diagonal_rank,
        median_diagonal_rank=median_diagonal_rank,
        diagonal_in_top_k_rate=diagonal_in_top_k,
        topk_jaccard_mean=jaccard_mean,
        topk_jaccard_std=jaccard_std,
        stable_lens_count=stable_count,
        unstable_lens_count=unstable_count,
        lens_stats=lens_stats,
        over_firing=over_firing_concepts,
    )

    # Print summary
    print(f"\n{'='*60}")
    print("VALIDATION SUMMARY")
    print(f"{'='*60}")
    print(f"  Concepts probed: {results.concepts_probed}")
    print(f"  Total lenses: {results.total_lenses}")
    print()
    print(f"  Diagonal Metrics:")
    print(f"    Avg diagonal rank: {results.avg_diagonal_rank:.1f}")
    print(f"    Median diagonal rank: {results.median_diagonal_rank:.1f}")
    print(f"    Diagonal in top-{top_k}: {results.diagonal_in_top_k_rate:.1%}")
    print()
    print(f"  Stability Metrics:")
    print(f"    Top-k Jaccard: mean={results.topk_jaccard_mean:.3f}, std={results.topk_jaccard_std:.3f}")
    print(f"    Stable lenses: {results.stable_lens_count}")
    print(f"    Unstable lenses: {results.unstable_lens_count}")
    print()
    if results.over_firing:
        print(f"  Top over-firers: {', '.join(results.over_firing[:5])}")

    return results


def merge_validation_into_calibration(
    calibration_path: Path,
    validation_results: ValidationResults,
) -> None:
    """
    Merge validation results into existing calibration.json.

    Adds a 'validation' section with quality metrics.
    """
    with open(calibration_path) as f:
        calibration = json.load(f)

    calibration['validation'] = {
        'timestamp': validation_results.timestamp,
        'concepts_probed': validation_results.concepts_probed,
        'avg_diagonal_rank': validation_results.avg_diagonal_rank,
        'median_diagonal_rank': validation_results.median_diagonal_rank,
        'diagonal_in_top_k_rate': validation_results.diagonal_in_top_k_rate,
        'topk_jaccard_mean': validation_results.topk_jaccard_mean,
        'topk_jaccard_std': validation_results.topk_jaccard_std,
        'stable_lens_count': validation_results.stable_lens_count,
        'unstable_lens_count': validation_results.unstable_lens_count,
        'top_over_firers': validation_results.over_firing,
    }

    with open(calibration_path, 'w') as f:
        json.dump(calibration, f, indent=2)

    print(f"\n  ✓ Merged validation into: {calibration_path}")


def main():
    import argparse
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    parser = argparse.ArgumentParser(description="Run calibration validation")
    parser.add_argument("--lens-pack", required=True, help="Path to lens pack")
    parser.add_argument("--concept-pack", required=True, help="Path to concept pack")
    parser.add_argument("--model", required=True, help="Model name/path")
    parser.add_argument("--device", default="cuda", help="Device")
    parser.add_argument("--layers", nargs="+", type=int, default=None, help="Layers")
    parser.add_argument("--top-k", type=int, default=10, help="Top-k for metrics")
    parser.add_argument("--max-concepts", type=int, default=None, help="Limit concepts")
    parser.add_argument("--layer-idx", type=int, default=-1, help="Model layer index (-1 = last layer)")
    parser.add_argument("--output", type=str, default=None, help="Output JSON path")
    parser.add_argument("--merge-calibration", action="store_true",
                        help="Merge results into calibration.json")

    args = parser.parse_args()

    lens_pack_dir = Path(args.lens_pack)
    concept_pack_dir = Path(args.concept_pack)

    print(f"Loading model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map=args.device,
    )

    results = run_validation(
        lens_pack_dir=lens_pack_dir,
        concept_pack_dir=concept_pack_dir,
        model=model,
        tokenizer=tokenizer,
        device=args.device,
        layers=args.layers,
        top_k=args.top_k,
        max_concepts=args.max_concepts,
        layer_idx=args.layer_idx,
    )

    # Save results
    output_path = Path(args.output) if args.output else lens_pack_dir / "validation.json"
    with open(output_path, 'w') as f:
        json.dump(asdict(results), f, indent=2)
    print(f"\n  Saved validation to: {output_path}")

    # Optionally merge into calibration.json
    if args.merge_calibration:
        calibration_path = lens_pack_dir / "calibration.json"
        if calibration_path.exists():
            merge_validation_into_calibration(calibration_path, results)
        else:
            print(f"  Warning: No calibration.json found at {calibration_path}")


if __name__ == "__main__":
    main()
