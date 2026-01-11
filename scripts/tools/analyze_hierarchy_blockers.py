#!/usr/bin/env python3
"""
Analyze hierarchy for blocked concepts and identify biggest blocker parents.

For each layer, calculates:
1. How many concepts have a valid parent chain (all ancestors CFR < threshold)
2. Which parent concepts are the biggest blockers (block most descendants)

This helps decide between:
- Bypassing over-firer parents in hierarchy
- Targeted retraining of blocker concepts
"""

import json
import argparse
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Set, Tuple, Optional
from dataclasses import dataclass


@dataclass
class ConceptInfo:
    name: str
    layer: int
    parents: List[str]
    cfr: float = 0.0
    gfr: float = 0.0
    has_lens: bool = False
    is_over_firer: bool = False
    descendant_count: int = 0
    coverage_pct: float = 0.0  # descendants / total concepts
    expected_cfr: float = 0.0  # coverage_pct (expected baseline)
    excess_cfr: float = 0.0    # cfr - expected_cfr (true "over-firing")


def load_hierarchy(concept_pack_path: Path) -> Dict[str, ConceptInfo]:
    """Load hierarchy from layer files."""
    concepts = {}
    hierarchy_dir = concept_pack_path / "hierarchy"

    for layer in range(10):
        layer_file = hierarchy_dir / f"layer{layer}.json"
        if not layer_file.exists():
            continue

        with open(layer_file) as f:
            data = json.load(f)

        for concept in data.get("concepts", []):
            name = concept.get("sumo_term")
            if name:
                concepts[name] = ConceptInfo(
                    name=name,
                    layer=layer,
                    parents=concept.get("parent_concepts", []),
                )

    return concepts


def load_calibration(lens_pack_path: Path) -> Dict[str, dict]:
    """Load calibration data."""
    cal_file = lens_pack_path / "calibration.json"
    if not cal_file.exists():
        print(f"Warning: No calibration file at {cal_file}")
        return {}

    with open(cal_file) as f:
        data = json.load(f)

    # Handle nested structure
    if "calibration" in data:
        return data["calibration"]
    return data


def check_lens_exists(lens_pack_path: Path, concept: str, layer: int) -> bool:
    """Check if a lens file exists for this concept."""
    lens_path = lens_pack_path / f"layer{layer}" / f"{concept}.pt"
    return lens_path.exists()


def analyze_blockers(
    concepts: Dict[str, ConceptInfo],
    calibration: Dict[str, dict],
    lens_pack_path: Path,
    cfr_threshold: float = 0.5,
    excess_cfr_threshold: float = 0.3,  # NEW: threshold for excess CFR
) -> Tuple[Dict[int, dict], Dict[str, int]]:
    """
    Analyze hierarchy for blockers using coverage-adjusted CFR.

    A concept's "expected CFR" is its coverage (descendants/total).
    A concept is an over-firer if excess_cfr = (cfr - expected_cfr) > threshold.

    Returns:
        layer_stats: Per-layer statistics
        blocker_counts: Concept -> number of descendants blocked
    """
    total_concepts = len(concepts)

    # Build parent->children map first (needed for descendant counts)
    children_map: Dict[str, Set[str]] = defaultdict(set)
    for name, info in concepts.items():
        for parent in info.parents:
            children_map[parent].add(name)

    # Recursive function to get all descendants
    def get_all_descendants(concept: str, visited: Set[str] = None) -> Set[str]:
        if visited is None:
            visited = set()
        if concept in visited:
            return set()
        visited.add(concept)

        descendants = set()
        for child in children_map.get(concept, []):
            descendants.add(child)
            descendants.update(get_all_descendants(child, visited))
        return descendants

    # First pass: populate calibration data, lens existence, AND descendant counts
    for name, info in concepts.items():
        cal_key = f"{name}_L{info.layer}"
        if cal_key in calibration:
            cal = calibration[cal_key]
            info.cfr = cal.get("cross_fire_rate", 0)
            info.gfr = cal.get("gen_fire_rate", 0)

        info.has_lens = check_lens_exists(lens_pack_path, name, info.layer)

        # Compute coverage-adjusted metrics
        descendants = get_all_descendants(name)
        info.descendant_count = len(descendants)
        info.coverage_pct = len(descendants) / total_concepts if total_concepts > 0 else 0
        info.expected_cfr = info.coverage_pct  # Expected CFR = coverage
        info.excess_cfr = max(0, info.cfr - info.expected_cfr)  # How much above expected

        # Over-firer based on EXCESS CFR, not raw CFR
        info.is_over_firer = info.excess_cfr >= excess_cfr_threshold

    # Check if a concept has a valid parent chain
    def has_valid_chain(concept: str, visited: Set[str] = None) -> Tuple[bool, Optional[str]]:
        """
        Returns (is_valid, first_blocker).
        A chain is valid if all ancestors have lenses and are not over-firers.
        """
        if visited is None:
            visited = set()
        if concept in visited:
            return True, None  # Cycle - assume valid
        visited.add(concept)

        info = concepts.get(concept)
        if not info:
            return True, None  # Unknown concept - assume valid

        # Check this concept
        if not info.has_lens:
            return False, concept  # No lens = blocker
        if info.is_over_firer:
            return False, concept  # Over-firer = blocker

        # Check parents (need at least one valid path to root)
        if not info.parents:
            return True, None  # Root concept - valid

        # For concepts with multiple parents, ANY valid path is enough
        for parent in info.parents:
            is_valid, blocker = has_valid_chain(parent, visited.copy())
            if is_valid:
                return True, None

        # All parent paths blocked - return the first blocker we find
        _, blocker = has_valid_chain(info.parents[0], visited.copy())
        return False, blocker

    # Analyze each layer
    layer_stats = {}
    blocker_counts: Dict[str, int] = defaultdict(int)
    blocked_by: Dict[str, str] = {}  # concept -> blocker

    for layer in range(10):
        layer_concepts = [c for c in concepts.values() if c.layer == layer]
        if not layer_concepts:
            continue

        total = len(layer_concepts)
        valid = 0
        blocked = 0
        no_lens = 0
        over_firer = 0

        for info in layer_concepts:
            if not info.has_lens:
                no_lens += 1
                blocked += 1
                continue

            if info.is_over_firer:
                over_firer += 1

            is_valid, blocker = has_valid_chain(info.name)
            if is_valid:
                valid += 1
            else:
                blocked += 1
                blocked_by[info.name] = blocker
                if blocker:
                    # Count how many descendants this blocker affects
                    blocker_counts[blocker] += 1

        layer_stats[layer] = {
            "total": total,
            "valid_chain": valid,
            "blocked": blocked,
            "no_lens": no_lens,
            "over_firer_at_layer": over_firer,
            "valid_pct": valid / total * 100 if total > 0 else 0,
        }

    # Add descendant counts to blockers
    for blocker in blocker_counts:
        descendants = get_all_descendants(blocker)
        blocker_counts[blocker] = len(descendants)

    return layer_stats, dict(blocker_counts), blocked_by


def main():
    parser = argparse.ArgumentParser(description="Analyze hierarchy blockers")
    parser.add_argument("--concept-pack", type=Path, default=Path("concept_packs/first-light"),
                        help="Path to concept pack")
    parser.add_argument("--lens-pack", type=Path, default=Path("lens_packs/gemma-3-4b_first-light-v1-bf16"),
                        help="Path to lens pack")
    parser.add_argument("--cfr-threshold", type=float, default=0.5,
                        help="Raw CFR threshold (legacy, less useful)")
    parser.add_argument("--excess-cfr-threshold", type=float, default=0.3,
                        help="Excess CFR threshold (CFR - expected_CFR) for over-firer")
    parser.add_argument("--top-blockers", type=int, default=20,
                        help="Show top N blockers")
    args = parser.parse_args()

    print("=" * 80)
    print("HIERARCHY BLOCKER ANALYSIS (Coverage-Adjusted)")
    print("=" * 80)
    print(f"Concept pack: {args.concept_pack}")
    print(f"Lens pack: {args.lens_pack}")
    print(f"Excess CFR threshold: {args.excess_cfr_threshold:.0%}")
    print("  (excess_cfr = actual_cfr - expected_cfr, where expected = coverage %)")
    print()

    # Load data
    print("Loading hierarchy...")
    concepts = load_hierarchy(args.concept_pack)
    print(f"  Loaded {len(concepts)} concepts")

    print("Loading calibration...")
    calibration = load_calibration(args.lens_pack)
    print(f"  Loaded calibration for {len(calibration)} concept-layers")

    # Analyze
    print("\nAnalyzing parent chains...")
    layer_stats, blocker_counts, blocked_by = analyze_blockers(
        concepts, calibration, args.lens_pack, args.cfr_threshold, args.excess_cfr_threshold
    )

    # Report layer stats
    print("\n" + "=" * 70)
    print("PER-LAYER VALIDITY")
    print("=" * 70)
    print(f"{'Layer':<8} {'Total':<8} {'Valid':<8} {'Blocked':<10} {'No Lens':<10} {'Over-fire':<10} {'Valid %':<10}")
    print("-" * 70)

    total_all = 0
    valid_all = 0
    blocked_all = 0

    for layer in sorted(layer_stats.keys()):
        s = layer_stats[layer]
        total_all += s["total"]
        valid_all += s["valid_chain"]
        blocked_all += s["blocked"]
        print(f"L{layer:<7} {s['total']:<8} {s['valid_chain']:<8} {s['blocked']:<10} {s['no_lens']:<10} {s['over_firer_at_layer']:<10} {s['valid_pct']:.1f}%")

    print("-" * 70)
    print(f"{'TOTAL':<8} {total_all:<8} {valid_all:<8} {blocked_all:<10} {'':<10} {'':<10} {valid_all/total_all*100:.1f}%")

    # Report top blockers - sorted by IMPACT (excess_cfr * descendants)
    print("\n" + "=" * 110)
    print(f"TOP {args.top_blockers} IMPACTFUL BLOCKERS (excess CFR > 0, sorted by descendants blocked)")
    print("=" * 110)

    # Get all blockers with their info
    blocker_concepts = [(c, concepts.get(c)) for c in blocker_counts.keys() if concepts.get(c)]

    # Filter to only those with descendants AND excess CFR > 0
    impactful_blockers = [(c, info) for c, info in blocker_concepts
                         if info.descendant_count > 0 and info.excess_cfr > 0 and info.has_lens]

    # Sort by descendant count (impact), then by excess_cfr
    sorted_blockers = sorted(impactful_blockers, key=lambda x: (x[1].descendant_count, x[1].excess_cfr), reverse=True)

    print(f"{'Concept':<35} {'Layer':<6} {'Coverage':<10} {'Expected':<10} {'Actual':<10} {'EXCESS':<10} {'Descendants':<12}")
    print("-" * 110)

    for concept, info in sorted_blockers[:args.top_blockers]:
        print(f"{concept:<35} L{info.layer:<5} {info.coverage_pct:>8.1%}  {info.expected_cfr:>8.1%}  {info.cfr:>8.1%}  {info.excess_cfr:>8.1%}  {info.descendant_count:<12}")

    # Also show leaf over-firers count
    leaf_overfirers = [(c, info) for c, info in blocker_concepts
                       if info.descendant_count == 0 and info.excess_cfr >= args.excess_cfr_threshold and info.has_lens]
    print(f"\n(Also {len(leaf_overfirers)} leaf concepts with 100% CFR but no descendants - broken but not blocking)")

    # Recommendations
    print("\n" + "=" * 80)
    print("RECOMMENDATIONS")
    print("=" * 80)

    # Re-categorize using ALL blocker concepts
    no_lens_blockers = [(c, info) for c, info in blocker_concepts if not info.has_lens and info.descendant_count > 0]

    # Concepts with descendants that are appropriately firing (excess < threshold)
    appropriate_firers = [(c, info) for c, info in blocker_concepts
                         if info.has_lens and info.descendant_count > 0 and info.excess_cfr < args.excess_cfr_threshold]

    if appropriate_firers:
        # Sort by descendants
        appropriate_firers = sorted(appropriate_firers, key=lambda x: x[1].descendant_count, reverse=True)
        print(f"\nAppropriate firers (CFR justified by coverage): {len(appropriate_firers)} parent concepts")
        print("  These fire frequently but have many descendants, so it's expected.")
        print("\n  Examples (L0-L2 are expected to fire broadly):")
        for concept, info in appropriate_firers[:8]:
            print(f"    {concept} (L{info.layer}): coverage={info.coverage_pct:.1%}, CFR={info.cfr:.1%}, excess={info.excess_cfr:+.1%} - {info.descendant_count} desc")

    if impactful_blockers:
        print(f"\nTRUE Over-firers with impact ({len(impactful_blockers)} concepts):")
        print("  These fire MORE than their tree coverage justifies AND block descendants.")
        print("  Option A: Bypass in hierarchy (mark as 'transparent')")
        print("  Option B: Targeted retraining to reduce excess CFR")
        print("\n  Top candidates for retraining (most descendants blocked):")
        for concept, info in sorted_blockers[:15]:
            print(f"    {concept} (L{info.layer}): coverage={info.coverage_pct:.1%}, CFR={info.cfr:.1%}, EXCESS={info.excess_cfr:+.1%} - {info.descendant_count} descendants")

    if no_lens_blockers:
        no_lens_blockers = sorted(no_lens_blockers, key=lambda x: x[1].descendant_count, reverse=True)
        print(f"\nNo-lens blockers ({len(no_lens_blockers)} concepts):")
        print("  These concepts exist in hierarchy but have no trained lens")
        print("\n  Top candidates for lens training:")
        for concept, info in no_lens_blockers[:10]:
            print(f"    {concept} (L{info.layer}) - {info.descendant_count} descendants")

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total concepts: {total_all}")
    print(f"Valid parent chains: {valid_all} ({valid_all/total_all*100:.1f}%)")
    print(f"Blocked concepts: {blocked_all} ({blocked_all/total_all*100:.1f}%)")
    print(f"Impactful over-firers (excess CFR > 0, have descendants): {len(impactful_blockers)}")
    print(f"Appropriate firers (high CFR but justified by coverage): {len(appropriate_firers)}")
    print(f"Leaf over-firers (100% CFR, no descendants): {len(leaf_overfirers)}")
    print(f"No-lens blockers: {len(no_lens_blockers)}")

    # Impact analysis
    if impactful_blockers:
        top_10_impact = sum(info.descendant_count for _, info in sorted_blockers[:10])
        print(f"\nRetraining top 10 impactful over-firers would unblock: ~{top_10_impact} descendants")


if __name__ == "__main__":
    main()
