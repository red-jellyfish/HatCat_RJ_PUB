"""
Production Significance Scoring.

Distinguishes "decision" tokens from "filler" tokens based on:
1. Activation delta - how much hidden state changed between layers
2. Entropy over top-k concepts - diffuse (filler) vs concentrated (decision)
3. Max above noise floor - signal strength above calibrated baseline

This addresses the "Dead Salmons" problem in interpretability where
concept activations on filler tokens (AND, THE, THERE) are meaningless noise.

Key insight: Model decisions happen in "spurts and runs" - bursts of activity
around punctuation and complex concepts, not on filler tokens.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Tuple, Dict, List
import torch


@dataclass
class ProdSigConfig:
    """Configuration for production significance scoring.

    Defaults calibrated from gemma-3-4b_first-light-v1 lens pack:
    - Noise floor ~0.60 (median gen_mean across 7696 concepts)
    - Most concepts fire <5% of the time during generation
    """

    # Entropy computation
    temp: float = 1.0  # softmax temperature for entropy
    eps: float = 1e-8

    # Feature weights for significance score
    # Entropy weighted higher for discrimination: high entropy = filler, low = decision
    w_delta: float = 1.0
    w_entropy: float = 2.0  # Increased from 1.0 for better filler/decision separation
    w_max_above: float = 1.0
    w_entropy_drop: float = 0.5  # Weight for layer-wise entropy cascade

    # Thresholding (calibrated from first-light lens pack)
    # delta_thresh: tokens with hidden state delta below this are likely filler
    delta_thresh: float = 0.1  # conservative - needs per-model calibration
    # entropy_thresh: high entropy = diffuse activations = filler
    entropy_thresh: float = 1.8  # Lowered for stricter filler detection
    # max_above_thresh: activation must exceed noise floor by this margin
    max_above_thresh: float = 0.10  # 10% above noise floor (raised with lower floor)

    # Default noise floor - post-normalization baseline (calibration pulls to ~0.5)
    default_noise_floor: float = 0.50

    # Layer-aware settings
    use_layer_cascade: bool = True
    layer_checkpoints: List[str] = field(default_factory=lambda: ["early", "mid", "late"])

    # Dynamic normalization per sequence (cheap)
    use_seq_robust_norm: bool = True


def _entropy_from_topk_scores(
    topk_scores: torch.Tensor, temp: float, eps: float
) -> torch.Tensor:
    """
    Compute entropy over top-k concept scores.

    Args:
        topk_scores: (T, k) raw scores for top-k concepts per token
        temp: Temperature for softmax
        eps: Epsilon for numerical stability

    Returns:
        (T,) entropy values
    """
    x = topk_scores / max(temp, eps)
    x = torch.clamp(x, max=50.0)  # Prevent exp overflow
    x = x - x.max(dim=-1, keepdim=True).values
    p = torch.softmax(x, dim=-1)
    return -(p * (p + eps).log()).sum(dim=-1)


def _robust_norm_1d(x: torch.Tensor, eps: float) -> torch.Tensor:
    """
    Robust in-sequence normalization: (x - median) / (MAD-scaled).

    Uses median absolute deviation for outlier resistance.
    Cheap: O(T) with medians.
    """
    med = x.median()
    mad = (x - med).abs().median()
    return (x - med) / (1.4826 * mad + eps)


def compute_entropy_at_layer(
    h_layer: torch.Tensor,
    concept_vectors: torch.Tensor,
    topk_ids: torch.Tensor,
    temp: float = 1.0,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Compute entropy at a specific layer using pre-selected concept IDs.

    This is efficient because we only project the k concepts we care about
    (from final layer selection) rather than all C concepts.

    Args:
        h_layer: (T, D) hidden states at this layer
        concept_vectors: (C, D) concept projection matrix
        topk_ids: (T, k) which concepts to check (from final layer)
        temp: Temperature for softmax
        eps: Epsilon for numerical stability

    Returns:
        (T,) entropy values at this layer
    """
    # Gather only the k concept vectors we care about: (T, k, D)
    vecs = concept_vectors[topk_ids]  # broadcast gather
    # Project: (T, k)
    scores = torch.einsum("td,tkd->tk", h_layer, vecs)
    return _entropy_from_topk_scores(scores, temp, eps)


def compute_sig_fast(
    h_last: torch.Tensor,  # (T, D)
    h_ref: torch.Tensor,  # (T, D) e.g., layer L-4 or L-8
    topk_concept_ids: torch.Tensor,  # (T, k) long
    topk_scores: torch.Tensor,  # (T, k) float
    noise_floor: Optional[torch.Tensor] = None,  # (C,) precomputed
    cfg: Optional[ProdSigConfig] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Production path: O(T*k) compute.

    Returns:
        sig: (T,) in [0,1] - significance score
        weighted_topk: (T, k) scores after (score - floor)+ and weighting
        filler_mask: (T,) bool - True for filler tokens
    """
    if cfg is None:
        cfg = ProdSigConfig()

    eps = cfg.eps
    device = h_last.device

    # 1) Activation delta: norm of hidden state change
    delta = (h_last - h_ref).norm(dim=-1)  # (T,)

    # 2) Entropy over the k concepts
    H = _entropy_from_topk_scores(topk_scores, cfg.temp, eps)  # (T,)

    # 3) Max above noise floor (only needs top-1)
    if noise_floor is None:
        max_above = topk_scores[:, 0]
        floor_topk = torch.zeros_like(topk_scores)
    else:
        noise_floor = noise_floor.to(device=device, dtype=topk_scores.dtype)
        floor_topk = noise_floor[topk_concept_ids]  # (T, k)
        max_above = topk_scores[:, 0] - floor_topk[:, 0]  # (T,)

    # Optional per-sequence robust normalization
    if cfg.use_seq_robust_norm:
        z_delta = _robust_norm_1d(delta, eps)
        z_H = _robust_norm_1d(H, eps)
        z_max = _robust_norm_1d(max_above, eps)
    else:
        z_delta, z_H, z_max = delta, H, max_above

    # Significance score: high delta, low entropy, high max_above
    logits = cfg.w_delta * z_delta - cfg.w_entropy * z_H + cfg.w_max_above * z_max
    sig = torch.sigmoid(logits)  # (T,)

    # Weight the displayed detections (top-k only)
    cleaned_topk = torch.relu(topk_scores - floor_topk)  # (T,k)
    weighted_topk = cleaned_topk * sig.unsqueeze(-1)  # (T,k)

    # Hard filler tag (optional - requires calibrated thresholds)
    filler_mask = (
        (delta < cfg.delta_thresh)
        & (H > cfg.entropy_thresh)
        & (max_above < cfg.max_above_thresh)
    )

    return sig, weighted_topk, filler_mask


def compute_sig_with_layer_cascade(
    h_early: torch.Tensor,  # (T, D)
    h_mid: torch.Tensor,  # (T, D)
    h_late: torch.Tensor,  # (T, D)
    topk_concept_ids: torch.Tensor,  # (T, k)
    topk_scores: torch.Tensor,  # (T, k) - scores from late layer
    concept_vectors: Optional[torch.Tensor] = None,  # (C, D)
    noise_floor: Optional[torch.Tensor] = None,  # (C,)
    cfg: Optional[ProdSigConfig] = None,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
    """
    Layer-aware significance with entropy cascade.

    Uses early/mid/late checkpoints (existing HatCat layer selection)
    to track entropy changes through the network.

    Decision tokens show: entropy drops early→mid (concept crystallizes)
    Filler tokens show: flat/high entropy throughout

    Args:
        h_early: Hidden states from early layer (definitional)
        h_mid: Hidden states from mid layer (behavioral)
        h_late: Hidden states from late layer (output)
        topk_concept_ids: Which concepts to track (from late layer)
        topk_scores: Concept scores from late layer
        concept_vectors: Projection matrix for entropy computation at layers
        noise_floor: Per-concept noise floor
        cfg: Configuration

    Returns:
        sig: (T,) significance scores
        entropy_by_layer: Dict with 'early', 'mid', 'late' entropy tensors
        filler_mask: (T,) bool
    """
    if cfg is None:
        cfg = ProdSigConfig()

    eps = cfg.eps
    device = h_late.device

    # Compute base significance (using late-mid delta)
    sig_base, weighted_topk, filler_mask = compute_sig_fast(
        h_last=h_late,
        h_ref=h_mid,
        topk_concept_ids=topk_concept_ids,
        topk_scores=topk_scores,
        noise_floor=noise_floor,
        cfg=cfg,
    )

    # Compute entropy at each layer if concept_vectors provided
    entropy_by_layer = {}

    if concept_vectors is not None:
        H_early = compute_entropy_at_layer(
            h_early, concept_vectors, topk_concept_ids, cfg.temp, eps
        )
        H_mid = compute_entropy_at_layer(
            h_mid, concept_vectors, topk_concept_ids, cfg.temp, eps
        )
        H_late = _entropy_from_topk_scores(topk_scores, cfg.temp, eps)

        entropy_by_layer = {
            "early": H_early,
            "mid": H_mid,
            "late": H_late,
        }

        if cfg.use_layer_cascade:
            # Entropy drop signals decision happening
            # Normalize for combination
            if cfg.use_seq_robust_norm:
                delta_H_early_mid = _robust_norm_1d(H_early - H_mid, eps)
                delta_H_mid_late = _robust_norm_1d(H_mid - H_late, eps)
            else:
                delta_H_early_mid = H_early - H_mid
                delta_H_mid_late = H_mid - H_late

            # Positive delta_H means entropy dropped = decision happening
            entropy_drop_signal = delta_H_early_mid + delta_H_mid_late

            # Combine with base significance
            combined_logits = (
                torch.logit(sig_base.clamp(0.01, 0.99))
                + cfg.w_entropy_drop * entropy_drop_signal
            )
            sig = torch.sigmoid(combined_logits)
        else:
            sig = sig_base
    else:
        # No concept vectors - use base significance only
        sig = sig_base
        entropy_by_layer = {"late": _entropy_from_topk_scores(topk_scores, cfg.temp, eps)}

    return sig, entropy_by_layer, filler_mask


def compute_sig_from_full_scores(
    h_last: torch.Tensor,  # (T, D)
    h_ref: torch.Tensor,  # (T, D)
    concept_scores: torch.Tensor,  # (T, C)
    k: int = 16,
    noise_floor: Optional[torch.Tensor] = None,  # (C,)
    cfg: Optional[ProdSigConfig] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Convenience wrapper when you have full concept scores (T, C).

    Not ideal for production if C is huge. Uses torch.topk => O(T*C).

    Returns:
        sig: (T,) significance scores
        topk_ids: (T, k) top-k concept indices
        topk_scores: (T, k) top-k concept scores
        weighted_topk: (T, k) weighted scores
    """
    topk_scores, topk_ids = torch.topk(
        concept_scores, k=min(k, concept_scores.shape[1]), dim=-1
    )
    sig, weighted_topk, _ = compute_sig_fast(
        h_last=h_last,
        h_ref=h_ref,
        topk_concept_ids=topk_ids,
        topk_scores=topk_scores,
        noise_floor=noise_floor,
        cfg=cfg,
    )
    return sig, topk_ids, topk_scores, weighted_topk


class SignificanceScorer:
    """
    Stateful wrapper for significance scoring during generation.

    Caches reference hidden states and provides per-token scoring.
    Integrates with DynamicLensManager layer checkpoints.
    """

    def __init__(
        self,
        lens_manager,  # DynamicLensManager
        cfg: Optional[ProdSigConfig] = None,
    ):
        self.lens_manager = lens_manager
        self.cfg = cfg or ProdSigConfig()

        # Layer indices for checkpoints (will be set on first use)
        self._layer_indices: Optional[Dict[str, int]] = None

        # Reference hidden states (from previous token for delta)
        self._prev_hidden: Optional[Dict[str, torch.Tensor]] = None

        # Noise floor cache
        self._noise_floor: Optional[torch.Tensor] = None

    def _get_layer_indices(self, n_layers: int) -> Dict[str, int]:
        """Get early/mid/late layer indices following HatCat's thirds approach."""
        if self._layer_indices is not None:
            return self._layer_indices

        # HatCat default: split into thirds
        early = n_layers // 3
        mid = n_layers // 2
        late = (2 * n_layers) // 3

        self._layer_indices = {
            "early": early,
            "mid": mid,
            "late": late,
        }
        return self._layer_indices

    def reset(self):
        """Reset state for new generation."""
        self._prev_hidden = None

    def score_tick(
        self,
        hidden_states: Dict[str, torch.Tensor],  # layer_name -> (D,)
        topk_concept_ids: torch.Tensor,  # (k,)
        topk_scores: torch.Tensor,  # (k,)
        concept_vectors: Optional[torch.Tensor] = None,  # (C, D)
    ) -> Tuple[float, Dict[str, float], bool]:
        """
        Score a single generation tick.

        Args:
            hidden_states: Dict with 'early', 'mid', 'late' hidden states
            topk_concept_ids: Top-k concept indices for this token
            topk_scores: Top-k concept scores for this token
            concept_vectors: Optional projection matrix for layer entropy

        Returns:
            significance: Float 0-1
            entropy_by_layer: Dict of layer -> entropy
            is_filler: Bool
        """
        device = topk_scores.device

        # Get hidden states for each layer
        h_late = hidden_states.get("late", hidden_states.get("mid"))
        h_mid = hidden_states.get("mid", h_late)
        h_early = hidden_states.get("early", h_mid)

        # Ensure tensors are 2D (1, D) for compute functions
        if h_late.dim() == 1:
            h_late = h_late.unsqueeze(0)
            h_mid = h_mid.unsqueeze(0)
            h_early = h_early.unsqueeze(0)

        if topk_scores.dim() == 1:
            topk_scores = topk_scores.unsqueeze(0)
            topk_concept_ids = topk_concept_ids.unsqueeze(0)

        # Compute significance with layer cascade
        if concept_vectors is not None and self.cfg.use_layer_cascade:
            sig, entropy_dict, filler = compute_sig_with_layer_cascade(
                h_early=h_early,
                h_mid=h_mid,
                h_late=h_late,
                topk_concept_ids=topk_concept_ids,
                topk_scores=topk_scores,
                concept_vectors=concept_vectors,
                noise_floor=self._noise_floor,
                cfg=self.cfg,
            )
            entropy_by_layer = {k: float(v[0]) for k, v in entropy_dict.items()}
        else:
            # Use reference from previous token if available
            h_ref = self._prev_hidden.get("mid", h_mid) if self._prev_hidden else h_mid

            sig, _, filler = compute_sig_fast(
                h_last=h_late,
                h_ref=h_ref,
                topk_concept_ids=topk_concept_ids,
                topk_scores=topk_scores,
                noise_floor=self._noise_floor,
                cfg=self.cfg,
            )
            entropy_by_layer = {
                "late": float(_entropy_from_topk_scores(topk_scores, self.cfg.temp, self.cfg.eps)[0])
            }

        # Update previous hidden states for next token
        self._prev_hidden = {
            "early": h_early.squeeze(0).detach(),
            "mid": h_mid.squeeze(0).detach(),
            "late": h_late.squeeze(0).detach(),
        }

        return float(sig[0]), entropy_by_layer, bool(filler[0])

    def set_noise_floor(self, noise_floor: torch.Tensor):
        """Set per-concept noise floor from calibration data."""
        self._noise_floor = noise_floor
