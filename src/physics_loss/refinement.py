"""
Inference-Time Position Refinement using Physics Loss.

Provides gradient-based optimization to refine network predictions using
physics-consistency loss. Useful for high-stakes predictions or when
network confidence is low.
"""

import torch
from typing import Optional, Tuple, Dict
from dataclasses import dataclass
import logging

from .core import compute_physics_loss
from .differentiable_lookup import differentiable_lookup

logger = logging.getLogger(__name__)



@dataclass
class RefineConfig:
    """Configuration for position refinement."""
    
    # Number of gradient descent steps
    num_steps: int = 10
    
    # Learning rate (in meters per step)
    learning_rate: float = 0.1
    
    # Scale factor for density term (-log p(y))
    # If 0.0, uses only physics loss. If > 0, combines physics + network density.
    density_weight: float = 1.0

    # Temperature for coarse logits (confidence calibration).
    # >1.0 softens, <1.0 sharpens.
    coarse_logit_temperature: float = 1.0

    # Kernel width for hypothesis density (as fraction of map extent).
    # Used when mixture_params are provided without per-candidate variances.
    candidate_sigma_ratio: float = 0.05

    # Combine confidence sources: "min", "product", "coarse", "fine".
    confidence_combine: str = "min"

    
    # Refine only low confidence predictions
    min_confidence_threshold: Optional[float] = 0.6
    
    # Clip refined positions to map extent
    clip_to_extent: bool = True

    # Limit how far refinement can move from the initial prediction.
    # Ratio is relative to the maximum map extent.
    max_displacement_ratio: Optional[float] = 0.05

    # Only accept refinement if total energy improves.
    require_improvement: bool = True
    energy_tolerance: float = 1e-6
    
    # Map extent for clipping (x_min, y_min, x_max, y_max)
    map_extent: Tuple[float, float, float, float] = (0.0, 0.0, 512.0, 512.0)
    
    # Physics loss configuration for normalization/weights
    physics_config: Optional[object] = None


def _apply_temperature(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    if temperature <= 0:
        raise ValueError("coarse_logit_temperature must be > 0")
    if temperature == 1.0:
        return logits
    return logits / temperature


def _resolve_candidate_sigma(config: RefineConfig) -> float:
    x_min, y_min, x_max, y_max = config.map_extent
    extent = max(abs(x_max - x_min), abs(y_max - y_min))
    sigma = extent * config.candidate_sigma_ratio
    return max(float(sigma), 1e-3)


def _combine_confidence(
    coarse_conf: Optional[torch.Tensor],
    fine_conf: Optional[torch.Tensor],
    mode: str,
) -> Optional[torch.Tensor]:
    if coarse_conf is None:
        return fine_conf
    if fine_conf is None:
        return coarse_conf
    if mode == "product":
        return coarse_conf * fine_conf
    if mode == "coarse":
        return coarse_conf
    if mode == "fine":
        return fine_conf
    if mode != "min":
        logger.warning("Unknown confidence_combine=%s; defaulting to 'min'.", mode)
    return torch.minimum(coarse_conf, fine_conf)


def _calibrate_mixture_params(
    mixture_params: Optional[Dict[str, torch.Tensor]],
    config: RefineConfig,
) -> Optional[Dict[str, torch.Tensor]]:
    if mixture_params is None:
        return None
    calibrated = dict(mixture_params)
    logits = calibrated.get('logits')
    if logits is not None:
        calibrated['logits'] = _apply_temperature(logits, config.coarse_logit_temperature)
    return calibrated


def _compute_confidence(
    mixture_params: Dict[str, torch.Tensor],
    config: RefineConfig,
) -> Optional[torch.Tensor]:
    logits = mixture_params.get('logits')
    if logits is None:
        return None
    probs = torch.softmax(logits, dim=-1)
    coarse_conf = probs.max(dim=-1).values
    return _combine_confidence(coarse_conf, None, config.confidence_combine)


def compute_density_nll(
    xy: torch.Tensor,
    mixture_params: Dict[str, torch.Tensor],
    candidate_sigma: float,
) -> torch.Tensor:
    """
    Compute negative log likelihood of position xy under a hypothesis density.

    Args:
        xy: (batch, 2) candidate positions
        mixture_params: Dictionary with:
            - logits: (batch, K) hypothesis weights (unnormalized log probs)
            - means: (batch, K, 2) hypothesis positions
        candidate_sigma: Kernel width (scalar, in same units as xy)

    Returns:
        nll: (batch,) negative log likelihood
    """
    # xy: [batch, 2] -> [batch, 1, 2]
    xy_expanded = xy.unsqueeze(1)
    
    means = mixture_params['means']  # [batch, K, 2]
    logits = mixture_params['logits']  # [batch, K]
    
    # LogSoftmax for weights
    # log_pi = log_softmax(logits)
    log_pi = torch.log_softmax(logits, dim=-1)
    
    # Isotropic Gaussian kernel around each hypothesis
    sigma = torch.as_tensor(candidate_sigma, device=xy.device, dtype=xy.dtype)
    sigma_sq = sigma ** 2
    residuals = xy_expanded - means
    log_norm = -0.5 * torch.log(2 * torch.pi * sigma_sq)
    log_prob_dim = log_norm - 0.5 * (residuals ** 2) / sigma_sq
    log_prob_xy = log_prob_dim.sum(dim=-1)  # [batch, K]
    
    # Mixture Log Likelihood
    # log sum_k exp(log_pi_k + log_N_k)
    log_mixture = torch.logsumexp(log_pi + log_prob_xy, dim=-1)
    
    return -log_mixture

def refine_position(
    initial_xy: torch.Tensor,
    observed_features: torch.Tensor,
    radio_maps: torch.Tensor,
    config: RefineConfig,
    confidence: Optional[torch.Tensor] = None,
    mixture_params: Optional[Dict[str, torch.Tensor]] = None,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Refine predicted positions using gradient-based optimization on Energy function.
    
    Energy E(y) = PhysicsLoss(y) + density_weight * NLL(y | hypothesis density)
    
    Args:
        initial_xy: (batch, 2) initial position estimates
        observed_features: (batch, C) observed radio features
        radio_maps: (batch, C, H, W) precomputed radio maps
        config: Refinement configuration
        mixture_params: Optional params for density term (means, logits)
        confidence: Optional confidence scores to mask refinement. If None and
            mixture_params are provided, confidence is derived from calibrated
            hypothesis weights.
            
    Returns:
        refined_xy: (batch, 2) refined positions
        info: refinement statistics
    """
    batch_size = initial_xy.shape[0]
    device = initial_xy.device

    calibrated_mix = _calibrate_mixture_params(mixture_params, config)
    if confidence is None and calibrated_mix is not None:
        confidence = _compute_confidence(calibrated_mix, config)
    if confidence is not None:
        if not torch.is_tensor(confidence):
            confidence = torch.as_tensor(confidence, device=device, dtype=initial_xy.dtype)
        else:
            confidence = confidence.to(device=device, dtype=initial_xy.dtype)
        if confidence.ndim == 0:
            confidence = confidence.expand(batch_size)
        elif confidence.shape[0] != batch_size:
            raise ValueError("confidence must have shape [batch] when provided")
    
    # Determine which samples to refine
    if confidence is not None and config.min_confidence_threshold is not None:
        refine_mask = confidence < config.min_confidence_threshold
        num_refined = refine_mask.sum().item()
    else:
        refine_mask = torch.ones(batch_size, dtype=torch.bool, device=device)
        num_refined = batch_size
    
    refined_xy = initial_xy.clone()
    
    if num_refined == 0:
        return refined_xy, {}

    # Setup Physics Loss
    # Use the class to handle weights and normalization consistently
    from .core import PhysicsLoss, PhysicsLossConfig as CoreConfig
    
    if config.physics_config:
        loss_fn = PhysicsLoss(config.physics_config)
    else:
        # Default fallback
        loss_fn = PhysicsLoss(CoreConfig())
    loss_fn.to(device)

    # Prepare inputs for optimization
    xy_opt = initial_xy[refine_mask].detach().clone().requires_grad_(True)
    obs_opt = observed_features[refine_mask]
    maps_opt = radio_maps[refine_mask]
    
    # Slice mixture params if available
    mix_opt = None
    if calibrated_mix is not None and config.density_weight > 0:
        if isinstance(calibrated_mix, dict) and 'logits' in calibrated_mix and 'means' in calibrated_mix:
            refine_indices = torch.where(refine_mask)[0]
            if calibrated_mix['logits'].ndim > 1:
                mix_opt = {
                    'logits': calibrated_mix['logits'][refine_indices],
                    'means': calibrated_mix['means'][refine_indices],
                }
            else:
                mix_opt = calibrated_mix
    
    candidate_sigma = _resolve_candidate_sigma(config)
    optimizer = torch.optim.Adam([xy_opt], lr=config.learning_rate)

    def _per_sample_physics_loss(xy: torch.Tensor, obs: torch.Tensor, maps: torch.Tensor) -> torch.Tensor:
        """Compute per-sample physics loss to gate refinement."""
        sim = differentiable_lookup(
            predicted_xy=xy,
            radio_maps=maps,
            map_extent=loss_fn.config.map_extent,
            padding_mode=loss_fn.config.padding_mode,
        )
        if loss_fn.config.normalize_features:
            obs_mean = obs.mean(dim=0, keepdim=True)
            obs_std = obs.std(dim=0, keepdim=True, unbiased=False) + 1e-6
            obs_norm = (obs - obs_mean) / obs_std
            sim_norm = (sim - obs_mean) / obs_std
        else:
            obs_norm = obs
            sim_norm = sim

        residuals = obs_norm - sim_norm
        if loss_fn.config.loss_type == 'mse':
            feature_losses = residuals ** 2
        elif loss_fn.config.loss_type == 'huber':
            feature_losses = torch.nn.functional.huber_loss(
                sim_norm,
                obs_norm,
                reduction='none',
                delta=loss_fn.config.huber_delta,
            )
        else:
            raise ValueError(f"Unknown loss_type: {loss_fn.config.loss_type}")

        feature_names = list(loss_fn.config.channel_names)
        weights = [loss_fn.config.feature_weights.get(name, 1.0) for name in feature_names]
        weights_t = torch.tensor(weights, device=feature_losses.device, dtype=feature_losses.dtype)
        if feature_losses.shape[1] != weights_t.shape[0]:
            if feature_losses.shape[1] < weights_t.shape[0]:
                weights_t = weights_t[:feature_losses.shape[1]]
            else:
                pad = torch.ones(
                    feature_losses.shape[1] - weights_t.shape[0],
                    device=feature_losses.device,
                    dtype=feature_losses.dtype,
                )
                weights_t = torch.cat([weights_t, pad])
        return (feature_losses * weights_t.unsqueeze(0)).sum(dim=1)
    
    initial_loss_phys = None
    
    for step in range(config.num_steps):
        optimizer.zero_grad()
        
        # 1. Physics Term
        loss_phys = loss_fn(xy_opt, obs_opt, maps_opt)
        
        # 2. Density Term (Optional)
        loss_dens = torch.tensor(0.0, device=device)
        if mix_opt:
            loss_dens = compute_density_nll(xy_opt, mix_opt, candidate_sigma).mean()
            
        # Total Energy
        total_loss = loss_phys + config.density_weight * loss_dens
        
        if step == 0:
            initial_loss_phys = loss_phys.item()
            
        total_loss.backward()
        optimizer.step()

        if config.max_displacement_ratio is not None:
            x_min, y_min, x_max, y_max = config.map_extent
            max_extent = max(abs(x_max - x_min), abs(y_max - y_min))
            max_disp = max_extent * float(config.max_displacement_ratio)
            with torch.no_grad():
                delta = xy_opt - initial_xy[refine_mask]
                dist = torch.norm(delta, dim=1, keepdim=True)
                scale = torch.clamp(max_disp / (dist + 1e-8), max=1.0)
                xy_opt.copy_(initial_xy[refine_mask] + delta * scale)
        
        # Clip to extent if configured
        if config.clip_to_extent:
            x_min, y_min, x_max, y_max = config.map_extent
            with torch.no_grad():
                xy_opt[:, 0].clamp_(x_min, x_max)
                xy_opt[:, 1].clamp_(y_min, y_max)
            
    # Update outputs
    refined_xy[refine_mask] = xy_opt.detach()

    num_reverted = 0
    if config.require_improvement:
        with torch.no_grad():
            refine_indices = torch.where(refine_mask)[0]
            init_phys = _per_sample_physics_loss(initial_xy[refine_mask], obs_opt, maps_opt)
            ref_phys = _per_sample_physics_loss(refined_xy[refine_mask], obs_opt, maps_opt)
            init_energy = init_phys.clone()
            ref_energy = ref_phys.clone()
            if mix_opt:
                init_energy += config.density_weight * compute_density_nll(
                    initial_xy[refine_mask],
                    mix_opt,
                    candidate_sigma,
                )
                ref_energy += config.density_weight * compute_density_nll(
                    refined_xy[refine_mask],
                    mix_opt,
                    candidate_sigma,
                )
            worse = ref_energy > (init_energy + config.energy_tolerance)
            if worse.any():
                refined_xy[refine_indices[worse]] = initial_xy[refine_indices[worse]]
                num_reverted = int(worse.sum().item())
    
    # Calculate distance moved
    distance_moved = torch.zeros(batch_size, device=device)
    distance_moved[refine_mask] = torch.norm(xy_opt.detach() - initial_xy[refine_mask], dim=1)
    
    # Final check
    with torch.no_grad():
        final_loss_phys = loss_fn(refined_xy[refine_mask], obs_opt, maps_opt).item() if num_refined > 0 else 0.0

    info = {
        'loss_initial': initial_loss_phys or 0.0,
        'loss_final': final_loss_phys,
        'num_refined': num_refined,
        'num_reverted': num_reverted,
        'distance_moved': distance_moved
    }
    if confidence is not None:
        info['confidence_mean'] = confidence.mean().item()
        info['confidence_min'] = confidence.min().item()
    return refined_xy, info


def batch_refine_positions(
    initial_xy: torch.Tensor,
    observed_features: torch.Tensor,
    radio_maps: torch.Tensor,
    config: RefineConfig,
    mixture_params: Optional[Dict[str, torch.Tensor]] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Batch refinement wrapper.
    """
    # Simply call refine_position (logic merged for simplicity)
    # This logic assumed candidates before, but now we usually refine the top prediction.
    # For simplicity in this 'best solution' iteration, we act on the best estimate.
    refined, info = refine_position(
        initial_xy,
        observed_features,
        radio_maps,
        config,
        mixture_params=mixture_params,
    )
    # Return refinement and dummy loss tensor
    return refined, torch.zeros(initial_xy.shape[0])
