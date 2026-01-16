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

    # Scale factor for fine-head variances (uncertainty calibration).
    fine_variance_scale: float = 1.0

    # Reference sigma for confidence conversion.
    # If None, uses fine_sigma_ref_ratio * max(map_extent).
    fine_sigma_ref: Optional[float] = None

    # Ratio for deriving sigma reference from map extent.
    fine_sigma_ref_ratio: float = 0.05

    # Combine confidence sources: "min", "product", "coarse", "fine".
    confidence_combine: str = "min"

    # Minimum variance to avoid degenerate Gaussians.
    min_variance: float = 1e-6
    
    # Refine only low confidence predictions
    min_confidence_threshold: Optional[float] = 0.6
    
    # Clip refined positions to map extent
    clip_to_extent: bool = True
    
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


def _calibrate_variances(variances: torch.Tensor, config: RefineConfig) -> torch.Tensor:
    variances = variances * config.fine_variance_scale
    variances = torch.nan_to_num(
        variances,
        nan=config.min_variance,
        posinf=config.min_variance,
        neginf=config.min_variance,
    )
    return torch.clamp(variances, min=config.min_variance)


def _resolve_sigma_ref(config: RefineConfig) -> float:
    if config.fine_sigma_ref is not None and config.fine_sigma_ref > 0:
        return float(config.fine_sigma_ref)
    x_min, y_min, x_max, y_max = config.map_extent
    extent = max(abs(x_max - x_min), abs(y_max - y_min))
    sigma_ref = extent * config.fine_sigma_ref_ratio
    return max(sigma_ref, config.min_variance ** 0.5)


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
    variances = calibrated.get('vars')
    if variances is not None:
        calibrated['vars'] = _calibrate_variances(variances, config)
    return calibrated


def _compute_confidence(
    mixture_params: Dict[str, torch.Tensor],
    config: RefineConfig,
) -> Optional[torch.Tensor]:
    logits = mixture_params.get('logits')
    variances = mixture_params.get('vars')
    coarse_conf = None
    fine_conf = None
    probs = None
    if logits is not None:
        probs = torch.softmax(logits, dim=-1)
        coarse_conf = probs.max(dim=-1).values
    if variances is not None:
        sigma = torch.sqrt(variances)
        sigma_mean = sigma.mean(dim=-1)
        if probs is not None and probs.shape == sigma_mean.shape:
            sigma_weighted = (probs * sigma_mean).sum(dim=-1)
        else:
            sigma_weighted = sigma_mean.min(dim=-1).values
        sigma_ref = _resolve_sigma_ref(config)
        fine_conf = torch.exp(-sigma_weighted / sigma_ref)
    return _combine_confidence(coarse_conf, fine_conf, config.confidence_combine)


def compute_density_nll(
    xy: torch.Tensor,
    mixture_params: Dict[str, torch.Tensor],
) -> torch.Tensor:
    """
    Compute negative log likelihood of position xy under the Gaussian Mixture.
    
    Args:
        xy: (batch, 2) candidate positions
        mixture_params: Dictionary with:
            - logits: (batch, K) mixture weights (unnormalized log probs)
            - means: (batch, K, 2) component means
            - vars: (batch, K, 2) component variances
            
    Returns:
        nll: (batch,) negative log likelihood
    """
    # xy: [batch, 2] -> [batch, 1, 2]
    xy_expanded = xy.unsqueeze(1)
    
    means = mixture_params['means']  # [batch, K, 2]
    variances = mixture_params['vars']  # [batch, K, 2]
    logits = mixture_params['logits']  # [batch, K]
    
    # LogSoftmax for weights
    # log_pi = log_softmax(logits)
    log_pi = torch.log_softmax(logits, dim=-1)
    
    # Gaussian Log Likelihood
    # log N(x; mu, sigma^2) = -0.5*log(2*pi*sigma^2) - 0.5*(x-mu)^2/sigma^2
    # Sum over x,y dimensions for multivariate diagonal gaussian
    residuals = xy_expanded - means
    log_prob_dim = -0.5 * torch.log(2 * torch.pi * variances) - 0.5 * (residuals ** 2) / variances
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
    
    Energy E(y) = PhysicsLoss(y) + density_weight * NLL(y | network)
    
    Args:
        initial_xy: (batch, 2) initial position estimates
        observed_features: (batch, C) observed radio features
        radio_maps: (batch, C, H, W) precomputed radio maps
        config: Refinement configuration
        mixture_params: Optional params for density term (means, vars, logits)
        confidence: Optional confidence scores to mask refinement. If None and
            mixture_params are provided, confidence is derived from calibrated
            logits/variances.
            
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
        # Check if mixture_params has the expected structure
        if isinstance(calibrated_mix, dict) and 'logits' in calibrated_mix:
            # Convert boolean mask to indices for proper indexing
            refine_indices = torch.where(refine_mask)[0]
            # Only index if the tensors are multi-dimensional
            if calibrated_mix['logits'].ndim > 1:
                mix_opt = {
                    'logits': calibrated_mix['logits'][refine_indices],
                    'means': calibrated_mix['means'][refine_indices],
                    'vars': calibrated_mix['vars'][refine_indices]
                }
            else:
                # If 1D, it's likely a single-sample case - don't index
                mix_opt = calibrated_mix
    
    optimizer = torch.optim.Adam([xy_opt], lr=config.learning_rate)
    
    initial_loss_phys = None
    
    for step in range(config.num_steps):
        optimizer.zero_grad()
        
        # 1. Physics Term
        loss_phys = loss_fn(xy_opt, obs_opt, maps_opt)
        
        # 2. Density Term (Optional)
        loss_dens = torch.tensor(0.0, device=device)
        if mix_opt:
            loss_dens = compute_density_nll(xy_opt, mix_opt).mean()
            
        # Total Energy
        total_loss = loss_phys + config.density_weight * loss_dens
        
        if step == 0:
            initial_loss_phys = loss_phys.item()
            
        total_loss.backward()
        optimizer.step()
        
        # Clip to extent if configured
        if config.clip_to_extent:
            x_min, y_min, x_max, y_max = config.map_extent
            with torch.no_grad():
                xy_opt[:, 0].clamp_(x_min, x_max)
                xy_opt[:, 1].clamp_(y_min, y_max)
            
    # Update outputs
    refined_xy[refine_mask] = xy_opt.detach()
    
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
