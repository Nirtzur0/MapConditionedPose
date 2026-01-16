"""
PyTorch Lightning Training Module

Wraps UELocalizationModel for Lightning training.
"""

try:
    import comet_ml
except ImportError:
    pass

import torch
import pytorch_lightning as pl
import logging
from torch.utils.data import DataLoader
from typing import Dict, Optional
import yaml
from pathlib import Path

from ..models.ue_localization_model import UELocalizationModel
from ..datasets.lmdb_dataset import LMDBRadioLocalizationDataset
# Legacy Zarr support (kept for backward compatibility)
try:
    from ..datasets.radio_dataset import RadioLocalizationDataset, collate_fn
    from ..datasets.combined_dataset import CombinedRadioLocalizationDataset
    ZARR_AVAILABLE = True
except ImportError:
    ZARR_AVAILABLE = False
    logger.warning("Zarr datasets not available (Zarr package not installed)")
from ..datasets.augmentations import RadioAugmentation
from ..physics_loss import PhysicsLoss, PhysicsLossConfig
from ..config.feature_schema import RTFeatureIndex, PHYFeatureIndex, MACFeatureIndex

logger = logging.getLogger(__name__)


def _worker_init_fn(worker_id):
    """Initialize each DataLoader worker - must be at module level for pickling with spawn."""
    import random
    import numpy as np
    worker_seed = torch.initial_seed() % 2**32
    random.seed(worker_seed)
    np.random.seed(worker_seed)


class UELocalizationLightning(pl.LightningModule):
    """Lightning wrapper for UE localization model.
    
    - Training/validation/test loops
    - Optimizer configuration
    - Metrics logging
    - Checkpointing
    """
    
    def __init__(self, config_path: str):
        super().__init__()
        
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Save hyperparameters
        self.save_hyperparameters()
        
        # Build model
        self.model = UELocalizationModel(self.config)
        
        # Loss weights
        self.loss_weights = {
            'coarse_weight': self.config['training']['loss']['coarse_weight'],
            'fine_weight': self.config['training']['loss']['fine_weight'],
            'position_weight': self.config['training']['loss'].get('position_weight', 0.2),
        }

        aux_cfg = self.config['training']['loss'].get('auxiliary', {})
        self.aux_enabled = aux_cfg.get('enabled', False)
        self.aux_weight = aux_cfg.get('weight', 0.0)
        self.aux_task_weights = aux_cfg.get('tasks', {}) or {}

        # Augmentation (GPU)
        augmentation_config = self.config['training'].get('augmentation', None)
        self.augmentor = RadioAugmentation(augmentation_config)
        
        # Physics loss (if enabled)
        self.use_physics_loss = (
            self.config['training']['loss'].get('use_physics_loss', False)
            or self.config.get('physics_loss', {}).get('enabled', False)
        )
        if self.use_physics_loss:
            scene_extent = self.config['dataset']['scene_extent']
            if isinstance(scene_extent, (list, tuple)):
                map_extent = tuple(scene_extent)
            else:
                map_extent = (0.0, 0.0, float(scene_extent), float(scene_extent))
            physics_config = PhysicsLossConfig(
                feature_weights=self.config['physics_loss']['feature_weights'],
                map_extent=map_extent,
                loss_type=self.config['physics_loss'].get('loss_type', 'mse'),
                normalize_features=self.config['physics_loss'].get('normalize_features', True),
            )
            self.physics_loss_fn = PhysicsLoss(physics_config)
            self.lambda_phys = self.config['physics_loss']['lambda_phys']
        else:
            self.physics_loss_fn = None
            self.lambda_phys = 0.0
        
        # Metrics tracking
        self.validation_step_outputs = []
        self.test_step_outputs = []
        self._last_val_sample = None
        self._last_test_sample = None
        self._comet_logged = {'val': False, 'test': False}

    def _get_comet_experiment(self):
        # Retrieve the Comet experiment object from the logger.
        logger = self.logger
        if logger is None:
            return None
        if hasattr(logger, "experiment"):
            return logger.experiment
        if hasattr(logger, "loggers"):
            for item in logger.loggers:
                if hasattr(item, "experiment"):
                    return item.experiment
        return None

    @staticmethod
    def _normalize_db_map(data, min_db=-150.0, max_db=-40.0):
        """Normalize dB map (e.g. path gain) for visualization."""
        import numpy as np
        
        data = np.asarray(data, dtype=np.float32)
        # Handle -inf or very low values
        data = np.nan_to_num(data, nan=min_db, posinf=max_db, neginf=min_db)
        
        # Clip to range
        clipped = np.clip(data, min_db, max_db)
        
        # Scale to [0, 1]
        normalized = (clipped - min_db) / (max_db - min_db)
        return normalized

    @staticmethod
    def _normalize_map(data):
        import numpy as np

        data = np.asarray(data, dtype=np.float32)
        
        # For percentile calculation, treat infs as NaNs to ignore them
        percentile_data = np.where(np.isinf(data), np.nan, data)

        if not np.isfinite(percentile_data).any():
            return np.zeros_like(data, dtype=np.float32)
        
        vmin = np.nanpercentile(percentile_data, 2)
        vmax = np.nanpercentile(percentile_data, 98)

        if vmax - vmin < 1e-6:
            # If dynamic range is negligible, it's a constant map.
            # Show black for zero-maps, gray for non-zero constant maps.
            median = np.nanmedian(percentile_data)
            if median and abs(median) > 1e-6:
                return np.full(data.shape, 0.5, dtype=np.float32)
            else:
                return np.zeros_like(data, dtype=np.float32)

        # Handle NaNs and Infs for scaling. NaNs map to min, Infs to max.
        safe_data = np.nan_to_num(data, nan=vmin, posinf=vmax, neginf=vmin)
        
        # Normalize and clip
        denominator = vmax - vmin
        if denominator > 0:
            normalized = (safe_data - vmin) / denominator
        else:
            normalized = np.zeros_like(safe_data)
        
        return np.clip(normalized, 0.0, 1.0)


    def forward(self, batch: Dict) -> Dict:
        """Forward pass."""
        return self.model(
            batch['measurements'],
            batch['radio_map'],
            batch['osm_map'],
            scene_idx=batch.get('scene_idx'),
        )
    
    def _extract_observed_features(self, measurements: Dict) -> torch.Tensor:
        """
        Extract and summarize observed radio features from measurements.
        
        Returns:
            Aggregated features: [batch, 5] (rsrp, rsrq, sinr, cqi, throughput)
        """
        # Extract features (use mean across temporal dimension, ignoring masked values)
        mask = measurements['mask']  # (batch, seq_len)
        mask_f = mask.float()
        denom = mask_f.sum(dim=1) + 1e-6
        
        # PHY features: rsrp, rsrq, sinr, cqi
        phy_features = measurements['phy_features']  # (batch, seq_len, 8)
        rsrp = (phy_features[:, :, PHYFeatureIndex.RSRP] * mask_f).sum(dim=1) / denom
        rsrq = (phy_features[:, :, PHYFeatureIndex.RSRQ] * mask_f).sum(dim=1) / denom
        sinr = (phy_features[:, :, PHYFeatureIndex.SINR] * mask_f).sum(dim=1) / denom
        cqi = (phy_features[:, :, PHYFeatureIndex.CQI] * mask_f).sum(dim=1) / denom
        
        # MAC features: throughput
        mac_features = measurements['mac_features']  # (batch, seq_len, 6)
        throughput = (mac_features[:, :, MACFeatureIndex.DL_THROUGHPUT] * mask_f).sum(dim=1) / denom
        
        # Stack into (batch, 5)
        observed = torch.stack([rsrp, rsrq, sinr, cqi, throughput], dim=1)
        
        return observed

    def _extract_aux_targets(self, measurements: Dict) -> Dict[str, torch.Tensor]:
        """Extract auxiliary targets from measurement sequences."""
        mask = measurements['mask']  # (batch, seq_len)
        mask_f = mask.float()
        denom = mask_f.sum(dim=1) + 1e-6

        rt_features = measurements['rt_features']
        mac_features = measurements['mac_features']

        nlos = (rt_features[:, :, RTFeatureIndex.IS_NLOS] * mask_f).sum(dim=1) / denom
        num_paths = (rt_features[:, :, RTFeatureIndex.NUM_PATHS] * mask_f).sum(dim=1) / denom
        timing_advance = (mac_features[:, :, MACFeatureIndex.TIMING_ADVANCE] * mask_f).sum(dim=1) / denom

        return {
            'nlos': nlos,
            'num_paths': num_paths,
            'timing_advance': timing_advance,
        }

    def _compute_aux_loss(self, aux_outputs: Dict[str, torch.Tensor], aux_targets: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute auxiliary multi-task loss."""
        losses = []
        if aux_outputs is None:
            return torch.tensor(0.0, device=self.device)

        if 'nlos' in aux_outputs and 'nlos' in aux_targets:
            weight = self.aux_task_weights.get('nlos', 1.0)
            losses.append(weight * torch.nn.functional.binary_cross_entropy_with_logits(
                aux_outputs['nlos'],
                aux_targets['nlos'],
            ))
        if 'num_paths' in aux_outputs and 'num_paths' in aux_targets:
            weight = self.aux_task_weights.get('num_paths', 1.0)
            losses.append(weight * torch.nn.functional.smooth_l1_loss(
                aux_outputs['num_paths'],
                aux_targets['num_paths'],
            ))
        if 'timing_advance' in aux_outputs and 'timing_advance' in aux_targets:
            weight = self.aux_task_weights.get('timing_advance', 1.0)
            losses.append(weight * torch.nn.functional.smooth_l1_loss(
                aux_outputs['timing_advance'],
                aux_targets['timing_advance'],
            ))

        if not losses:
            return torch.tensor(0.0, device=self.device)
        return torch.stack(losses).sum()
    
    def training_step(self, batch: Dict, batch_idx: int) -> torch.Tensor:
        """Run a single training step."""
        # Apply GPU Augmentations
        if self.augmentor.enabled:
            # Position is needed for map transforms to stay consistent
            if 'position' in batch:
                batch['measurements'], batch['radio_map'], batch['osm_map'], augmented_pos = self.augmentor(
                    batch['measurements'], 
                    batch['radio_map'], 
                    batch['osm_map'],
                    position=batch['position']
                )
                
                if augmented_pos is not None:
                    # Clamp augmented position to [0, 1] range to ensure valid grid mapping
                    augmented_pos = torch.clamp(augmented_pos, 0.0, 1.0 - 1e-6)
                    batch['position'] = augmented_pos
                    
                    # Recompute cell_grid after augmentation (flip/rotate/scale)
                    # grid_size is usually 32
                    grid_size = self.model.grid_size
                    grid_x = (augmented_pos[:, 0] * grid_size).long()
                    # Flip Y for grid index: 1.0 -> 0, 0.0 -> grid_size-1
                    grid_y = ((1.0 - augmented_pos[:, 1]) * grid_size).long()
                    
                    # Ensure indices are within [0, grid_size-1]
                    grid_x = torch.clamp(grid_x, 0, grid_size - 1)
                    grid_y = torch.clamp(grid_y, 0, grid_size - 1)
                    
                    batch['cell_grid'] = grid_y * grid_size + grid_x
            else:
                 batch['measurements'], batch['radio_map'], batch['osm_map'], _ = self.augmentor(
                    batch['measurements'], batch['radio_map'], batch['osm_map']
                )

        outputs = self.forward(batch)
        
        targets = {
            'position': batch['position'],
            'cell_grid': batch['cell_grid'],
        }
        losses = self.model.compute_loss(outputs, targets, self.loss_weights)

        if self.aux_enabled:
            aux_targets = self._extract_aux_targets(batch['measurements'])
            aux_loss = self._compute_aux_loss(outputs.get('aux_outputs'), aux_targets)
            losses['aux_loss'] = aux_loss
            losses['loss'] = losses['loss'] + self.aux_weight * aux_loss
        
        # Add physics loss if enabled
        radio_maps = batch.get('radio_maps', batch.get('radio_map'))
        if self.use_physics_loss and radio_maps is not None:
            pred_position = outputs['predicted_position']
            observed_features = batch.get('observed_features', self._extract_observed_features(batch['measurements']))
            
            physics_loss = self.physics_loss_fn(
                predicted_xy=pred_position,
                observed_features=observed_features,
                radio_maps=radio_maps,
            )
            
            losses['physics_loss'] = physics_loss
            losses['loss'] = losses['loss'] + self.lambda_phys * physics_loss
        
        # Log losses
        self.log('train_loss', losses['loss'], on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_coarse_loss', losses['coarse_loss'], on_step=True, on_epoch=True)
        self.log('train_fine_loss', losses['fine_loss'], on_step=True, on_epoch=True)
        self.log('train_position_loss', losses.get('position_loss', 0.0), on_step=True, on_epoch=True)
        if self.use_physics_loss:
            self.log('train_physics_loss', losses.get('physics_loss', 0.0), on_step=True, on_epoch=True)
        if self.aux_enabled:
            self.log('train_aux_loss', losses.get('aux_loss', 0.0), on_step=True, on_epoch=True)
        
        return losses['loss']
    
    def validation_step(self, batch: Dict, batch_idx: int):
        """Run a single validation step."""
        outputs = self.forward(batch)
        
        targets = {
            'position': batch['position'],
            'cell_grid': batch['cell_grid'],
        }
        losses = self.model.compute_loss(outputs, targets, self.loss_weights)

        if self.aux_enabled:
            aux_targets = self._extract_aux_targets(batch['measurements'])
            aux_loss = self._compute_aux_loss(outputs.get('aux_outputs'), aux_targets)
            losses['aux_loss'] = aux_loss
            losses['loss'] = losses['loss'] + self.aux_weight * aux_loss
        
        # Add physics loss if enabled
        radio_maps = batch.get('radio_maps', batch.get('radio_map'))
        if self.use_physics_loss and radio_maps is not None:
            pred_position = outputs['predicted_position']
            observed_features = batch.get('observed_features', self._extract_observed_features(batch['measurements']))
            
            physics_loss = self.physics_loss_fn(
                predicted_xy=pred_position,
                observed_features=observed_features,
                radio_maps=radio_maps,
            )
            losses['physics_loss'] = physics_loss
            losses['loss'] = losses['loss'] + self.lambda_phys * physics_loss
        
        # Compute errors
        # Compute errors in meters
        pred_pos = outputs['predicted_position']
        true_pos = batch['position']
        # Distance in normalized space [0,1] * extent [meters]
        # scene_extent shape: [batch]
        extent = batch.get('scene_extent', torch.tensor(512.0, device=pred_pos.device))
        if isinstance(extent, (int, float)):
            extent = torch.tensor(extent, device=pred_pos.device)
        errors = torch.norm(pred_pos - true_pos, dim=-1) * extent

        # Always visualize the first sample of the first batch for consistency across epochs
        if batch_idx == 0:
            self._last_val_sample = {
                'radio_map': batch['radio_map'][0].detach().cpu(),
                'osm_map': batch['osm_map'][0].detach().cpu(),
                'true_pos': true_pos[0].detach().cpu(),
                'pred_pos': pred_pos[0].detach().cpu(),
                'top_k_indices': outputs['top_k_indices'][0].detach().cpu(),
                'top_k_probs': outputs['top_k_probs'][0].detach().cpu(),
                'fine_offsets': outputs['fine_offsets'][0].detach().cpu(),
                'fine_uncertainties': outputs['fine_uncertainties'][0].detach().cpu(),
            }
        
        # Store for epoch-end aggregation
        output_dict = {
            'loss': losses['loss'],
            'coarse_loss': losses['coarse_loss'],
            'fine_loss': losses['fine_loss'],
            'errors': errors,
        }
        if self.use_physics_loss:
            output_dict['physics_loss'] = losses.get('physics_loss', torch.tensor(0.0))
        if self.aux_enabled:
            output_dict['aux_loss'] = losses.get('aux_loss', torch.tensor(0.0))
        
        self.validation_step_outputs.append(output_dict)
        
        return losses['loss']
    
    def on_validation_epoch_end(self):
        """Aggregate validation metrics at epoch end."""
        if not self.validation_step_outputs:
            return
        
        # Aggregate losses
        avg_loss = torch.stack([x['loss'] for x in self.validation_step_outputs]).mean()
        avg_coarse = torch.stack([x['coarse_loss'] for x in self.validation_step_outputs]).mean()
        avg_fine = torch.stack([x['fine_loss'] for x in self.validation_step_outputs]).mean()
        
        if self.use_physics_loss:
            avg_phys = torch.stack([x['physics_loss'] for x in self.validation_step_outputs]).mean()
            self.log('val_physics_loss', avg_phys)
        if self.aux_enabled:
            avg_aux = torch.stack([x['aux_loss'] for x in self.validation_step_outputs]).mean()
            self.log('val_aux_loss', avg_aux)
        
        # Aggregate errors
        all_errors = torch.cat([x['errors'] for x in self.validation_step_outputs])
        
        median_error = torch.median(all_errors)
        rmse = torch.sqrt(torch.mean(all_errors ** 2))
        percentile_67 = torch.quantile(all_errors, 0.67)
        percentile_90 = torch.quantile(all_errors, 0.90)
        percentile_95 = torch.quantile(all_errors, 0.95)
        
        # Success rates
        success_5m = (all_errors <= 5.0).float().mean() * 100
        success_10m = (all_errors <= 10.0).float().mean() * 100
        
        # Log metrics
        self.log('val_loss', avg_loss, prog_bar=True)
        self.log('val_coarse_loss', avg_coarse)
        self.log('val_fine_loss', avg_fine)
        self.log('val_median_error', median_error, prog_bar=True)
        self.log('val_rmse', rmse)
        self.log('val_p67', percentile_67)
        self.log('val_p90', percentile_90)
        self.log('val_p95', percentile_95)
        self.log('val_success_5m', success_5m)
        self.log('val_success_10m', success_10m)

        self._log_comet_visuals('val')
        
        # Clear
        self.validation_step_outputs.clear()
    
    def test_step(self, batch: Dict, batch_idx: int):
        """Run a single test step."""
        outputs = self.forward(batch)
        
        # Compute errors
        # Compute errors in meters
        pred_pos = outputs['predicted_position']
        true_pos = batch['position']
        # Distance in normalized space [0,1] * extent [meters]
        extent = batch.get('scene_extent', torch.tensor(512.0, device=pred_pos.device))
        if isinstance(extent, (int, float)):
            extent = torch.tensor(extent, device=pred_pos.device)
        errors = torch.norm(pred_pos - true_pos, dim=-1) * extent
        
        # Store results
        self.test_step_outputs.append({
            'errors': errors,
            'predictions': pred_pos,
            'ground_truth': true_pos,
            'uncertainties': outputs['fine_uncertainties'][:, 0, :],
        })

        # Always visualize the first sample of the first batch
        if batch_idx == 0:
            self._last_test_sample = {
                'radio_map': batch['radio_map'][0].detach().cpu(),
                'osm_map': batch['osm_map'][0].detach().cpu(),
                'true_pos': true_pos[0].detach().cpu(),
                'pred_pos': pred_pos[0].detach().cpu(),
                'top_k_indices': outputs['top_k_indices'][0].detach().cpu(),
                'top_k_probs': outputs['top_k_probs'][0].detach().cpu(),
                'fine_offsets': outputs['fine_offsets'][0].detach().cpu(),
                'fine_uncertainties': outputs['fine_uncertainties'][0].detach().cpu(),
            }
        
        return errors

    def _render_gmm_heatmap(self, h, w, top_k_indices, top_k_probs, fine_offsets, fine_uncertainties):
        """Render Gaussian Mixture Model heatmap."""
        import numpy as np
        
        # Create grid coordinates [0, 1] with FLIPPED Y for image coords
        # Image coords: (0,0) = top-left, Y increases downward
        # Normalized coords: (0,0) = bottom-left, Y increases upward
        x = np.linspace(0, 1, w)
        y = np.linspace(1, 0, h)  # Flip Y: top=1, bottom=0
        X, Y = np.meshgrid(x, y)
        pos = np.stack([X, Y], axis=-1) # [H, W, 2]
        
        # Get component parameters
        cell_size = self.model.cell_size
        top_k_indices = top_k_indices.to(self.device).long()
        grid_size = self.model.grid_size
        
        # row = index // grid_size, col = index % grid_size
        gy = top_k_indices // grid_size
        gx = top_k_indices % grid_size
        
        # Center of cell in [0, 1] coords
        center_x = (gx.float() + 0.5) * cell_size
        # Flip Y back for visualization: grid_y=0 is top=1.0, grid_y=N-1 is bottom=0.0
        center_y = (grid_size - 1 - gy.float() + 0.5) * cell_size
        
        centers = torch.stack([center_x, center_y], dim=-1).cpu().numpy() # [K, 2]
        
        # Add predicted offsets
        centers = centers + fine_offsets.numpy() # [K, 2]
        
        # Scale variances to be visible (add minimum variance to prevent too-peaked Gaussians)
        variances = (fine_uncertainties.numpy() ** 2) + 0.001  # [K, 2]
        probs = top_k_probs.numpy() # [K]
        
        # Normalize probs
        probs = probs / (probs.sum() + 1e-6)
        
        # Compute heatmap
        heatmap = np.zeros((h, w), dtype=np.float32)
        
        for k in range(len(probs)):
            mu = centers[k]
            var = variances[k]
            pi = probs[k]
            
            # Evaluate Gaussian with epsilon to prevent division by zero
            dx = (pos[..., 0] - mu[0]) ** 2 / (2 * var[0] + 1e-6)
            dy = (pos[..., 1] - mu[1]) ** 2 / (2 * var[1] + 1e-6)
            
            component = pi * np.exp(-(dx + dy))
            heatmap += component
            
        # Normalize heatmap to [0, 1]
        if heatmap.max() > 0:
            heatmap = heatmap / heatmap.max()
            
        return heatmap

    def _log_comet_visuals(self, split: str, errors: Optional[torch.Tensor] = None):
        experiment = self._get_comet_experiment()
        if experiment is None:
            return

        sample = self._last_val_sample if split == 'val' else self._last_test_sample
        if sample is None:
            return

        import matplotlib.pyplot as plt
        import numpy as np

        radio_map = sample['radio_map'].numpy()
        osm_map = sample['osm_map'].numpy()
        true_pos = sample['true_pos'].numpy()
        pred_pos = sample['pred_pos'].numpy()

        # Use adaptive percentile-based normalization for radio map (handles variable dB ranges)
        radio_img = self._normalize_map(radio_map[0])

        h, w = radio_img.shape

        # OSM visualization: R=Height(0), G=Footprint(2), B=Road(3)+Terrain(4) combined
        # This ensures road data shows when available in other datasets
        osm_height = self._normalize_map(osm_map[0]) if osm_map.shape[0] > 0 else np.zeros((h, w), dtype=np.float32)
        osm_footprint = self._normalize_map(osm_map[2]) if osm_map.shape[0] > 2 else np.zeros((h, w), dtype=np.float32)
        # Combine road and terrain channels for blue (road takes precedence where it exists)
        osm_road = osm_map[3] if osm_map.shape[0] > 3 else np.zeros((h, w), dtype=np.float32)
        osm_terrain = osm_map[4] if osm_map.shape[0] > 4 else np.zeros((h, w), dtype=np.float32)
        osm_blue = self._normalize_map(np.maximum(osm_road, osm_terrain * 0.3))  # Roads bright, terrain subtle
        osm_rgb = np.stack([osm_height, osm_footprint, osm_blue], axis=-1)

        # Convert from normalized [0,1] coords (bottom-left origin) to image pixels (top-left origin)
        true_px = true_pos[0] * (w - 1)
        true_py = (1.0 - true_pos[1]) * (h - 1)  # Flip Y-axis
        pred_px = pred_pos[0] * (w - 1)
        pred_py = (1.0 - pred_pos[1]) * (h - 1)  # Flip Y-axis
        
        # Render GMM
        gmm_heatmap = self._render_gmm_heatmap(
            h, w, 
            sample['top_k_indices'], 
            sample['top_k_probs'], 
            sample['fine_offsets'], 
            sample['fine_uncertainties']
        )
        
        # Apply colormap to GMM with adaptive transparency
        # Use jet/turbo but ensure zero probability is fully transparent
        gmm_colored = plt.cm.turbo(gmm_heatmap) # RGBA [H, W, 4]
        
        # Set alpha channel: 
        # Low prob -> Transparent
        # High prob -> More opaque but still see-through (max 0.7)
        gmm_colored[..., 3] = np.clip(gmm_heatmap * 0.8, 0.0, 0.8)

        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        axes[0].imshow(radio_img, cmap='inferno')
        axes[0].set_title("Radio Map (Path Gain)")
        axes[0].axis("off")
        
        axes[1].imshow(osm_rgb)
        axes[1].set_title("OSM Map (R:Height G:Build B:Road/Terrain)")
        axes[1].axis("off")
        
        # Overlay GMM + Points on top of OSM map for context
        # 1. Base map
        axes[2].imshow(osm_rgb) 
        # 2. GMM overlay
        axes[2].imshow(gmm_colored)
        # 3. Points
        axes[2].scatter([true_px], [true_py], c="lime", s=60, edgecolors='black', label="True", zorder=10)
        # axes[2].scatter([pred_px], [pred_py], c="red", s=60, marker="x", label="Pred", zorder=10)
        axes[2].set_title(f"{split.upper()} GMM Prediction + True Pos")
        axes[2].legend(loc="lower right", fontsize=8)
        axes[2].axis("off")
        fig.tight_layout()

        if hasattr(experiment, "log_figure"):
            experiment.log_figure(figure=fig, figure_name=f"{split}_maps")
        
        plt.close(fig)

        if errors is not None and hasattr(experiment, "log_histogram"):
            experiment.log_histogram(errors.detach().cpu().numpy(), name=f"{split}_error_hist_m")


    
    def on_test_epoch_end(self):
        """Aggregate test metrics at epoch end."""
        if not self.test_step_outputs:
            return
        
        # Aggregate errors
        all_errors = torch.cat([x['errors'] for x in self.test_step_outputs])
        
        median_error = torch.median(all_errors)
        mean_error = torch.mean(all_errors)
        rmse = torch.sqrt(torch.mean(all_errors ** 2))
        percentile_67 = torch.quantile(all_errors, 0.67)
        percentile_90 = torch.quantile(all_errors, 0.90)
        percentile_95 = torch.quantile(all_errors, 0.95)
        
        # Success rates
        success_5m = (all_errors <= 5.0).float().mean() * 100
        success_10m = (all_errors <= 10.0).float().mean() * 100
        success_20m = (all_errors <= 20.0).float().mean() * 100
        success_50m = (all_errors <= 50.0).float().mean() * 100
        
        # Log metrics
        self.log('test_median_error', median_error)
        self.log('test_mean_error', mean_error)
        self.log('test_rmse', rmse)
        self.log('test_p67', percentile_67)
        self.log('test_p90', percentile_90)
        self.log('test_p95', percentile_95)
        self.log('test_success_5m', success_5m)
        self.log('test_success_10m', success_10m)
        self.log('test_success_20m', success_20m)
        self.log('test_success_50m', success_50m)

        self._log_comet_visuals('test', errors=all_errors)
        
        # Print summary
        logger.info("\n" + "="*60)
        logger.info("TEST SET RESULTS")
        logger.info("="*60)
        logger.info(f"Median Error:     {median_error:.2f} m")
        logger.info(f"Mean Error:       {mean_error:.2f} m")
        logger.info(f"RMSE:             {rmse:.2f} m")
        logger.info(f"67th Percentile:  {percentile_67:.2f} m")
        logger.info(f"90th Percentile:  {percentile_90:.2f} m")
        logger.info(f"95th Percentile:  {percentile_95:.2f} m")
        logger.info(f"Success @ 5m:     {success_5m:.1f}%")
        logger.info(f"Success @ 10m:    {success_10m:.1f}%")
        logger.info(f"Success @ 20m:    {success_20m:.1f}%")
        logger.info(f"Success @ 50m:    {success_50m:.1f}%")
        logger.info("="*60 + "\n")
        
        # Clear
        self.test_step_outputs.clear()
    
    def configure_optimizers(self):
        """Configure optimizer and learning rate scheduler."""
        cfg = self.config['training']
        
        # Optimizer
        if cfg['optimizer'] == 'adamw':
            optimizer = torch.optim.AdamW(
                self.parameters(),
                lr=cfg['learning_rate'],
                weight_decay=cfg['weight_decay'],
            )
        else:
            raise ValueError(f"Unknown optimizer: {cfg['optimizer']}")
        
        # Scheduler
        if cfg['scheduler'] == 'cosine_with_warmup':
            from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

            total_steps = None
            if getattr(self, "trainer", None) is not None:
                total_steps = getattr(self.trainer, "estimated_stepping_batches", None)
            if total_steps is None:
                total_steps = cfg.get('num_epochs', 1)
            total_steps = max(1, int(total_steps))

            warmup_steps = int(cfg.get('warmup_steps', 0))
            if warmup_steps >= total_steps:
                warmup_steps = max(0, total_steps - 1)

            if warmup_steps > 0:
                warmup_scheduler = LinearLR(
                    optimizer,
                    start_factor=0.01,
                    total_iters=warmup_steps,
                )

                cosine_scheduler = CosineAnnealingLR(
                    optimizer,
                    T_max=max(1, total_steps - warmup_steps),
                    eta_min=cfg['learning_rate'] * 0.01,
                )

                scheduler = SequentialLR(
                    optimizer,
                    schedulers=[warmup_scheduler, cosine_scheduler],
                    milestones=[warmup_steps],
                )
            else:
                scheduler = CosineAnnealingLR(
                    optimizer,
                    T_max=total_steps,
                    eta_min=cfg['learning_rate'] * 0.01,
                )
            
            return {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': scheduler,
                    'interval': 'step',
                },
            }
        
        return optimizer
    
    def _build_dataset(self, split: str):
        dataset_config = self.config['dataset']
        
        # Get augmentation config (only for training)
        augmentation = None
        if split == 'train' and 'training' in self.config:
            augmentation = self.config['training'].get('augmentation', None)
        
        # Support both LMDB and Zarr paths (prefer LMDB)
        # LMDB: train_lmdb_paths, val_lmdb_paths, test_lmdb_paths
        # Zarr (DEPRECATED): train_zarr_paths, val_zarr_paths, test_zarr_paths
        dataset_config = self.config['dataset']
        
        # Check for LMDB paths first (preferred)
        train_paths = dataset_config.get('train_lmdb_paths', dataset_config.get('train_zarr_paths', []))
        val_paths = dataset_config.get('val_lmdb_paths', dataset_config.get('val_zarr_paths', []))
        test_paths = dataset_config.get('test_lmdb_paths', dataset_config.get('test_zarr_paths', []))
        
        # Select paths based on split
        if split == 'train':
            paths = train_paths
        elif split == 'val':
            paths = val_paths if val_paths else train_paths  # Fallback to train if no val
        elif split == 'test':
            paths = test_paths if test_paths else val_paths  # Fallback to val if no test
        else:
            raise ValueError(f"Unknown split: {split}")
        
        if not paths:
            raise ValueError(f"No dataset paths configured for split '{split}'")

        # Filter out empty paths
        valid_paths = [p for p in paths if p and str(p) not in ['', '.']]
        if not valid_paths:
            raise ValueError(f"No valid dataset paths found for split {split}. Original paths: {paths}")
        
        # Check if using LMDB or Zarr
        use_lmdb = dataset_config.get('use_lmdb', False)
        first_path = Path(valid_paths[0])
        
        # Auto-detect if path is LMDB
        if str(first_path).endswith('.lmdb') or (first_path.is_dir() and (first_path / 'data.mdb').exists()):
            use_lmdb = True
        
        if use_lmdb:
            # Use LMDB dataset (multiprocessing-friendly!)
            logger.info(f"✓ Using LMDB dataset for {split} split (multiprocessing-safe)")
            if len(valid_paths) > 1:
                logger.warning(f"Multiple LMDB paths not yet supported, using first: {valid_paths[0]}")
            
            return LMDBRadioLocalizationDataset(
                lmdb_path=str(valid_paths[0]),
                split=split,
                map_resolution=dataset_config['map_resolution'],
                scene_extent=dataset_config['scene_extent'],
                normalize=dataset_config['normalize_features'],
                handle_missing=dataset_config['handle_missing_values'],
                augmentation=augmentation,
                split_seed=dataset_config.get('split_seed', 42),
                map_cache_size=dataset_config.get('map_cache_size', 0),
                sequence_length=dataset_config.get('sequence_length', 0),
                max_cells=dataset_config.get('max_cells', 2),
            )
        else:
            # Use Zarr dataset (DEPRECATED - has multiprocessing issues)
            if not ZARR_AVAILABLE:
                raise ImportError(
                    "Zarr dataset requested but Zarr not installed. "
                    "Either install zarr (pip install zarr) or migrate to LMDB datasets."
                )
            
            logger.warning(f"⚠️  Using deprecated Zarr dataset for {split} split - migrate to LMDB for better multiprocessing")
            logger.warning("   Set num_workers=0 in config to avoid multiprocessing issues with Zarr")
            
            return CombinedRadioLocalizationDataset(
                zarr_paths=valid_paths,
                split=split,
                map_resolution=dataset_config['map_resolution'],
                scene_extent=dataset_config['scene_extent'],
                normalize=dataset_config['normalize_features'],
                handle_missing=dataset_config['handle_missing_values'],
                augmentation=augmentation,
                split_seed=dataset_config.get('split_seed', 42),
                map_cache_size=dataset_config.get('map_cache_size', 0),
                sequence_length=dataset_config.get('sequence_length', 0),
                max_cells=dataset_config.get('max_cells', 2),
            )

    def train_dataloader(self):
        """Create training dataloader."""
        dataset = self._build_dataset('train')

        num_batches = max(
            1,
            (len(dataset) + self.config['training']['batch_size'] - 1)
            // self.config['training']['batch_size'],
        )
        logger.info(f"Training samples: {len(dataset)} ({num_batches} batches)")
        
        # Use pin_memory for GPU training (unless on MPS which has warnings)
        num_workers = self.config['infrastructure']['num_workers']
        use_pin_memory = torch.cuda.is_available()
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            use_pin_memory = False
            
        return DataLoader(
            dataset,
            batch_size=self.config['training']['batch_size'],
            shuffle=True,
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=use_pin_memory,
            persistent_workers=(num_workers > 0),
            prefetch_factor=2 if num_workers > 0 else None,
            worker_init_fn=_worker_init_fn if num_workers > 0 else None,
        )
    
    def val_dataloader(self):
        """Create validation dataloader."""
        dataset = self._build_dataset('val')

        num_batches = max(
            1,
            (len(dataset) + self.config['training']['batch_size'] - 1)
            // self.config['training']['batch_size'],
        )
        logger.info(f"Validation samples: {len(dataset)} ({num_batches} batches)")
        
        # Use pin_memory for GPU training (unless on MPS which has warnings)
        num_workers = self.config['infrastructure']['num_workers']
        use_pin_memory = torch.cuda.is_available()
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            use_pin_memory = False
            
        return DataLoader(
            dataset,
            batch_size=self.config['training']['batch_size'],
            shuffle=False,
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=use_pin_memory,
            persistent_workers=(num_workers > 0),
            prefetch_factor=2 if num_workers > 0 else None,
            worker_init_fn=_worker_init_fn if num_workers > 0 else None,
        )
    
    def test_dataloader(self):
        """Create test dataloader."""
        dataset = self._build_dataset('test')

        num_batches = max(
            1,
            (len(dataset) + self.config['training']['batch_size'] - 1)
            // self.config['training']['batch_size'],
        )
        logger.info(f"Test samples: {len(dataset)} ({num_batches} batches)")
        
        num_workers = self.config['infrastructure']['num_workers']
        return DataLoader(
            dataset,
            batch_size=self.config['training']['batch_size'],
            shuffle=False,
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=True,
            persistent_workers=(num_workers > 0),
            prefetch_factor=2 if num_workers > 0 else None,
        )
