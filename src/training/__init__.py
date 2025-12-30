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
from ..datasets.radio_dataset import RadioLocalizationDataset, collate_fn
from ..datasets.combined_dataset import CombinedRadioLocalizationDataset
from ..physics_loss import PhysicsLoss, PhysicsLossConfig

logger = logging.getLogger(__name__)


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
        }
        
        # Physics loss (if enabled)
        self.use_physics_loss = self.config['training']['loss'].get('use_physics_loss', False)
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

    def _log_comet_visuals(self, split: str, errors: Optional[torch.Tensor] = None):
        experiment = self._get_comet_experiment()
        if experiment is None or self._comet_logged.get(split):
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

        true_px = true_pos[0] * (w - 1)
        true_py = true_pos[1] * (h - 1)
        pred_px = pred_pos[0] * (w - 1)
        pred_py = pred_pos[1] * (h - 1)

        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        axes[0].imshow(radio_img, cmap='inferno')
        axes[0].set_title("Radio Map (Path Gain)")
        axes[0].axis("off")
        axes[1].imshow(osm_rgb)
        axes[1].set_title("OSM Map (R:Height G:Build B:Road/Terrain)")
        axes[1].axis("off")
        axes[2].imshow(osm_rgb)
        axes[2].scatter([true_px], [true_py], c="lime", s=40, label="True")
        axes[2].scatter([pred_px], [pred_py], c="red", s=40, marker="x", label="Pred")
        axes[2].set_title(f"{split.upper()} Prediction Overlay")
        axes[2].legend(loc="lower right", fontsize=8)
        axes[2].axis("off")
        fig.tight_layout()

        if hasattr(experiment, "log_figure"):
            experiment.log_figure(figure=fig, figure_name=f"{split}_maps")
        
        # Also save as PNG asset for reliable viewing
        import io
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        buf.seek(0)
        if hasattr(experiment, "log_asset"):
            experiment.log_asset(buf, file_name=f"{split}_maps.png")
        
        plt.close(fig)

        if errors is not None and hasattr(experiment, "log_histogram"):
            experiment.log_histogram(errors.detach().cpu().numpy(), name=f"{split}_error_hist_m")

        self._comet_logged[split] = True
    
    def forward(self, batch: Dict) -> Dict:
        """Forward pass."""
        return self.model(
            batch['measurements'],
            batch['radio_map'],
            batch['osm_map'],
        )
    
    def _extract_observed_features(self, measurements: Dict) -> torch.Tensor:
        """
        Extract and summarize observed radio features from measurements.
        
        Returns:
            Aggregated features: [batch, 5] (path_gain, snr, sinr, throughput, bler)
        """
        batch_size = measurements['rt_features'].shape[0]
        device = measurements['rt_features'].device
        
        # Extract features (use mean across temporal dimension, ignoring masked values)
        mask = measurements['mask']  # (batch, seq_len)
        
        # RT features: path_gain (0)
        rt_features = measurements['rt_features']  # (batch, seq_len, 10 or 8)
        path_gain = (rt_features[:, :, 0] * mask).sum(dim=1) / (mask.sum(dim=1) + 1e-6)
        
        # PHY features: snr (2), sinr (3)
        phy_features = measurements['phy_features']  # (batch, seq_len, 8)
        snr = (phy_features[:, :, 2] * mask).sum(dim=1) / (mask.sum(dim=1) + 1e-6)
        sinr = (phy_features[:, :, 3] * mask).sum(dim=1) / (mask.sum(dim=1) + 1e-6)
        
        # MAC features: throughput (0), bler (1)
        mac_features = measurements['mac_features']  # (batch, seq_len, 6)
        throughput = (mac_features[:, :, 0] * mask).sum(dim=1) / (mask.sum(dim=1) + 1e-6)
        bler = (mac_features[:, :, 1] * mask).sum(dim=1) / (mask.sum(dim=1) + 1e-6)
        
        # Stack into (batch, 5)
        observed = torch.stack([path_gain, snr, sinr, throughput, bler], dim=1)
        
        return observed
    
    def training_step(self, batch: Dict, batch_idx: int) -> torch.Tensor:
        """Run a single training step."""
        outputs = self.forward(batch)
        
        targets = {
            'position': batch['position'],
            'cell_grid': batch['cell_grid'],
        }
        losses = self.model.compute_loss(outputs, targets, self.loss_weights)
        
        # Add physics loss if enabled
        if self.use_physics_loss and 'radio_maps' in batch:
            pred_position = outputs['predicted_position']
            observed_features = batch.get('observed_features', self._extract_observed_features(batch['measurements']))
            
            physics_loss = self.physics_loss_fn(
                predicted_xy=pred_position,
                observed_features=observed_features,
                radio_maps=batch['radio_maps'],
            )
            
            losses['physics_loss'] = physics_loss
            losses['loss'] = losses['loss'] + self.lambda_phys * physics_loss
        
        # Log losses
        self.log('train_loss', losses['loss'], on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_coarse_loss', losses['coarse_loss'], on_step=True, on_epoch=True)
        self.log('train_fine_loss', losses['fine_loss'], on_step=True, on_epoch=True)
        if self.use_physics_loss:
            self.log('train_physics_loss', losses.get('physics_loss', 0.0), on_step=True, on_epoch=True)
        
        return losses['loss']
    
    def validation_step(self, batch: Dict, batch_idx: int):
        """Run a single validation step."""
        outputs = self.forward(batch)
        
        targets = {
            'position': batch['position'],
            'cell_grid': batch['cell_grid'],
        }
        losses = self.model.compute_loss(outputs, targets, self.loss_weights)
        
        # Add physics loss if enabled
        if self.use_physics_loss and 'radio_maps' in batch:
            pred_position = outputs['predicted_position']
            observed_features = batch.get('observed_features', self._extract_observed_features(batch['measurements']))
            
            physics_loss = self.physics_loss_fn(
                predicted_xy=pred_position,
                observed_features=observed_features,
                radio_maps=batch['radio_maps'],
            )
            losses['physics_loss'] = physics_loss
            losses['loss'] = losses['loss'] + self.lambda_phys * physics_loss
        
        # Compute errors
        # Compute errors in meters
        pred_pos = outputs['predicted_position']
        true_pos = batch['position']
        # Distance in normalized space [0,1] * extent [meters]
        # sample_extent shape: [batch]
        extent = batch.get('sample_extent', torch.tensor(512.0, device=pred_pos.device))
        errors = torch.norm(pred_pos - true_pos, dim=-1) * extent

        if batch_idx == 0 and self._last_val_sample is None:
            self._last_val_sample = {
                'radio_map': batch['radio_map'][0].detach().cpu(),
                'osm_map': batch['osm_map'][0].detach().cpu(),
                'true_pos': true_pos[0].detach().cpu(),
                'pred_pos': pred_pos[0].detach().cpu(),
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
        extent = batch.get('sample_extent', torch.tensor(512.0, device=pred_pos.device))
        errors = torch.norm(pred_pos - true_pos, dim=-1) * extent
        
        # Store results
        self.test_step_outputs.append({
            'errors': errors,
            'predictions': pred_pos,
            'ground_truth': true_pos,
            'uncertainties': outputs['fine_uncertainties'][:, 0, :],
        })

        if batch_idx == 0 and self._last_test_sample is None:
            self._last_test_sample = {
                'radio_map': batch['radio_map'][0].detach().cpu(),
                'osm_map': batch['osm_map'][0].detach().cpu(),
                'true_pos': true_pos[0].detach().cpu(),
                'pred_pos': pred_pos[0].detach().cpu(),
            }
        
        return errors
    
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
        
        # Determine paths and split mode
        paths = None
        single_path = None
        target_split = split

        # Check for new explicit configuration
        train_paths = dataset_config.get('train_zarr_paths')
        test_path = dataset_config.get('test_zarr_path')
        test_on_eval = dataset_config.get('test_on_eval', False)

        if train_paths or (test_path and test_on_eval):
            if split == 'test' and test_on_eval and test_path:
                # Test on explicit evaluation dataset (use all data)
                single_path = test_path
                target_split = 'all'
            elif split == 'train':
                # Train on training datasets (80% split)
                paths = train_paths 
                target_split = 'train_80'
            elif split == 'val':
                # Val on training datasets (20% split)
                paths = train_paths
                target_split = 'val_20'
            elif split == 'test':
                 # Fallback: Use validation set as test set if no explicit test set provided
                 paths = train_paths
                 target_split = 'val_20' 
                 logger.warning("No test set configured. Using val_20 split for testing.")
        else:
            # Legacy fallback
            split_key = f"{split}_zarr_paths"
            paths = dataset_config.get(split_key) or dataset_config.get('zarr_paths')
            single_path = dataset_config.get(f"{split}_zarr_path") or dataset_config.get('zarr_path')

        # Instantiate Dataset
        if paths:
            # Filter out empty paths (including Path('') or '.')
            valid_paths = [p for p in paths if p and str(p) not in ['', '.']]
            if not valid_paths:
                raise ValueError(f"No valid dataset paths found for split {split}. Original paths: {paths}")
            
            return CombinedRadioLocalizationDataset(
                zarr_paths=valid_paths,
                split=target_split,
                map_resolution=dataset_config['map_resolution'],
                scene_extent=dataset_config['scene_extent'],
                normalize=dataset_config['normalize_features'],
                handle_missing=dataset_config['handle_missing_values'],
                augmentation=augmentation,
            )

        if single_path:
            return RadioLocalizationDataset(
                zarr_path=single_path,
                split=target_split,
                map_resolution=dataset_config['map_resolution'],
                scene_extent=dataset_config['scene_extent'],
                normalize=dataset_config['normalize_features'],
                handle_missing=dataset_config['handle_missing_values'],
                augmentation=augmentation,
            )
        
        raise ValueError(f"No dataset configuration found for split {split}")

    def train_dataloader(self):
        """Create training dataloader."""
        dataset = self._build_dataset('train')

        num_batches = max(
            1,
            (len(dataset) + self.config['training']['batch_size'] - 1)
            // self.config['training']['batch_size'],
        )
        logger.info(f"Training samples: {len(dataset)} ({num_batches} batches)")
        
        # Disable pin_memory on MPS to avoid warnings
        use_pin_memory = True
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            use_pin_memory = False
            
        return DataLoader(
            dataset,
            batch_size=self.config['training']['batch_size'],
            shuffle=True,
            num_workers=self.config['infrastructure']['num_workers'],
            collate_fn=collate_fn,
            pin_memory=use_pin_memory,
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
        
        # Disable pin_memory on MPS to avoid warnings
        use_pin_memory = True
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            use_pin_memory = False
            
        return DataLoader(
            dataset,
            batch_size=self.config['training']['batch_size'],
            shuffle=False,
            num_workers=self.config['infrastructure']['num_workers'],
            collate_fn=collate_fn,
            pin_memory=use_pin_memory,
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
        
        return DataLoader(
            dataset,
            batch_size=self.config['training']['batch_size'],
            shuffle=False,
            num_workers=self.config['infrastructure']['num_workers'],
            collate_fn=collate_fn,
            pin_memory=True,
        )
