"""
PyTorch Lightning Training Module

Wraps UELocalizationModel for training with Lightning infrastructure.
"""

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
    
    Handles:
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
            physics_config = PhysicsLossConfig(
                feature_weights=self.config['physics_loss']['feature_weights'],
                map_extent=tuple(self.config['dataset']['scene_extent']),
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
    def _normalize_map(data):
        import numpy as np

        data = np.asarray(data, dtype=np.float32)
        
        # Debugging logs
        logger.debug(f"Normalizing map data. Shape: {data.shape}, "
                     f"Non-finite: {np.count_nonzero(~np.isfinite(data))}, "
                     f"Min: {np.nanmin(data, initial=None)}, Max: {np.nanmax(data, initial=None)}, "
                     f"Mean: {np.nanmean(data)}")

        # For percentile calculation, treat infs as NaNs so they are ignored
        percentile_data = np.where(np.isinf(data), np.nan, data)

        if not np.isfinite(percentile_data).any():
            return np.zeros_like(data, dtype=np.float32)
        
        vmin = np.nanpercentile(percentile_data, 2)
        vmax = np.nanpercentile(percentile_data, 98)

        logger.debug(f"Normalization params: vmin={vmin}, vmax={vmax}")

        if vmax - vmin < 1e-6:
            # If dynamic range is negligible, it's a constant map.
            # Show black for zero-maps, gray for non-zero constant maps.
            median = np.nanmedian(percentile_data)
            if median and abs(median) > 1e-6:
                return np.full(data.shape, 0.5, dtype=np.float32)
            else:
                return np.zeros_like(data, dtype=np.float32)

        # Handle NaNs and Infs in the original data for scaling.
        # NaNs will be mapped to the min value (black).
        # Infs will be mapped to the max value (white).
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

        radio_img = self._normalize_map(radio_map[0])
        h, w = radio_img.shape

        osm_channels = []
        for idx in (0, 2, 3):
            if idx < osm_map.shape[0]:
                osm_channels.append(self._normalize_map(osm_map[idx]))
            else:
                osm_channels.append(np.zeros((h, w), dtype=np.float32))
        osm_rgb = np.stack(osm_channels, axis=-1)

        extent = float(self.config['dataset'].get('scene_extent', max(h, w)))
        true_px = (true_pos[0] / extent) * (w - 1)
        true_py = (true_pos[1] / extent) * (h - 1)
        pred_px = (pred_pos[0] / extent) * (w - 1)
        pred_py = (pred_pos[1] / extent) * (h - 1)

        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        axes[0].imshow(radio_img, cmap='inferno')
        axes[0].set_title("Radio Map (Path Gain)")
        axes[0].axis("off")
        axes[1].imshow(osm_rgb)
        axes[1].set_title("OSM Map (Height/Footprint/Road)")
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
        Extract observed radio features from measurements.
        
        Args:
            measurements: Dict with RT/PHY/MAC features
                - rt_features: (batch, seq_len, 8) [path_gain, toa, aoa_az, aoa_el, ...]
                - phy_features: (batch, seq_len, 10) [rsrp, rsrq, snr, sinr, ...]
                - mac_features: (batch, seq_len, 6) [throughput, bler, ...]
                
        Returns:
            observed: (batch, 7) features [path_gain, toa, aoa, snr, sinr, throughput, bler]
        """
        batch_size = measurements['rt_features'].shape[0]
        device = measurements['rt_features'].device
        
        # Extract features (use mean across temporal dimension, ignoring masked values)
        mask = measurements['mask']  # (batch, seq_len)
        
        # RT features: path_gain (0), toa (1), aoa_azimuth (2)
        rt_features = measurements['rt_features']  # (batch, seq_len, 8)
        path_gain = (rt_features[:, :, 0] * mask).sum(dim=1) / (mask.sum(dim=1) + 1e-6)
        toa = (rt_features[:, :, 1] * mask).sum(dim=1) / (mask.sum(dim=1) + 1e-6)
        aoa = (rt_features[:, :, 2] * mask).sum(dim=1) / (mask.sum(dim=1) + 1e-6)
        
        # PHY features: snr (2), sinr (3)
        phy_features = measurements['phy_features']  # (batch, seq_len, 10)
        snr = (phy_features[:, :, 2] * mask).sum(dim=1) / (mask.sum(dim=1) + 1e-6)
        sinr = (phy_features[:, :, 3] * mask).sum(dim=1) / (mask.sum(dim=1) + 1e-6)
        
        # MAC features: throughput (0), bler (1)
        mac_features = measurements['mac_features']  # (batch, seq_len, 6)
        throughput = (mac_features[:, :, 0] * mask).sum(dim=1) / (mask.sum(dim=1) + 1e-6)
        bler = (mac_features[:, :, 1] * mask).sum(dim=1) / (mask.sum(dim=1) + 1e-6)
        
        # Stack into (batch, 7)
        observed = torch.stack([path_gain, toa, aoa, snr, sinr, throughput, bler], dim=1)
        
        return observed
    
    def training_step(self, batch: Dict, batch_idx: int) -> torch.Tensor:
        """Training step."""
        # Forward pass
        outputs = self.forward(batch)
        
        # Compute losses
        targets = {
            'position': batch['position'],
            'cell_grid': batch['cell_grid'],
        }
        losses = self.model.compute_loss(outputs, targets, self.loss_weights)
        
        # Add physics loss if enabled
        if self.use_physics_loss and 'radio_maps' in batch:
            # Extract predicted position (best candidate from fine head)
            pred_position = outputs['predicted_position']  # (batch, 2)
            
            # Extract observed features from measurements
            # Assume batch['observed_features'] contains: [path_gain, toa, aoa, snr, sinr, throughput, bler]
            # If not provided, need to extract from measurements
            if 'observed_features' in batch:
                observed_features = batch['observed_features']
            else:
                # Extract mean features from measurements (simplified)
                # In practice, should aggregate across temporal measurements
                observed_features = self._extract_observed_features(batch['measurements'])
            
            # Compute physics loss
            physics_loss = self.physics_loss_fn(
                predicted_xy=pred_position,
                observed_features=observed_features,
                radio_maps=batch['radio_maps'],  # (batch, C, H, W)
            )
            
            # Add to total loss
            losses['physics_loss'] = physics_loss
            losses['loss'] = losses['loss'] + self.lambda_phys * physics_loss
        
        # Log losses
        self.log('train_loss', losses['loss'], on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_coarse_loss', losses['coarse_loss'], on_step=False, on_epoch=True)
        self.log('train_fine_loss', losses['fine_loss'], on_step=False, on_epoch=True)
        if self.use_physics_loss:
            self.log('train_physics_loss', losses.get('physics_loss', 0.0), on_step=False, on_epoch=True)
        
        return losses['loss']
    
    def validation_step(self, batch: Dict, batch_idx: int):
        """Validation step."""
        # Forward pass
        outputs = self.forward(batch)
        
        # Compute losses
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
        pred_pos = outputs['predicted_position']
        true_pos = batch['position']
        errors = torch.norm(pred_pos - true_pos, dim=-1)  # [B]

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
        """Aggregate validation metrics."""
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
        """Test step."""
        # Forward pass
        outputs = self.forward(batch)
        
        # Compute errors
        pred_pos = outputs['predicted_position']
        true_pos = batch['position']
        errors = torch.norm(pred_pos - true_pos, dim=-1)  # [B]
        
        # Store
        self.test_step_outputs.append({
            'errors': errors,
            'predictions': pred_pos,
            'ground_truth': true_pos,
            'uncertainties': outputs['fine_uncertainties'][:, 0, :],  # Best prediction
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
        """Aggregate test metrics."""
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
        
        # Log
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
            
            warmup_scheduler = LinearLR(
                optimizer,
                start_factor=0.01,
                total_iters=cfg['warmup_steps'],
            )
            
            cosine_scheduler = CosineAnnealingLR(
                optimizer,
                T_max=cfg['num_epochs'] - cfg['warmup_steps'],
                eta_min=cfg['learning_rate'] * 0.01,
            )
            
            scheduler = SequentialLR(
                optimizer,
                schedulers=[warmup_scheduler, cosine_scheduler],
                milestones=[cfg['warmup_steps']],
            )
            
            return {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': scheduler,
                    'interval': 'step',
                },
            }
        
        return optimizer
    
    def train_dataloader(self):
        """Create training dataloader."""
        dataset_config = self.config['dataset']
        if 'zarr_paths' in dataset_config and dataset_config['zarr_paths']:
            dataset = CombinedRadioLocalizationDataset(
                zarr_paths=dataset_config['zarr_paths'],
                split='train',
                map_resolution=dataset_config['map_resolution'],
                scene_extent=dataset_config['scene_extent'],
                normalize=dataset_config['normalize_features'],
                handle_missing=dataset_config['handle_missing_values'],
            )
        else:
            dataset = RadioLocalizationDataset(
                zarr_path=dataset_config['zarr_path'],
                split='train',
                map_resolution=dataset_config['map_resolution'],
                scene_extent=dataset_config['scene_extent'],
                normalize=dataset_config['normalize_features'],
                handle_missing=dataset_config['handle_missing_values'],
            )

        num_batches = max(1, (len(dataset) + self.config['training']['batch_size'] - 1) // self.config['training']['batch_size'])
        logger.info(f"Training samples: {len(dataset)} ({num_batches} batches)")
        
        return DataLoader(
            dataset,
            batch_size=self.config['training']['batch_size'],
            shuffle=True,
            num_workers=self.config['infrastructure']['num_workers'],
            collate_fn=collate_fn,
            pin_memory=True,
        )
    
    def val_dataloader(self):
        """Create validation dataloader."""
        dataset_config = self.config['dataset']
        if 'zarr_paths' in dataset_config and dataset_config['zarr_paths']:
            dataset = CombinedRadioLocalizationDataset(
                zarr_paths=dataset_config['zarr_paths'],
                split='val',
                map_resolution=dataset_config['map_resolution'],
                scene_extent=dataset_config['scene_extent'],
                normalize=dataset_config['normalize_features'],
                handle_missing=dataset_config['handle_missing_values'],
            )
        else:
            dataset = RadioLocalizationDataset(
                zarr_path=dataset_config['zarr_path'],
                split='val',
                map_resolution=dataset_config['map_resolution'],
                scene_extent=dataset_config['scene_extent'],
                normalize=dataset_config['normalize_features'],
                handle_missing=dataset_config['handle_missing_values'],
            )

        num_batches = max(1, (len(dataset) + self.config['training']['batch_size'] - 1) // self.config['training']['batch_size'])
        logger.info(f"Validation samples: {len(dataset)} ({num_batches} batches)")
        
        return DataLoader(
            dataset,
            batch_size=self.config['training']['batch_size'],
            shuffle=False,
            num_workers=self.config['infrastructure']['num_workers'],
            collate_fn=collate_fn,
            pin_memory=True,
        )
    
    def test_dataloader(self):
        """Create test dataloader."""
        dataset_config = self.config['dataset']
        if 'zarr_paths' in dataset_config and dataset_config['zarr_paths']:
            dataset = CombinedRadioLocalizationDataset(
                zarr_paths=dataset_config['zarr_paths'],
                split='test',
                map_resolution=dataset_config['map_resolution'],
                scene_extent=dataset_config['scene_extent'],
                normalize=dataset_config['normalize_features'],
                handle_missing=dataset_config['handle_missing_values'],
            )
        else:
            dataset = RadioLocalizationDataset(
                zarr_path=dataset_config['zarr_path'],
                split='test',
                map_resolution=dataset_config['map_resolution'],
                scene_extent=dataset_config['scene_extent'],
                normalize=dataset_config['normalize_features'],
                handle_missing=dataset_config['handle_missing_values'],
            )

        num_batches = max(1, (len(dataset) + self.config['training']['batch_size'] - 1) // self.config['training']['batch_size'])
        logger.info(f"Test samples: {len(dataset)} ({num_batches} batches)")
        
        return DataLoader(
            dataset,
            batch_size=self.config['training']['batch_size'],
            shuffle=False,
            num_workers=self.config['infrastructure']['num_workers'],
            collate_fn=collate_fn,
            pin_memory=True,
        )
