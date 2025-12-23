"""
PyTorch Lightning Training Module

Wraps UELocalizationModel for training with Lightning infrastructure.
"""

import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from typing import Dict, Optional
import yaml
from pathlib import Path

from ..models.ue_localization_model import UELocalizationModel
from ..datasets.radio_dataset import RadioLocalizationDataset, collate_fn


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
        
        # Metrics tracking
        self.validation_step_outputs = []
        self.test_step_outputs = []
    
    def forward(self, batch: Dict) -> Dict:
        """Forward pass."""
        return self.model(
            batch['measurements'],
            batch['radio_map'],
            batch['osm_map'],
        )
    
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
        
        # Log losses
        self.log('train_loss', losses['loss'], on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_coarse_loss', losses['coarse_loss'], on_step=False, on_epoch=True)
        self.log('train_fine_loss', losses['fine_loss'], on_step=False, on_epoch=True)
        
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
        
        # Compute errors
        pred_pos = outputs['predicted_position']
        true_pos = batch['position']
        errors = torch.norm(pred_pos - true_pos, dim=-1)  # [B]
        
        # Store for epoch-end aggregation
        self.validation_step_outputs.append({
            'loss': losses['loss'],
            'coarse_loss': losses['coarse_loss'],
            'fine_loss': losses['fine_loss'],
            'errors': errors,
        })
        
        return losses['loss']
    
    def on_validation_epoch_end(self):
        """Aggregate validation metrics."""
        if not self.validation_step_outputs:
            return
        
        # Aggregate losses
        avg_loss = torch.stack([x['loss'] for x in self.validation_step_outputs]).mean()
        avg_coarse = torch.stack([x['coarse_loss'] for x in self.validation_step_outputs]).mean()
        avg_fine = torch.stack([x['fine_loss'] for x in self.validation_step_outputs]).mean()
        
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
        
        # Print summary
        print("\n" + "="*60)
        print("TEST SET RESULTS")
        print("="*60)
        print(f"Median Error:     {median_error:.2f} m")
        print(f"Mean Error:       {mean_error:.2f} m")
        print(f"RMSE:             {rmse:.2f} m")
        print(f"67th Percentile:  {percentile_67:.2f} m")
        print(f"90th Percentile:  {percentile_90:.2f} m")
        print(f"95th Percentile:  {percentile_95:.2f} m")
        print(f"Success @ 5m:     {success_5m:.1f}%")
        print(f"Success @ 10m:    {success_10m:.1f}%")
        print(f"Success @ 20m:    {success_20m:.1f}%")
        print(f"Success @ 50m:    {success_50m:.1f}%")
        print("="*60 + "\n")
        
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
        dataset = RadioLocalizationDataset(
            zarr_path=self.config['dataset']['zarr_path'],
            split='train',
            map_resolution=self.config['dataset']['map_resolution'],
            scene_extent=self.config['dataset']['scene_extent'],
            normalize=self.config['dataset']['normalize_features'],
            handle_missing=self.config['dataset']['handle_missing_values'],
        )
        
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
        dataset = RadioLocalizationDataset(
            zarr_path=self.config['dataset']['zarr_path'],
            split='val',
            map_resolution=self.config['dataset']['map_resolution'],
            scene_extent=self.config['dataset']['scene_extent'],
            normalize=self.config['dataset']['normalize_features'],
            handle_missing=self.config['dataset']['handle_missing_values'],
        )
        
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
        dataset = RadioLocalizationDataset(
            zarr_path=self.config['dataset']['zarr_path'],
            split='test',
            map_resolution=self.config['dataset']['map_resolution'],
            scene_extent=self.config['dataset']['scene_extent'],
            normalize=self.config['dataset']['normalize_features'],
            handle_missing=self.config['dataset']['handle_missing_values'],
        )
        
        return DataLoader(
            dataset,
            batch_size=self.config['training']['batch_size'],
            shuffle=False,
            num_workers=self.config['infrastructure']['num_workers'],
            collate_fn=collate_fn,
            pin_memory=True,
        )
