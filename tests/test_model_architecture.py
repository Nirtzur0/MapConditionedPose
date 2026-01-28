"""
Tests for M3: Transformer Model Components

Tests all model components and integration.
"""

import pytest
import torch
import yaml
from pathlib import Path
import logging

# Model components
from src.models.radio_encoder import RadioEncoder, PositionalEncoding
from src.models.map_encoder import StandardMapEncoder
from src.models.fusion import CrossAttentionFusion
from src.models.heads import CoarseHead, FineHead
from src.models.ue_localization_model import UELocalizationModel

logger = logging.getLogger(__name__)


@pytest.fixture
def config():
    """Load model configuration."""
    config_path = Path(__file__).parent.parent / 'configs' / 'model.yaml'
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


@pytest.fixture
def batch_measurements(config):
    """Create dummy measurement batch."""
    batch_size = 4
    seq_len = 10
    radio_cfg = config['model']['radio_encoder']

    return {
        'rt_features': torch.randn(batch_size, seq_len, radio_cfg['rt_features_dim']),
        'phy_features': torch.randn(batch_size, seq_len, radio_cfg['phy_features_dim']),
        'mac_features': torch.randn(batch_size, seq_len, radio_cfg['mac_features_dim']),
        'cell_ids': torch.randint(0, radio_cfg['num_cells'], (batch_size, seq_len)),
        'beam_ids': torch.randint(0, radio_cfg['num_beams'], (batch_size, seq_len)),
        'timestamps': torch.rand(batch_size, seq_len) * 5,  # 0-5 seconds
        'mask': torch.ones(batch_size, seq_len, dtype=torch.bool),
    }


@pytest.fixture
def batch_maps():
    """Create dummy map batch."""
    batch_size = 4
    H = W = 256
    
    return {
        'radio_map': torch.randn(batch_size, 5, H, W),
        'osm_map': torch.randn(batch_size, 5, H, W),
    }


class TestRadioEncoder:
    """Test RadioEncoder component."""
    
    def test_initialization(self, config):
        """Test encoder initialization."""
        cfg = config['model']['radio_encoder']
        encoder = RadioEncoder(
            num_cells=cfg['num_cells'],
            num_beams=cfg['num_beams'],
            d_model=cfg['d_model'],
            nhead=cfg['nhead'],
            num_layers=cfg['num_layers'],
        )
        
        assert encoder.d_model == cfg['d_model']
        assert encoder.num_cells == cfg['num_cells']
    
    @pytest.mark.parametrize('batch_size', [1, 4, 16])
    def test_forward_pass(self, config, batch_measurements, batch_size):
        """Test forward pass."""
        cfg = config['model']['radio_encoder']
        encoder = RadioEncoder(
            num_cells=cfg['num_cells'],
            num_beams=cfg['num_beams'],
            d_model=cfg['d_model'],
            nhead=cfg['nhead'],
            num_layers=cfg['num_layers'],
            rt_features_dim=cfg['rt_features_dim'],
            phy_features_dim=cfg['phy_features_dim'],
            mac_features_dim=cfg['mac_features_dim'],
        )
        
        # Adjust batch size of measurements
        current_bs = batch_measurements['cell_ids'].shape[0]
        if batch_size > current_bs:
            # Repeat/Tile to match size
            repeats = (batch_size + current_bs - 1) // current_bs
            measurements = {
                k: v.repeat(repeats, *([1]*(v.ndim-1)))[:batch_size] 
                if isinstance(v, torch.Tensor) 
                else v # Handle non-tensors (mask is tensor)
                for k, v in batch_measurements.items()
            }
        else:
            measurements = {k: v[:batch_size] if isinstance(v, torch.Tensor) else v 
                        for k, v in batch_measurements.items()}
        
        output = encoder(measurements)
        
        # Check output shape
        assert output.shape == (batch_size, cfg['d_model'])
    
    @pytest.mark.parametrize('mask_ratio', [0.0, 0.5, 0.9])
    def test_masked_sequence(self, config, batch_measurements, mask_ratio):
        """Test with masked (padded) sequences."""
        cfg = config['model']['radio_encoder']
        encoder = RadioEncoder(
            num_cells=cfg['num_cells'],
            num_beams=cfg['num_beams'],
            d_model=cfg['d_model'],
            nhead=cfg['nhead'],
            num_layers=2,  # Fewer layers for speed
            rt_features_dim=cfg['rt_features_dim'],
            phy_features_dim=cfg['phy_features_dim'],
            mac_features_dim=cfg['mac_features_dim'],
        )
        
        # Create mask
        seq_len = batch_measurements['mask'].shape[1]
        valid_len = int(seq_len * (1 - mask_ratio))
        valid_len = max(1, valid_len)  # At least one valid
        
        batch_measurements['mask'][:] = True
        batch_measurements['mask'][:, valid_len:] = False
        
        output = encoder(batch_measurements)
        
        # Should still produce valid output
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()
    
    def test_positional_encoding(self):
        """Test PositionalEncoding module."""
        pe = PositionalEncoding(d_model=128, max_len=100)
        
        timestamps = torch.rand(4, 10) * 5  # [batch, seq_len]
        encoded = pe(timestamps)
        
        assert encoded.shape == (4, 10, 128)


class TestCrossAttentionFusion:
    """Test CrossAttentionFusion module."""
    
    def test_forward_pass(self, config):
        """Test fusion forward pass."""
        cfg_radio = config['model']['radio_encoder']
        cfg_map = config['model']['map_encoder']
        cfg_fusion = config['model']['fusion']
        
        fusion = CrossAttentionFusion(
            d_radio=cfg_radio['d_model'],
            d_map=cfg_map['d_model'],
            d_fusion=cfg_fusion['d_fusion'],
            nhead=cfg_fusion['nhead'],
        )
        
        batch_size = 4
        radio_emb = torch.randn(batch_size, cfg_radio['d_model'])
        map_tokens = torch.randn(batch_size, 1024, cfg_map['d_model'])
        
        fused = fusion(radio_emb, map_tokens)
        
        assert fused.shape == (batch_size, cfg_fusion['d_fusion'])
    
    def test_attention_weights(self, config):
        """Test attention weights extraction."""
        cfg_fusion = config['model']['fusion']
        
        fusion = CrossAttentionFusion(
            d_radio=512,
            d_map=768,
            d_fusion=cfg_fusion['d_fusion'],
            nhead=cfg_fusion['nhead'],
        )
        
        batch_size = 2
        radio_emb = torch.randn(batch_size, 512)
        map_tokens = torch.randn(batch_size, 100, 768)
        
        fused, attn_weights = fusion.forward_with_attention(radio_emb, map_tokens)
        
        assert fused.shape == (batch_size, cfg_fusion['d_fusion'])
        assert attn_weights.shape == (batch_size, 1, 100)


class TestPredictionHeads:
    """Test CoarseHead and FineHead."""
    
    def test_coarse_head(self, config):
        """Test coarse prediction head."""
        cfg = config['model']['coarse_head']
        
        coarse_head = CoarseHead(
            d_input=768,
            grid_size=cfg['grid_size'],
        )
        
        batch_size = 4
        fused = torch.randn(batch_size, 768)
        
        logits, heatmap = coarse_head(fused)
        
        assert logits.shape == (batch_size, cfg['grid_size'] ** 2)
        assert heatmap.shape == (batch_size, cfg['grid_size'], cfg['grid_size'])
        
        # Check heatmap is valid probability distribution
        assert torch.allclose(heatmap.sum(dim=(1, 2)), torch.ones(batch_size))
    
    def test_top_k_cells(self, config):
        """Test top-K cell selection."""
        cfg = config['model']['coarse_head']
        
        coarse_head = CoarseHead(d_input=768, grid_size=cfg['grid_size'])
        
        heatmap = torch.rand(2, cfg['grid_size'], cfg['grid_size'])
        heatmap = heatmap / heatmap.sum(dim=(1, 2), keepdim=True)
        
        top_k_indices, top_k_probs = coarse_head.get_top_k_cells(heatmap, k=5)
        
        assert top_k_indices.shape == (2, 5)
        assert top_k_probs.shape == (2, 5)
        
        # Check probabilities are sorted descending
        assert (top_k_probs[:, :-1] >= top_k_probs[:, 1:]).all()
    
    def test_fine_head(self, config):
        """Test fine prediction head."""
        cfg = config['model']['fine_head']
        
        fine_head = FineHead(
            d_input=768,
            d_hidden=cfg['d_hidden'],
            top_k=cfg['top_k'],
        )
        
        batch_size = 4
        fused = torch.randn(batch_size, 768)
        grid_size = config['model']['coarse_head']['grid_size']
        top_k_indices = torch.randint(0, grid_size ** 2, (batch_size, cfg['top_k']))
        
        cell_size = 1.0 / config['model']['coarse_head']['grid_size']
        offsets, scores = fine_head(fused, top_k_indices, cell_size=cell_size)
        
        assert offsets.shape == (batch_size, cfg['top_k'], 2)
        assert scores.shape == (batch_size, cfg['top_k'])
        assert torch.isfinite(scores).all()
    
    def test_rerank_weights(self, config):
        """Test re-ranking weights computation."""
        cfg = config['model']['fine_head']
        
        fine_head = FineHead(d_input=768, top_k=cfg['top_k'])
        
        batch_size = 4
        fused = torch.randn(batch_size, 768)
        top_k_indices = torch.randint(0, 1024, (batch_size, cfg['top_k']))
        _, scores = fine_head(fused, top_k_indices, cell_size=1.0 / 32)
        top_k_probs = torch.rand(batch_size, cfg['top_k'])
        top_k_probs = top_k_probs / top_k_probs.sum(dim=1, keepdim=True)

        weights = torch.softmax(torch.log(top_k_probs + 1e-8) + scores, dim=-1)

        assert weights.shape == (batch_size, cfg['top_k'])
        assert torch.allclose(weights.sum(dim=-1), torch.ones(batch_size), atol=1e-5)


class TestUELocalizationModel:
    """Test full UE localization model."""
    
    def test_initialization(self, config):
        """Test model initialization."""
        model = UELocalizationModel(config)
        
        # Check all components exist
        assert hasattr(model, 'radio_encoder')
        assert hasattr(model, 'map_encoder')
        assert hasattr(model, 'fusion')
        assert hasattr(model, 'coarse_head')
        assert hasattr(model, 'fine_head')
        
        # Check parameter count
        num_params = model.num_parameters
        assert num_params > 0
        logger.info(f"\nModel has {num_params:,} parameters")
    
    def test_forward_pass(self, config, batch_measurements, batch_maps):
        """Test full forward pass."""
        model = UELocalizationModel(config)
        
        outputs = model(
            batch_measurements,
            batch_maps['radio_map'],
            batch_maps['osm_map'],
        )
        
        # Check all outputs present
        assert 'coarse_logits' in outputs
        assert 'coarse_heatmap' in outputs
        assert 'top_k_indices' in outputs
        assert 'top_k_probs' in outputs
        assert 'fine_offsets' in outputs
        assert 'fine_scores' in outputs
        assert 'hypothesis_weights' in outputs
        assert 'predicted_position' in outputs
        
        # Check shapes
        batch_size = batch_measurements['cell_ids'].shape[0]
        assert outputs['predicted_position'].shape == (batch_size, 2)
    
    def test_predict(self, config, batch_measurements, batch_maps):
        """Test simplified predict interface."""
        model = UELocalizationModel(config)
        
        position, weights = model.predict(
            batch_measurements,
            batch_maps['radio_map'],
            batch_maps['osm_map'],
            return_uncertainty=True,
        )
        
        batch_size = batch_measurements['cell_ids'].shape[0]
        assert position.shape == (batch_size, 2)
        assert weights.shape[0] == batch_size
        assert torch.allclose(weights.sum(dim=-1), torch.ones(batch_size), atol=1e-4)
    
    def test_compute_loss(self, config, batch_measurements, batch_maps):
        """Test loss computation."""
        model = UELocalizationModel(config)
        
        outputs = model(
            batch_measurements,
            batch_maps['radio_map'],
            batch_maps['osm_map'],
        )
        
        # Create dummy targets
        batch_size = batch_measurements['cell_ids'].shape[0]
        grid_size = config['model']['coarse_head']['grid_size']
        targets = {
            'position': torch.rand(batch_size, 2),  # Normalized coords [0, 1]
            'cell_grid': torch.randint(0, grid_size * grid_size, (batch_size,)),  # Random cells
        }
        
        loss_weights = {
            'coarse_weight': 1.0,
            'fine_weight': 1.0,
        }
        
        losses = model.compute_loss(outputs, targets, loss_weights)
        
        # Check losses
        assert 'loss' in losses
        assert 'coarse_loss' in losses
        assert 'fine_loss' in losses
        
        # All losses should be finite
        assert torch.isfinite(losses['loss'])
        assert torch.isfinite(losses['coarse_loss'])
        assert torch.isfinite(losses['fine_loss'])
    
    def test_attention_visualization(self, config, batch_measurements, batch_maps):
        """Test attention weights extraction for visualization."""
        model = UELocalizationModel(config)
        model.eval()
        
        outputs, attention_weights = model.forward_with_attention(
            batch_measurements,
            batch_maps['radio_map'],
            batch_maps['osm_map'],
        )
        
        batch_size = batch_measurements['cell_ids'].shape[0]
        num_patches = (config['model']['map_encoder']['img_size'] // 
                       config['model']['map_encoder']['patch_size']) ** 2
        
        assert attention_weights.shape == (batch_size, 1, num_patches)
        
        # Attention weights should sum to 1 (with tolerance for numerical precision)
        assert torch.allclose(
            attention_weights.sum(dim=-1),
            torch.ones(batch_size, 1),
            atol=1e-2,  # Allow small numerical errors
        )
    
    def test_gradient_flow(self, config, batch_measurements, batch_maps):
        """Test that gradients flow through the model."""
        model = UELocalizationModel(config)
        
        outputs = model(
            batch_measurements,
            batch_maps['radio_map'],
            batch_maps['osm_map'],
        )
        
        # Dummy loss
        loss = outputs['predicted_position'].sum()
        loss.backward()
        
        # Check that some parameters have gradients
        has_grad = any(
            p.grad is not None and not torch.all(p.grad == 0)
            for p in model.parameters()
        )
        assert has_grad, "No gradients computed"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
