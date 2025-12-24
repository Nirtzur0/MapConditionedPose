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
from src.models.map_encoder import MapEncoder, PatchEmbedding
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
def batch_measurements():
    """Create dummy measurement batch."""
    batch_size = 4
    seq_len = 10
    
    return {
        'rt_features': torch.randn(batch_size, seq_len, 8),
        'phy_features': torch.randn(batch_size, seq_len, 10),
        'mac_features': torch.randn(batch_size, seq_len, 6),
        'cell_ids': torch.randint(0, 512, (batch_size, seq_len)),
        'beam_ids': torch.randint(0, 64, (batch_size, seq_len)),
        'timestamps': torch.rand(batch_size, seq_len) * 5,  # 0-5 seconds
        'mask': torch.ones(batch_size, seq_len, dtype=torch.bool),
    }


@pytest.fixture
def batch_maps():
    """Create dummy map batch."""
    batch_size = 4
    H = W = 512
    
    return {
        'radio_map': torch.randn(batch_size, 5, H, W),
        'osm_map': torch.randn(batch_size, 4, H, W),
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
    
    def test_forward_pass(self, config, batch_measurements):
        """Test forward pass."""
        cfg = config['model']['radio_encoder']
        encoder = RadioEncoder(
            num_cells=cfg['num_cells'],
            num_beams=cfg['num_beams'],
            d_model=cfg['d_model'],
            nhead=cfg['nhead'],
            num_layers=cfg['num_layers'],
            rt_features_dim=8,
            phy_features_dim=10,
            mac_features_dim=6,
        )
        
        output = encoder(batch_measurements)
        
        # Check output shape
        batch_size = batch_measurements['cell_ids'].shape[0]
        assert output.shape == (batch_size, cfg['d_model'])
    
    def test_masked_sequence(self, config, batch_measurements):
        """Test with masked (padded) sequences."""
        cfg = config['model']['radio_encoder']
        encoder = RadioEncoder(
            num_cells=cfg['num_cells'],
            num_beams=cfg['num_beams'],
            d_model=cfg['d_model'],
            nhead=cfg['nhead'],
            num_layers=2,  # Fewer layers for speed
            rt_features_dim=8,
            phy_features_dim=10,
            mac_features_dim=6,
        )
        
        # Mask second half of sequence
        batch_measurements['mask'][:, 5:] = False
        
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


class TestMapEncoder:
    """Test MapEncoder component."""
    
    def test_initialization(self, config):
        """Test map encoder initialization."""
        cfg = config['model']['map_encoder']
        encoder = MapEncoder(
            img_size=cfg['img_size'],
            patch_size=cfg['patch_size'],
            in_channels=cfg['in_channels'],
            d_model=cfg['d_model'],
            nhead=cfg['nhead'],
            num_layers=cfg['num_layers'],
        )
        
        assert encoder.img_size == cfg['img_size']
        assert encoder.d_model == cfg['d_model']
        assert encoder.num_patches == (cfg['img_size'] // cfg['patch_size']) ** 2
    
    def test_forward_pass(self, config, batch_maps):
        """Test forward pass."""
        cfg = config['model']['map_encoder']
        encoder = MapEncoder(
            img_size=cfg['img_size'],
            patch_size=cfg['patch_size'],
            in_channels=cfg['in_channels'],
            d_model=cfg['d_model'],
            nhead=cfg['nhead'],
            num_layers=cfg['num_layers'],
        )
        
        spatial_tokens, cls_token = encoder(
            batch_maps['radio_map'],
            batch_maps['osm_map'],
        )
        
        # Check shapes
        batch_size = batch_maps['radio_map'].shape[0]
        num_patches = (cfg['img_size'] // cfg['patch_size']) ** 2
        
        assert spatial_tokens.shape == (batch_size, num_patches, cfg['d_model'])
        assert cls_token.shape == (batch_size, cfg['d_model'])
    
    def test_patch_embedding(self, config):
        """Test PatchEmbedding module."""
        patch_embed = PatchEmbedding(
            img_size=512,
            patch_size=16,
            in_channels=9,
            embed_dim=768,
        )
        
        x = torch.randn(2, 9, 512, 512)
        patches, grid_size = patch_embed(x)
        
        assert patches.shape == (2, 1024, 768)  # 32x32=1024 patches
        assert grid_size == 32


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
        top_k_indices = torch.randint(0, 1024, (batch_size, cfg['top_k']))
        
        offsets, uncertainties = fine_head(fused, top_k_indices)
        
        assert offsets.shape == (batch_size, cfg['top_k'], 2)
        assert uncertainties.shape == (batch_size, cfg['top_k'], 2)
        
        # Uncertainties should be positive
        assert (uncertainties > 0).all()
    
    def test_nll_loss(self, config):
        """Test NLL loss computation."""
        cfg = config['model']['fine_head']
        
        fine_head = FineHead(d_input=768, top_k=cfg['top_k'])
        
        batch_size = 4
        predicted_offsets = torch.randn(batch_size, cfg['top_k'], 2)
        predicted_uncert = torch.rand(batch_size, cfg['top_k'], 2) * 2 + 0.5
        true_offsets = torch.randn(batch_size, 2)
        top_k_probs = torch.rand(batch_size, cfg['top_k'])
        top_k_probs = top_k_probs / top_k_probs.sum(dim=1, keepdim=True)
        
        loss = fine_head.nll_loss(
            predicted_offsets,
            predicted_uncert,
            true_offsets,
            top_k_probs,
        )
        
        # Loss should be finite scalar
        assert loss.ndim == 0
        assert torch.isfinite(loss)


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
        assert 'fine_uncertainties' in outputs
        assert 'predicted_position' in outputs
        
        # Check shapes
        batch_size = batch_measurements['cell_ids'].shape[0]
        assert outputs['predicted_position'].shape == (batch_size, 2)
    
    def test_predict(self, config, batch_measurements, batch_maps):
        """Test simplified predict interface."""
        model = UELocalizationModel(config)
        
        position, uncertainty = model.predict(
            batch_measurements,
            batch_maps['radio_map'],
            batch_maps['osm_map'],
            return_uncertainty=True,
        )
        
        batch_size = batch_measurements['cell_ids'].shape[0]
        assert position.shape == (batch_size, 2)
        assert uncertainty.shape == (batch_size, 2)
        assert (uncertainty > 0).all()
    
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
        targets = {
            'position': torch.rand(batch_size, 2) * 512,  # Random positions
            'cell_grid': torch.randint(0, 32*32, (batch_size,)),  # Random cells
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
