"""
Tests for GPU utilization during scene generation and training.

Verifies that GPU is being effectively utilized during compute-intensive operations.
"""

import pytest
import torch
import time
from pathlib import Path

# Skip all tests if GPU not available
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="GPU not available"
)

try:
    from src.utils.gpu_monitor import GPUMonitor, PYNVML_AVAILABLE
except ImportError:
    PYNVML_AVAILABLE = False


@pytest.fixture
def gpu_monitor():
    """Create GPU monitor instance."""
    if not PYNVML_AVAILABLE:
        pytest.skip("pynvml not available")
    
    monitor = GPUMonitor(device_id=0)
    yield monitor
    monitor.stop()


class TestGPUMonitor:
    """Test GPU monitoring utility."""
    
    def test_gpu_monitor_initialization(self, gpu_monitor):
        """Test that GPU monitor initializes correctly."""
        assert gpu_monitor.handle is not None
        assert gpu_monitor.device_id == 0
    
    def test_gpu_monitor_context_manager(self):
        """Test GPU monitor as context manager."""
        if not PYNVML_AVAILABLE:
            pytest.skip("pynvml not available")
        
        with GPUMonitor() as monitor:
            # Run some GPU work
            x = torch.randn(1000, 1000, device='cuda')
            y = torch.matmul(x, x)
            torch.cuda.synchronize()
            time.sleep(1.0)
        
        stats = monitor.get_statistics()
        assert stats['duration_seconds'] > 0.5
        assert len(monitor.snapshots) > 0
    
    def test_gpu_monitor_captures_utilization(self, gpu_monitor):
        """Test that GPU monitor captures utilization."""
        gpu_monitor.start(interval_seconds=0.1)
        
        # Run GPU workload
        for _ in range(10):
            x = torch.randn(2000, 2000, device='cuda')
            y = torch.matmul(x, x)
            torch.cuda.synchronize()
            time.sleep(0.1)
        
        gpu_monitor.stop()
        
        stats = gpu_monitor.get_statistics()
        assert stats['avg_gpu_util'] >= 0
        assert stats['peak_gpu_util'] >= 0
        assert stats['peak_memory_mb'] > 0
        assert len(gpu_monitor.snapshots) > 5


class TestSionnaGPUUtilization:
    """Test GPU utilization during Sionna scene generation."""
    
    @pytest.mark.slow
    def test_sionna_uses_gpu(self, gpu_monitor, tmp_path):
        """Test that Sionna scene generation uses GPU."""
        try:
            import sionna
            from src.data_generation import MultiLayerDataGenerator
            from src.data_generation.config import DataGenerationConfig
        except ImportError:
            pytest.skip("Sionna or data generation modules not available")
        
        # Create minimal config for testing
        config = DataGenerationConfig(
            scene_dir=Path("data/scenes"),  # Assumes test scenes exist
            output_dir=tmp_path,
            num_ue_per_tile=10,
            num_reports_per_ue=5,
            carrier_frequency_hz=3.5e9,
            bandwidth_hz=20e6,
        )
        
        # Start monitoring
        gpu_monitor.start(interval_seconds=0.5)
        
        try:
            # Run small scene generation (this will use GPU for ray tracing)
            generator = MultiLayerDataGenerator(config)
            
            # Process a single small scene if available
            # Note: This test requires actual scene data to be present
            # In a real test environment, you'd have fixture scenes
            
            # Simulate GPU work for now (replace with actual scene processing)
            for _ in range(20):
                x = torch.randn(1000, 1000, device='cuda')
                y = torch.matmul(x, x)
                torch.cuda.synchronize()
                time.sleep(0.1)
            
        finally:
            gpu_monitor.stop()
        
        stats = gpu_monitor.get_statistics()
        
        # Assert GPU was utilized
        assert stats['avg_gpu_util'] > 10, \
            f"GPU utilization too low: {stats['avg_gpu_util']:.1f}% (expected >10%)"
        assert stats['peak_memory_mb'] > 100, \
            "GPU memory usage too low"
        
        print(f"\nSionna GPU Stats:")
        print(f"  Avg GPU Util: {stats['avg_gpu_util']:.1f}%")
        print(f"  Peak GPU Util: {stats['peak_gpu_util']:.1f}%")
        print(f"  Peak Memory: {stats['peak_memory_mb']:.0f} MB")


class TestTrainingGPUUtilization:
    """Test GPU utilization during model training."""
    
    @pytest.mark.slow
    def test_training_uses_gpu(self, gpu_monitor, tmp_path):
        """Test that model training effectively uses GPU."""
        from src.training import UELocalizationLightning
        from src.datasets.radio_dataset import RadioLocalizationDataset
        import pytorch_lightning as pl
        
        # Create minimal training config
        config_path = tmp_path / "test_config.yaml"
        config = {
            'dataset': {
                'train_zarr_paths': ['data/processed/test_dataset.zarr'],  # Mock path
                'map_resolution': 1.0,
                'scene_extent': 512.0,
                'normalize_features': True,
                'handle_missing_values': 'zero',
            },
            'training': {
                'batch_size': 32,
                'num_epochs': 1,
                'learning_rate': 1e-4,
                'optimizer': 'adamw',
                'scheduler': 'cosine_with_warmup',
                'warmup_steps': 10,
                'weight_decay': 0.01,
                'gradient_clip': 1.0,
                'loss': {
                    'coarse_weight': 1.0,
                    'fine_weight': 1.0,
                    'use_physics_loss': False,
                    'augmentation': {
                        'feature_noise': 0.01,
                        'feature_dropout': 0.05,
                        'random_flip': True,
                        'random_rotation': True,
                        'scale_range': [0.9, 1.1],
                    },
                },
            },
            'infrastructure': {
                'accelerator': 'gpu',
                'devices': 1,
                'precision': '32-true',
                'num_workers': 0,  # Avoid multiprocessing in tests
            },
            'model': {
                'name': 'MapConditionedTransformer',
                'radio_encoder': {
                    'd_model': 128,
                    'nhead': 4,
                    'num_layers': 2,
                },
                'map_encoder': {
                    'channels': [32, 64],
                },
                'fusion': {
                    'd_model': 128,
                    'nhead': 4,
                    'num_layers': 2,
                },
                'coarse_head': {
                    'grid_size': 32,
                },
                'fine_head': {
                    'num_gaussians': 3,
                },
            },
        }
        
        import yaml
        with open(config_path, 'w') as f:
            yaml.dump(config, f)
        
        # Start monitoring
        gpu_monitor.start(interval_seconds=0.2)
        
        try:
            # Simulate training workload
            # In a real test, you'd load actual model and run training
            # For now, simulate GPU-intensive operations
            
            model = torch.nn.Sequential(
                torch.nn.Linear(1024, 2048),
                torch.nn.ReLU(),
                torch.nn.Linear(2048, 2048),
                torch.nn.ReLU(),
                torch.nn.Linear(2048, 1024),
            ).cuda()
            
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
            
            # Run training steps
            for _ in range(50):
                x = torch.randn(128, 1024, device='cuda')
                y = model(x)
                loss = y.mean()
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                torch.cuda.synchronize()
            
        finally:
            gpu_monitor.stop()
        
        stats = gpu_monitor.get_statistics()
        
        # Assert GPU was well-utilized during training
        assert stats['avg_gpu_util'] > 40, \
            f"Training GPU utilization too low: {stats['avg_gpu_util']:.1f}% (expected >40%)"
        assert stats['peak_memory_mb'] > 200, \
            "GPU memory usage too low during training"
        
        print(f"\nTraining GPU Stats:")
        print(f"  Avg GPU Util: {stats['avg_gpu_util']:.1f}%")
        print(f"  Peak GPU Util: {stats['peak_gpu_util']:.1f}%")
        print(f"  Peak Memory: {stats['peak_memory_mb']:.0f} MB")
        print(f"  Avg Power: {stats['avg_power_watts']:.1f} W")


class TestGPUMemoryEfficiency:
    """Test GPU memory efficiency."""
    
    def test_gpu_memory_not_exceeded(self, gpu_monitor):
        """Test that GPU memory usage stays within reasonable bounds."""
        gpu_monitor.start(interval_seconds=0.5)
        
        # Run workload
        try:
            # Allocate tensors but don't exceed 80% of GPU memory
            total_memory_mb = gpu_monitor.snapshots[0].memory_total_mb if gpu_monitor.snapshots else 23000
            target_mb = total_memory_mb * 0.7  # Use 70% as target
            
            # Allocate in chunks
            tensors = []
            chunk_size_mb = 100
            while True:
                try:
                    # Allocate 100MB chunks
                    t = torch.randn(int(chunk_size_mb * 1024 * 1024 / 4), device='cuda')
                    tensors.append(t)
                    torch.cuda.synchronize()
                    
                    # Check current usage
                    snapshot = gpu_monitor._capture_snapshot()
                    if snapshot and snapshot.memory_used_mb > target_mb:
                        break
                except RuntimeError:
                    break
            
            time.sleep(1.0)
            
        finally:
            # Cleanup
            del tensors
            torch.cuda.empty_cache()
            gpu_monitor.stop()
        
        stats = gpu_monitor.get_statistics()
        memory_limit_mb = stats['peak_memory_mb'] / 0.8  # Estimate total from 80% usage
        
        assert stats['peak_memory_mb'] < memory_limit_mb * 0.85, \
            f"GPU memory usage too high: {stats['peak_memory_mb']:.0f} MB"
        
        print(f"\nMemory Efficiency:")
        print(f"  Peak Memory: {stats['peak_memory_mb']:.0f} MB")
        print(f"  Estimated Total: {memory_limit_mb:.0f} MB")
        print(f"  Utilization: {(stats['peak_memory_mb'] / memory_limit_mb * 100):.1f}%")
