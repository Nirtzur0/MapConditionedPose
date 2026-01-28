"""
End-to-End System Integration Tests
Validates the entire pipeline from Scene Generation (M1) to Physics Loss (M4).
"""

import pytest
import torch
import numpy as np
import tempfile
import shutil
import json
from pathlib import Path

# M1: Scene Generation
from src.scene_generation import SceneGenerator, SitePlacer, MaterialRandomizer

# M2: Data Generation
from src.data_generation.multi_layer_generator import (
    MultiLayerDataGenerator, DataGenerationConfig
)
from src.data_generation.features import (
    RTLayerFeatures, PHYFAPILayerFeatures, MACRRCLayerFeatures
)

# M3: Model
from src.models.ue_localization_model import UELocalizationModel

# M4: Physics Loss
from src.physics_loss import compute_physics_loss, PhysicsLossConfig


class TestSystemIntegration:
    """
    End-to-End Pipeline Tests.
    Ensures that components M1->M2->M3->M4 flow data correctly.
    """

    @pytest.fixture
    def temp_workspace(self):
        """Create a temporary workspace for scene and data files."""
        temp_dir = Path(tempfile.mkdtemp())
        yield temp_dir
        shutil.rmtree(temp_dir)

    def test_e2e_pipeline_flow(self, temp_workspace):
        """
        Run a complete minimal pipeline:
        1. Generate Scene (M1)
        2. Generate Data (M2) - Mocked Ray Tracing
        3. Initialize Model (M3)
        4. Forward Pass & Physics Loss (M4)
        """
        # =========================================================================
        # Step 1: M1 Scene Generation
        # =========================================================================
        scene_dir = temp_workspace / "scene_test"
        scene_dir.mkdir()
        
        # Create minimal metadata
        site_placer = SitePlacer(strategy="grid")
        sites = site_placer.place(
            bounds=(-100, -100, 100, 100),
            num_tx=2,
            num_rx=5
        )
        
        metadata = {
            'scene_id': 'integration_test_001',
            'bbox': {'x_min': -100, 'x_max': 100, 'y_min': -100, 'y_max': 100},
            'sites': [s.to_dict() for s in sites],
            'num_tx': 2,
            'num_rx': 5
        }
        
        metadata_path = scene_dir / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f)
            
        # Create dummy scene.xml (required by generator)
        scene_xml_path = scene_dir / "scene.xml"
        scene_xml_path.write_text("<scene version='2.0.0'/>")
            
        print(f"\n[M1] Scene generated at {scene_dir}")
        assert metadata_path.exists()

        # =========================================================================
        # Step 2: M2 Data Generation (Real Simulation)
        # =========================================================================
        # Check if Sionna is available - Requirement: No Mock Data
        try:
            import sionna
            SIONNA_AVAILABLE = True
        except ImportError:
            SIONNA_AVAILABLE = False
            
        if not SIONNA_AVAILABLE:
            pytest.fail("Sionna not found. User requirement 'no mock data' requires Sionna for real simulation.")

        data_config = DataGenerationConfig(
            scene_dir=temp_workspace, # Generator looks for scenes inside this dir
            scene_metadata_path=metadata_path,
            carrier_frequency_hz=3.5e9,
            bandwidth_hz=100e6,
            num_ue_per_tile=2,     # Minimal
            num_reports_per_ue=2,  # Minimal for speed
            ue_height_range=(1.5, 1.5),
            ue_velocity_range=(0.0, 0.0),
            report_interval_ms=100.0,
            output_dir=temp_workspace / "dataset",
            enable_beam_management=False, # Disable for speed in test
            num_beams=8,
            max_neighbors=2,
            tx_power_dbm=23.0,
            noise_figure_db=9.0
        )
        
        generator = MultiLayerDataGenerator(data_config)
        
        # Run Real Generation
        # This will run ray tracing if Sionna is installed
        output_path = generator.generate_dataset(create_splits=False)
        
        assert output_path.exists()
        print(f"[M2] Real dataset generated at {output_path}")
        
        # Load one sample from LMDB for M3
        from src.datasets.lmdb_dataset import LMDBRadioLocalizationDataset

        dataset = LMDBRadioLocalizationDataset(
            lmdb_path=str(output_path),
            split='all',
            map_resolution=1.0,
            scene_extent=512,
            normalize=False,
            handle_missing='mask',
        )
        sample = dataset[0]

        measurements = {
            k: v.unsqueeze(0)
            for k, v in sample['measurements'].items()
        }
        radio_map = sample['radio_map'].unsqueeze(0)
        osm_map = sample['osm_map'].unsqueeze(0)

        batch_size = 1
        seq_len = measurements['cell_ids'].shape[1]
        print("[M2] Data loaded from LMDB successfully")

        # =========================================================================
        # Step 3: M3 Model Initialization & Forward
        # =========================================================================
        # Create minimal config
        num_cells = int(measurements['cell_ids'].max().item()) + 1
        num_beams = int(measurements['beam_ids'].max().item()) + 1
        rt_dim = measurements['rt_features'].shape[-1]
        phy_dim = measurements['phy_features'].shape[-1]
        mac_dim = measurements['mac_features'].shape[-1]
        radio_channels = radio_map.shape[1]
        osm_channels = osm_map.shape[1]
        img_size = radio_map.shape[-1]

        model_config = {
            'dataset': {
                'map_size': img_size,
            },
            'model': {
                'radio_encoder': {
                    'num_cells': max(num_cells, 1),
                    'num_beams': max(num_beams, 1),
                    'd_model': 32,
                    'nhead': 2,
                    'num_layers': 2,
                    'dropout': 0.1,
                    'max_seq_len': seq_len,
                    'rt_features_dim': rt_dim,
                    'phy_features_dim': phy_dim,
                    'mac_features_dim': mac_dim,
                },
                'map_encoder': {
                    'img_size': img_size,
                    'patch_size': 16,
                    'in_channels': radio_channels + osm_channels,
                    'd_model': 32,
                    'nhead': 2,
                    'num_layers': 1,
                    'dropout': 0.1,
                    'radio_map_channels': radio_channels,
                    'osm_map_channels': osm_channels,
                },
                'fusion': {
                    'd_fusion': 32,
                    'nhead': 2,
                    'dropout': 0.1,
                },
                'coarse_head': {
                    'grid_size': 16,
                    'dropout': 0.1,
                },
                'fine_head': {
                    'd_hidden': 32,
                    'top_k': 4,
                    'dropout': 0.1,
                }
            }
        }
        
        model = UELocalizationModel(model_config)
        
        # Prepare Batch Tensors (Real Data)
        # Use the tensors extracted in Step 2
        # batch_size and seq_len defined above (seq_len=1)
        
        batch_measurements = measurements
    
        # Forward Pass
        outputs = model(
            measurements=batch_measurements,
            radio_map=radio_map,
            osm_map=osm_map
        )
        
        assert 'predicted_position' in outputs
        assert outputs['predicted_position'].shape == (batch_size, 2)
        print("[M3] Model forward pass successful")
        
        # =========================================================================
        # Step 4: M4 Physics Loss
        # =========================================================================
        # Use real RT features for observation (subset)
        # RT features are [Batch, SeqLen, 10]
        # Physics loss typically expects [Batch, Features] (last step?) or sequence?
        # Let's use the last time step
        # Match radio_maps channels (5)
        observed_features = batch_measurements['rt_features'][:, 0, :radio_map.shape[1]]
        
        # Physics map: In real pipeline, this comes from radio_map or separate physics map
        # We reuse the radio_map (assuming channels match or we slice)
        # Physics loss usually needs specific channels.
        # If radio_map has 5 channels and we need 7, we might crash.
        # Let's check typical physics loss config.
        # For this test, we slice or pad the real map to fit if needed, 
        # OR just use the real map and hope it works (as "No Bandages").
        # If it fails, that's a "Real Issue" to fix.
        physics_map = radio_map
        
        # However, PhysicsLoss might fail if channels mismatch.
        # Let's try running it. If it fails, we fix the config or map generation.
        
        try:
            loss = compute_physics_loss(
                predicted_xy=outputs['predicted_position'],
                observed_features=observed_features,
                radio_maps=physics_map
            )
            print(f"[M4] Physics loss computed: {loss.item()}")
            assert loss.item() >= 0
        except Exception as e:
            print(f"[M4] Physics loss failed (Expected if map channels mismatch?): {e}")
            # If strictly "no bandages", we should have fixed correct map generation.
            # But we are in Step 1 of "Run and Identify Failures".
            # So failing here is acceptable information gathering.
            raise e
