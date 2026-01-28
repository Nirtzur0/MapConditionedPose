import torch
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import yaml
from torch.utils.data import DataLoader
import sys
import os

# Add project root to path
sys.path.append(os.getcwd())

from src.training import UELocalizationLightning
from src.datasets.lmdb_dataset import LMDBRadioLocalizationDataset as RadioDataset

def load_model_and_data(checkpoint_path, data_path, num_tx=None):
    """Load the model and a sample batch."""
    print(f"Loading checkpoint: {checkpoint_path}")
    
    # Load model
    model = UELocalizationLightning.load_from_checkpoint(checkpoint_path)
    model.eval()
    model.freeze()
    
    # Load config to get dataset params
    with open(model.hparams.config_path, 'r') as f:
        config = yaml.safe_load(f)
        
    print(f"Loading dataset: {data_path}")
    # Extract params from config or defaults
    ds_config = config.get('dataset', {})
    dataset = RadioDataset(
        lmdb_path=data_path,
        split='all', # Force using all data to avoid empty split issues
        map_resolution=ds_config.get('map_resolution', 1.0),
        scene_extent=ds_config.get('scene_extent', 512),
        normalize=True
    )
    
    print(f"Dataset length: {len(dataset)}")
    if len(dataset) == 0:
        print("Warning: Dataset is empty!")
        return model, []

    # Create loader
    loader = DataLoader(
        dataset, 
        batch_size=32, 
        shuffle=False, 
        num_workers=4
    )
    
    return model, loader

def move_to_device(data, device):
    """Recursively move tensors to device."""
    if isinstance(data, torch.Tensor):
        return data.to(device)
    elif isinstance(data, dict):
        return {k: move_to_device(v, device) for k, v in data.items()}
    elif isinstance(data, list):
        return [move_to_device(v, device) for v in data]
    return data

def get_sample_batch(model, loader):
    """Get the first batch for visualization."""
    print("Loading first batch...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    for batch in loader:
        # Move batch to device
        batch = move_to_device(batch, device)
        return batch
    
    return None

def render_figures(model, batch, outputs, output_dir):
    """Render the requested figures."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    idx = 0 # Select first sample in batch for now
    
    # 1. Visualize Map Context (Input)
    # Radio Map: usually channel 0 is Path Gain
    radio_map = batch['radio_maps'][idx].cpu().numpy() # [C, H, W]
    path_gain = radio_map[0]
    
    # OSM Map
    osm_map = batch['osm_maps'][idx].cpu().numpy() # [C, H, W] 
    # Usually: 0=Height, 2=Footprint, 3=Roads (based on previous diff)
    buildings = osm_map[2]
    
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.imshow(path_gain, cmap='inferno')
    plt.title("Radio Map (Path Gain)")
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(buildings, cmap='gray_r')
    plt.title("Building Footprints (Context)")
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(output_dir / "fig1_environment.png", dpi=300, bbox_inches='tight')
    print(f"Saved {output_dir / 'fig1_environment.png'}")
    plt.close()
    
    # TODO: Add Top-K hypothesis overlay figure here if needed.
    return

def main():
    checkpoint_path = "checkpoints/trial_0/best_model-v2.ckpt"
    data_path = "data/processed/sionna_dataset/dataset_20251231_143438.lmdb"
    
    try:
        model, loader = load_model_and_data(checkpoint_path, data_path)
    except Exception as e:
        print(f"Failed to load: {e}")
        import glob
        ckpts = glob.glob("checkpoints/**/*.ckpt", recursive=True)
        if ckpts:
            print(f"Found alternative checkpoint: {ckpts[0]}")
            model, loader = load_model_and_data(ckpts[0], data_path)
        else:
            raise FileNotFoundError("No checkpoints found.")

    batch = get_sample_batch(model, loader)
    
    if batch is None:
        print("No data available!")
        return
        
    # Run validation step to get all outputs including Top-K hypothesis params
    print("Running validation step to compute Top-K hypotheses...")
    with torch.no_grad():
        device = model.device
        batch = move_to_device(batch, device)
        _ = model.validation_step(batch, 0)
        
        # Access the internal sample storage (set during validation_step)
        if not hasattr(model, '_last_val_sample') or model._last_val_sample is None:
            print("Error: Model did not store validation sample!")
            return
            
        step_out = model._last_val_sample
        
    # Generate figures
    idx = 0  # First sample in batch
    
    # Get dimensions
    h, w = batch['radio_map'].shape[-2:]
    
    # Figure 1: Environment Context
    plt.figure(figsize=(12, 5))
    
    radio = batch['radio_map'][idx, 0].cpu().numpy()  # Path gain channel
    plt.subplot(1, 2, 1)
    plt.imshow(radio, cmap='turbo')
    plt.colorbar(label='Path Gain (dB)', fraction=0.046)
    plt.title("Radio Map (Signal Propagation)")
    plt.axis('off')
    
    osm_map = batch['osm_map'][idx].cpu().numpy()
    buildings = osm_map[2]  # Building footprint channel
    plt.subplot(1, 2, 2)
    plt.imshow(buildings, cmap='binary')
    plt.title("Semantic Map (Buildings)")
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig("docs/paper/figures/environment_context.png", dpi=300, bbox_inches='tight')
    print("Saved docs/paper/figures/environment_context.png")
    plt.close()
    
    # Figure 2: Top-K Hypotheses Overlay
    grid_size = getattr(model.model, "grid_size", 32)
    top_k_indices = step_out['top_k_indices'].cpu().numpy()
    top_k_probs = step_out['top_k_probs'].cpu().numpy()
    fine_offsets = step_out['fine_offsets'].cpu().numpy()
    fine_scores = step_out['fine_scores'].cpu().numpy()
    weights = step_out.get('hypothesis_weights')
    if weights is None:
        logits = np.log(np.clip(top_k_probs, 1e-8, None)) + fine_scores
        weights = np.exp(logits - logits.max())
        weights = weights / (weights.sum() + 1e-8)
    else:
        weights = weights.cpu().numpy()

    cols = (top_k_indices % grid_size).astype(float)
    rows = (top_k_indices // grid_size).astype(float)
    centers = np.stack([(cols + 0.5) / grid_size, (rows + 0.5) / grid_size], axis=-1)
    candidates = centers + fine_offsets
    
    plt.figure(figsize=(8, 8))
    
    # Base: Building Footprints (faded)
    plt.imshow(buildings, cmap='gray_r', alpha=0.3)
    
    # Overlay: Top-K hypotheses (color/size by weight)
    cand_px = candidates[:, 0] * (w - 1)
    cand_py = candidates[:, 1] * (h - 1)
    sizes = 50 + 300 * (weights / (weights.max() + 1e-8))
    sc = plt.scatter(
        cand_px,
        cand_py,
        c=weights,
        s=sizes,
        cmap='viridis',
        alpha=0.9,
        edgecolors='none',
        label='Top-K hypotheses',
        zorder=8,
    )
    
    # True Position
    true_pos = batch['position'][idx].cpu().numpy()
    px, py = true_pos[0] * (w-1), true_pos[1] * (h-1)
    plt.scatter([px], [py], c='red', marker='*', s=300, edgecolors='white', linewidths=2, label='Ground Truth', zorder=10)
    
    # Predicted Position (soft mean)
    pred_pos = step_out['pred_pos'].cpu().numpy()
    ppx, ppy = pred_pos[0] * (w-1), pred_pos[1] * (h-1)
    plt.scatter([ppx], [ppy], c='tab:red', marker='x', s=150, linewidths=2, label='Prediction', zorder=10)
    
    plt.colorbar(sc, label='Hypothesis weight', fraction=0.046)
    plt.legend(loc='upper right', fontsize=12)
    plt.title("Top-K Hypothesis Overlay", fontsize=14)
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig("docs/paper/figures/posterior_density.png", dpi=300, bbox_inches='tight')
    print("Saved docs/paper/figures/posterior_density.png")
    plt.close()
    
    print("\n✓ Successfully generated paper figures!")

if __name__ == "__main__":
    main()
