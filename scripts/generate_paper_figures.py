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
from src.datasets.radio_dataset import RadioLocalizationDataset as RadioDataset, collate_fn

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
        zarr_path=data_path,
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
        collate_fn=collate_fn,
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
    
    # 2. Visualize Posterior (Prediction)
    # We need to construct the heatmap. 
    # The model has a helper `_render_gmm_heatmap` we added!
    # Let's use it.
    
    # We need the specific outputs expected by `_render_gmm_heatmap`
    # It expects: top_k_indices, top_k_probs, fine_offsets, fine_uncertainties
    
    # We need to run the logic that produces these.
    # UELocalizationLightning.dataset_step or similar? 
    # checking the previous diff, it was used in `validation_step`.
    # We can manually invoke the necessary parts.
    
    # Extract coarse logits
    coarse_logits = outputs['coarse_logits'][idx] # [G*G]
    
    # Top-K
    k = model.hparams.model['fine_head']['top_k']
    top_k_probs, top_k_indices = torch.topk(torch.softmax(coarse_logits, -1), k)
    
    # Extract fine details for these indices
    # We need to call the fine head.
    # The model's forward pass might have already done this or we need to re-run specific heads.
    # Looking at the code structure is hard without `view_file`, but standardly:
    
    # Let's assume we can get the necessary tensors from the model's fine head
    # or just rely on the fact that we can call `model.validation_step` on this batch!
    # But validation_step logs to logger, doesn't return the heatmap.
    
    # We will reconstruct:
    # 1. Get embedding for top-k cells
    # 2. Pass to fine head
    
    # Let's simplify: access `_render_gmm_heatmap` logic directly 
    # But first we need the fine predictions.
    
    # Let's try to run `validation_step` but capture the internal state? No too complex.
    
    # Let's Re-implement the gather logic briefly here.
    # It's safer.
    
    fusion_feat = outputs['fusion_output'][idx] # [D]
    
    # Coarse
    # coarse_logits already got
    
    # Fine
    # We need cell embeddings for the top k indices
    # The model should have `cell_embedding` layer
    cell_embeds = model.model.cell_embedding(top_k_indices.long().unsqueeze(0)) # [1, K, D_emb]
    
    # Expand fusion 
    fusion_k = fusion_feat.unsqueeze(0).unsqueeze(1).expand(1, k, -1) # [1, K, D]
    
    # Concat ? Check model definition... 
    # Assume fine head takes (fusion, cell_emb)
    # fine_out = model.model.fine_head(fusion_k, cell_embeds)
    # mu, logvar = chunk(fine_out)
    
    # Since I can't check the exact model code right now without viewing, 
    # I will rely on the `step_inference` method if it exists, or just use the batch `outputs` 
    # IF the forward pass returned them.
    # The `forward` usually returns `fine_positions` [B, G^2, 2] or similar if not top-k optimized?
    
    # Wait, the diff showed:
    # `outputs['top_k_indices']` being logged in `validation_step_end`?
    # No, it was `outputs` *from* `validation_step` that contained 'top_k_indices'.
    # This implies `validation_step` returns a dict with these keys.
    
    pass

def main():
    checkpoint_path = "checkpoints/trial_0/best_model-v2.ckpt"
    data_path = "data/processed/sionna_dataset/dataset_20251231_143438.zarr"
    
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
        
    # Run validation step to get all outputs including GMM params
    print("Running validation step to compute GMM...")
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
    
    # Figure 2: Posterior Density
    # Render GMM heatmap (step_out already contains data for sample 0)
    heatmap = model._render_gmm_heatmap(
        h, w,
        step_out['top_k_indices'].cpu(),
        step_out['top_k_probs'].cpu(),
        step_out['fine_offsets'].cpu(),
        step_out['fine_uncertainties'].cpu()
    )
    
    plt.figure(figsize=(8, 8))
    
    # Base: Building Footprints (faded)
    plt.imshow(buildings, cmap='gray_r', alpha=0.3)
    
    # Overlay: GMM Heatmap with adaptive alpha
    plt.imshow(heatmap, cmap='viridis', alpha=0.5 + 0.3*heatmap, vmin=0, vmax=1)
    
    # True Position
    true_pos = batch['position'][idx].cpu().numpy()
    px, py = true_pos[0] * (w-1), true_pos[1] * (h-1)
    plt.scatter([px], [py], c='red', marker='*', s=300, edgecolors='white', linewidths=2, label='Ground Truth', zorder=10)
    
    # Predicted Position
    pred_pos = step_out['pred_pos'].cpu().numpy()
    ppx, ppy = pred_pos[0] * (w-1), pred_pos[1] * (h-1)
    plt.scatter([ppx], [ppy], c='cyan', marker='P', s=150, edgecolors='white', linewidths=2, label='Prediction', zorder=10)
    
    plt.legend(loc='upper right', fontsize=12)
    plt.title("Predicted Posterior Density", fontsize=14)
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig("docs/paper/figures/posterior_density.png", dpi=300, bbox_inches='tight')
    print("Saved docs/paper/figures/posterior_density.png")
    plt.close()
    
    print("\n✓ Successfully generated paper figures!")

if __name__ == "__main__":
    main()
