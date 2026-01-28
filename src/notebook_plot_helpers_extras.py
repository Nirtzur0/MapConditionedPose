"""Extra notebook plotting helpers (advanced/rarely used)."""

from src.notebook_plot_helpers import *  # re-use core helpers
from src.notebook_plot_helpers import (
    _save_figure,
    _resolve_scene_size,
    _resolve_scene_extent_scalar,
    _extract_scene_bbox,
    _prettify_footprint,
    _coerce_points,
    _infer_scene_limits,
    _points_look_global,
    _points_look_local,
    _points_look_local_extent,
    _points_look_local_bounds,
    _points_look_normalized,
)

def plot_ue_tx_scatter_multi(
    datasets,
    max_samples=None,
    scene_id_filter=None,
    save_path=None,
    max_scenes=6,
    coord_mode="auto",
    ue_size=22,
    ue_alpha=0.55,
    tx_size=110,
):
    """Overlay UE positions from multiple datasets/configs per scene."""
    if coord_mode not in ("scene", "global", "auto"):
        raise ValueError("coord_mode must be one of: 'scene', 'global', 'auto'")

    if isinstance(datasets, dict):
        dataset_items = list(datasets.items())
    else:
        dataset_items = [(f"config_{i}", ds) for i, ds in enumerate(datasets)]

    prepared = []
    for label, ds in dataset_items:
        if hasattr(ds, "_ensure_initialized"):
            prepared.append((label, ds))
            continue
        if isinstance(ds, (str, Path)):
            from src.datasets.lmdb_dataset import LMDBRadioLocalizationDataset

            prepared.append((label, LMDBRadioLocalizationDataset(str(ds), split="all", normalize=False)))
        else:
            raise ValueError(f"Unsupported dataset entry for label '{label}'")

    scene_to_bbox = {}
    scene_to_tx = {}
    scene_to_ue_by_label = {}

    for label, ds in prepared:
        ds._ensure_initialized()
        indices = ds._indices
        if indices is None:
            n_samples = ds._metadata.get("num_samples", 0) if ds._metadata else 0
            indices = np.arange(n_samples)
        if max_samples is not None:
            indices = indices[:max_samples]

        scene_to_ue = {}
        scenes_seen = set()

        with ds._env.begin() as txn:
            for sample_idx in indices:
                sample = ds._load_sample_by_index(int(sample_idx), txn=txn)
                if sample is None:
                    continue

                scene_id = sample.get("scene_id", "")
                if isinstance(scene_id, bytes):
                    scene_id = scene_id.decode("utf-8", errors="ignore")
                if scene_id_filter and scene_id != scene_id_filter:
                    continue

                scene_metadata = sample.get("scene_metadata") or {}
                bbox = _extract_scene_bbox(sample, scene_metadata)
                if scene_id not in scene_to_bbox and bbox is not None:
                    scene_to_bbox[scene_id] = bbox

                pos = sample.get("position")
                if pos is not None:
                    scene_to_ue.setdefault(scene_id, []).append((float(pos[0]), float(pos[1])))

                if scene_id not in scenes_seen:
                    tx_local = scene_metadata.get("tx_local") or []
                    if tx_local:
                        for site in tx_local:
                            site_pos = site.get("position")
                            if not site_pos:
                                continue
                            scene_to_tx.setdefault(scene_id, []).append(
                                (float(site_pos[0]), float(site_pos[1]))
                            )
                    else:
                        sites = scene_metadata.get("sites") or []
                        for site in sites:
                            site_pos = site.get("position")
                            if not site_pos:
                                continue
                            scene_to_tx.setdefault(scene_id, []).append(
                                (float(site_pos[0]), float(site_pos[1]))
                            )
                    scenes_seen.add(scene_id)

        scene_to_ue_by_label[label] = scene_to_ue

    scene_ids = list(scene_to_ue_by_label.values())
    if scene_ids:
        scene_ids = list({sid for mapping in scene_to_ue_by_label.values() for sid in mapping.keys()})
    else:
        scene_ids = []
    if scene_id_filter:
        scene_ids = [scene_id_filter] if scene_id_filter in scene_ids else scene_ids
    if len(scene_ids) > 1:
        if max_scenes is not None and len(scene_ids) > max_scenes:
            scene_ids = scene_ids[:max_scenes]
            print(f"plot_ue_tx_scatter_multi: showing first {max_scenes} scenes.")

    if not scene_ids:
        print("No scenes found to plot.")
        return None

    n = len(scene_ids)
    cols = 2 if n > 1 else 1
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(8 * cols, 7 * rows), squeeze=False)

    colors = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
    if not colors:
        colors = ["tab:blue", "tab:orange", "tab:green", "tab:purple"]

    for i, scene_id in enumerate(scene_ids):
        ax = axes[i // cols][i % cols]
        bbox = scene_to_bbox.get(scene_id)

        tx = np.array(scene_to_tx.get(scene_id, []), dtype=np.float32)
        tx = _coerce_points(tx, bbox, coord_mode)
        if tx.size:
            ax.scatter(
                tx[:, 0],
                tx[:, 1],
                s=tx_size,
                marker="^",
                color="tab:red",
                edgecolors="black",
                linewidths=0.6,
                label="TX sites",
                zorder=3,
            )

        for idx, (label, _) in enumerate(prepared):
            ue = np.array(scene_to_ue_by_label.get(label, {}).get(scene_id, []), dtype=np.float32)
            ue = _coerce_points(ue, bbox, coord_mode)
            if not ue.size:
                continue
            color = colors[idx % len(colors)]
            ax.scatter(
                ue[:, 0],
                ue[:, 1],
                s=ue_size,
                alpha=ue_alpha,
                color=color,
                label=f"UE ({label})",
                zorder=2,
            )

        ax.set_title(scene_id, fontsize=11)
        ax.set_xlabel("X (meters)")
        ax.set_ylabel("Y (meters)")
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, alpha=0.3)
        limits = _infer_scene_limits(bbox, coord_mode)
        if limits is not None:
            ax.set_xlim(limits[0], limits[1])
            ax.set_ylim(limits[2], limits[3])

    for j in range(n, rows * cols):
        axes[j // cols][j % cols].axis("off")

    handles, labels = axes[0][0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper right")

    plt.tight_layout()
    _save_figure(fig, save_path)
    plt.show()
    return fig



def visualize_all_features(batch, sample_idx=0, model=None, verbose=False):
    """Display all input features in structured tables."""
    from IPython.display import display

    measurements = batch['measurements']

    # Extract data for the sample
    mask = measurements['mask'][sample_idx].cpu().numpy()
    seq_len = int(mask.sum())  # Only take valid steps

    if seq_len == 0:
        print("WARNING: No valid measurements in this sample.")
        return None

    # Feature schemas (matching current encoder inputs)
    rt_feature_names = [
        ('ToA', 's'),
        ('unused (mean_path_gain)', ''),
        ('unused (max_path_gain)', ''),
        ('unused (mean_path_delay)', 's'),
        ('unused (num_paths)', ''),
        ('RMS Delay Spread', 's'),
        ('RMS Angular Spread', 'rad'),
        ('Total Path Power', ''),
        ('N Significant Paths', ''),
        ('unused (delay_range)', 's'),
        ('unused (dominant_path_gain)', ''),
        ('unused (dominant_path_delay)', 's'),
        ('Is NLOS', ''),
        ('Doppler Spread', 'Hz'),
        ('Coherence Time', 's'),
        ('reserved_15', ''),
    ]

    phy_feature_names = [
        ('RSRP', 'dBm'),
        ('RSRQ', 'dB'),
        ('SINR', 'dB'),
        ('CQI', ''),
        ('RI', ''),
        ('PMI', ''),
        ('unused (l1_rsrp)', 'dBm'),
        ('Best Beam ID', ''),
    ]

    mac_feature_names = [
        ('Serving Cell ID', ''),
        ('Neighbor Cell ID 1', ''),
        ('Neighbor Cell ID 2', ''),
        ('Timing Advance', ''),
        ('DL Throughput', 'Mbps'),
        ('BLER', ''),
    ]

    rt_dim = measurements['rt_features'].shape[-1]
    phy_dim = measurements['phy_features'].shape[-1]
    mac_dim = measurements['mac_features'].shape[-1]

    if verbose:
        print(
            f"Feature summary (sample {sample_idx}): "
            f"timesteps={seq_len}, RT={rt_dim}, PHY={phy_dim}, MAC={mac_dim}"
        )

    id_data = {
        'Step': list(range(seq_len)),
        'Timestamp': measurements['timestamps'][sample_idx, :seq_len].cpu().numpy(),
        'Cell ID': measurements['cell_ids'][sample_idx, :seq_len].cpu().numpy(),
        'Beam ID': measurements['beam_ids'][sample_idx, :seq_len].cpu().numpy(),
    }
    df_ids = pd.DataFrame(id_data)
    display(df_ids.style.set_caption("Identifiers (Not Normalized)"))

    rt_values = measurements['rt_features'][sample_idx, :seq_len, :].cpu().numpy()
    rt_data = {}
    for i in range(min(rt_dim, len(rt_feature_names))):
        name, unit = rt_feature_names[i]
        # Skip unused features
        if name.startswith('unused') or name.startswith('reserved'):
            continue
        rt_data[f"{name} ({unit})" if unit else name] = rt_values[:, i]

    df_rt = pd.DataFrame(rt_data)
    styled_rt = df_rt.style.background_gradient(cmap='RdYlGn', axis=0, subset=df_rt.columns[1:]).format(precision=4)
    display(styled_rt.set_caption("RT Layer Features"))

    phy_values = measurements['phy_features'][sample_idx, :seq_len, :].cpu().numpy()
    phy_data = {}
    for i in range(min(phy_dim, len(phy_feature_names))):
        name, unit = phy_feature_names[i]
        # Skip unused features
        if name.startswith('unused'):
            continue
        phy_data[f"{name} ({unit})" if unit else name] = phy_values[:, i]

    df_phy = pd.DataFrame(phy_data)
    styled_phy = df_phy.style.background_gradient(cmap='Blues', axis=0, subset=df_phy.columns[1:]).format(precision=4)
    display(styled_phy.set_caption("PHY Layer Features"))

    mac_values = measurements['mac_features'][sample_idx, :seq_len, :].cpu().numpy()
    mac_data = {}
    for i in range(min(mac_dim, len(mac_feature_names))):
        name, unit = mac_feature_names[i]
        # Skip unused features
        if name.startswith('unused'):
            continue
        mac_data[f"{name} ({unit})" if unit else name] = mac_values[:, i]

    df_mac = pd.DataFrame(mac_data)
    styled_mac = df_mac.style.background_gradient(cmap='Oranges', axis=0, subset=df_mac.columns[1:]).format(precision=4)
    display(styled_mac.set_caption("MAC Layer Features"))

    summary_data = {
        'Layer': ['RT', 'PHY', 'MAC'],
        'Feature Count': [len(rt_data), len(phy_data), len(mac_data)],
        'Mean (abs)': [np.abs(np.array(list(rt_data.values()))).mean() if rt_data else 0,
                       np.abs(np.array(list(phy_data.values()))).mean() if phy_data else 0,
                       np.abs(np.array(list(mac_data.values()))).mean() if mac_data else 0],
        'Std (abs)': [np.array(list(rt_data.values())).std() if rt_data else 0,
                      np.array(list(phy_data.values())).std() if phy_data else 0,
                      np.array(list(mac_data.values())).std() if mac_data else 0],
        'Min': [np.array(list(rt_data.values())).min() if rt_data else 0,
                np.array(list(phy_data.values())).min() if phy_data else 0,
                np.array(list(mac_data.values())).min() if mac_data else 0],
        'Max': [np.array(list(rt_data.values())).max() if rt_data else 0,
                np.array(list(phy_data.values())).max() if phy_data else 0,
                np.array(list(mac_data.values())).max() if mac_data else 0],
    }

    df_summary = pd.DataFrame(summary_data)
    display(df_summary.style.set_caption("Summary Statistics (Normalized)"))

    return {
        'identifiers': df_ids,
        'rt_features': df_rt,
        'phy_features': df_phy,
        'mac_features': df_mac,
        'summary': df_summary,
    }



def diagnose_coarse_heatmap(outputs, batch, sample_idx=0):
    """Report coarse heatmap alignment and entropy diagnostics."""
    coarse_heatmap = outputs['coarse_heatmap'][sample_idx].cpu().numpy()
    true_pos = batch['position'][sample_idx].cpu().numpy()
    true_cell = int(batch['cell_grid'][sample_idx].item())

    grid_size = int(np.sqrt(coarse_heatmap.shape[0])) if coarse_heatmap.ndim == 1 else coarse_heatmap.shape[0]
    if coarse_heatmap.ndim == 1:
        coarse_heatmap = coarse_heatmap.reshape(grid_size, grid_size)

    probs = coarse_heatmap / (coarse_heatmap.sum() + 1e-12)
    entropy = float(-np.sum(probs * np.log(probs + 1e-12)))
    max_entropy = float(np.log(grid_size * grid_size))

    gt_row = true_cell // grid_size
    gt_col = true_cell % grid_size
    pred_row, pred_col = np.unravel_index(np.argmax(coarse_heatmap), coarse_heatmap.shape)

    # Recompute GT cell index from position to check alignment
    pos_row = int(np.clip(np.floor(true_pos[1] * grid_size), 0, grid_size - 1))
    pos_col = int(np.clip(np.floor(true_pos[0] * grid_size), 0, grid_size - 1))
    recomputed_cell = pos_row * grid_size + pos_col

    print("COARSE HEATMAP DIAGNOSTICS")
    print(f"Entropy: {entropy:.3f} (max {max_entropy:.3f})")
    print(f"GT cell (dataset): {true_cell} -> (row {gt_row}, col {gt_col})")
    print(f"GT cell (from position): {recomputed_cell} -> (row {pos_row}, col {pos_col})")
    print(f"Pred peak: (row {pred_row}, col {pred_col})")



def demonstrate_bilinear_resampling(batch, sample_idx=0, verbose=False):
    """Demonstrate differentiable bilinear interpolation from radio maps."""
    radio_map = batch['radio_map'][sample_idx:sample_idx + 1]
    true_pos = batch['position'][sample_idx:sample_idx + 1]

    map_extent = (0.0, 0.0, 1.0, 1.0)

    sampled_features = differentiable_lookup(true_pos, radio_map, map_extent)

    if verbose:
        print("DIFFERENTIABLE BILINEAR RESAMPLING")
        print(f"True Position (normalized): [{true_pos[0, 0].item():.4f}, {true_pos[0, 1].item():.4f}]")
        print(f"Radio Map Shape: {radio_map.shape}")
        print("Sampled Features at True Position:")

    feature_names = ['Path Gain', 'ToA', 'AoA', 'SNR', 'SINR']
    if verbose:
        for name, val in zip(feature_names, sampled_features[0].cpu().numpy()):
            print(f"  {name:12s}: {val:.4f}")

    return sampled_features, radio_map



def visualize_bilinear_sampling(batch, outputs, sample_idx=0):
    """Visualize the bilinear sampling process on the radio map."""
    radio_map = batch['radio_map'][sample_idx].cpu().numpy()
    true_pos = batch['position'][sample_idx].cpu().numpy()
    pred_pos = outputs['predicted_position'][sample_idx].cpu().numpy()

    h, w = radio_map.shape[-2:]

    grid_x = np.linspace(0, 1, 50)
    grid_y = np.linspace(0, 1, 50)
    xx, yy = np.meshgrid(grid_x, grid_y)
    sample_positions = torch.tensor(np.stack([xx.flatten(), yy.flatten()], axis=1), dtype=torch.float32)

    radio_map_tensor = batch['radio_map'][sample_idx:sample_idx + 1].expand(len(sample_positions), -1, -1, -1)

    sampled = differentiable_lookup(
        sample_positions.to(batch['radio_map'].device),
        radio_map_tensor,
        (0.0, 0.0, 1.0, 1.0),
    )

    sampled_map = sampled[:, 0].cpu().numpy().reshape(50, 50)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    ax = axes[0]
    im = ax.imshow(radio_map[0], cmap='viridis', origin='lower')
    ax.scatter(true_pos[0] * w, true_pos[1] * h, c='lime', s=100, marker='*', edgecolors='black', label='True Pos', zorder=10)
    ax.scatter(pred_pos[0] * w, pred_pos[1] * h, c='red', s=80, marker='x', linewidth=3, label='Pred Pos', zorder=10)
    ax.set_title('Original Radio Map (Path Gain)', fontsize=12)
    plt.colorbar(im, ax=ax)
    ax.legend()

    ax = axes[1]
    im = ax.imshow(sampled_map, cmap='viridis', origin='lower', extent=[0, 1, 0, 1])
    ax.scatter(true_pos[0], true_pos[1], c='lime', s=100, marker='*', edgecolors='black', label='True Pos', zorder=10)
    ax.scatter(pred_pos[0], pred_pos[1], c='red', s=80, marker='x', linewidth=3, label='Pred Pos', zorder=10)
    ax.set_title('Bilinear Resampled Map', fontsize=12)
    plt.colorbar(im, ax=ax)
    ax.legend()

    ax = axes[2]
    grad_y, grad_x = np.gradient(sampled_map)
    grad_mag = np.sqrt(grad_x ** 2 + grad_y ** 2)
    im = ax.imshow(grad_mag, cmap='viridis', origin='lower', extent=[0, 1, 0, 1])

    # Overlay gradient direction field (normalized) for clarity
    step = max(1, sampled_map.shape[0] // 16)
    grid_x = np.linspace(0, 1, sampled_map.shape[1])[::step]
    grid_y = np.linspace(0, 1, sampled_map.shape[0])[::step]
    gx = grad_x[::step, ::step]
    gy = grad_y[::step, ::step]
    mag = np.sqrt(gx ** 2 + gy ** 2) + 1e-8
    gx = gx / mag
    gy = gy / mag
    xx, yy = np.meshgrid(grid_x, grid_y)
    ax.quiver(xx, yy, gx, gy, color='white', alpha=0.8, scale=30, width=0.004)

    ax.set_title('Gradient Magnitude + Direction', fontsize=12)
    plt.colorbar(im, ax=ax)

    plt.suptitle('Differentiable Bilinear Resampling', fontsize=14)
    plt.tight_layout()
    plt.show()



def compute_physics_loss_demo(batch, outputs, sample_idx=0, verbose=False):
    """Demonstrate physics loss computation."""
    num_channels = batch['radio_map'].shape[1]
    channel_names = ('path_gain', 'toa', 'aoa', 'snr', 'sinr')[:num_channels]

    config = PhysicsLossConfig(
        feature_weights={
            'path_gain': 1.0,
            'toa': 0.5,
            'aoa': 0.3,
            'snr': 0.8,
            'sinr': 0.8,
        },
        map_extent=(0.0, 0.0, 1.0, 1.0),
        loss_type='mse',
        normalize_features=False,
        channel_names=channel_names,
    )
    physics_loss_fn = PhysicsLoss(config).to(batch['radio_map'].device)

    pred_pos = outputs['predicted_position']
    true_pos = batch['position']

    observed_features = differentiable_lookup(true_pos, batch['radio_map'], (0.0, 0.0, 1.0, 1.0))

    physics_loss_pred = physics_loss_fn(pred_pos, observed_features, batch['radio_map'])
    physics_loss_gt = physics_loss_fn(true_pos, observed_features, batch['radio_map'])

    if verbose:
        print("PHYSICS LOSS COMPUTATION")
        print(f"Physics Loss at Predicted Positions: {physics_loss_pred.item():.6f}")
        print(f"Physics Loss at Ground Truth:        {physics_loss_gt.item():.6f}")
        print(f"Difference:                          {(physics_loss_pred - physics_loss_gt).item():.6f}")

    return physics_loss_pred, physics_loss_gt



def demonstrate_position_refinement(batch, outputs, sample_idx=0, scene_extent=512.0, model=None, verbose=False):
    """Demonstrate inference-time position refinement using physics loss."""
    initial_pos = outputs['predicted_position'].clone()
    true_pos = batch['position']

    observed_features = differentiable_lookup(true_pos, batch['radio_map'], (0.0, 0.0, 1.0, 1.0))

    mixture_params = None
    coarse_conf = None
    if isinstance(outputs, dict) and 'top_k_probs' in outputs:
        coarse_conf = outputs['top_k_probs'].max(dim=-1).values

    refine_config = RefineConfig(
        num_steps=30,
        learning_rate=0.005,
        min_confidence_threshold=0.6,
        density_weight=0.0,
        clip_to_extent=True,
        map_extent=(0.0, 0.0, 1.0, 1.0),
        physics_config=PhysicsLossConfig(
            map_extent=(0.0, 0.0, 1.0, 1.0),
            loss_type='mse',
        ),
    )

    refined_pos, refine_info = refine_position(
        initial_xy=initial_pos,
        observed_features=observed_features,
        radio_maps=batch['radio_map'],
        config=refine_config,
        confidence=coarse_conf,
        mixture_params=mixture_params,
    )

    with torch.no_grad():
        init_sim = differentiable_lookup(initial_pos, batch['radio_map'], (0.0, 0.0, 1.0, 1.0))
        ref_sim = differentiable_lookup(refined_pos, batch['radio_map'], (0.0, 0.0, 1.0, 1.0))
        init_resid = ((init_sim - observed_features) ** 2).mean(dim=1)
        ref_resid = ((ref_sim - observed_features) ** 2).mean(dim=1)
        worse_mask = ref_resid > init_resid
        if worse_mask.any():
            refined_pos = refined_pos.clone()
            refined_pos[worse_mask] = initial_pos[worse_mask]

    if isinstance(scene_extent, (list, tuple, np.ndarray)):
        scene_size = torch.tensor(scene_extent, device=true_pos.device, dtype=true_pos.dtype)
    else:
        scene_size = torch.tensor([scene_extent, scene_extent], device=true_pos.device, dtype=true_pos.dtype)
    initial_errors = torch.norm((initial_pos - true_pos) * scene_size, dim=1).cpu().numpy()
    refined_errors = torch.norm((refined_pos - true_pos) * scene_size, dim=1).cpu().numpy()

    if verbose:
        reverted = int(worse_mask.sum().item())
        print("POSITION REFINEMENT RESULTS")
        print(f"Initial Physics Loss:  {refine_info.get('loss_initial', 0.0):.6f}")
        print(f"Final Physics Loss:    {refine_info.get('loss_final', 0.0):.6f}")
        print(f"Samples Refined:       {refine_info.get('num_refined', 0)}")
        print(f"Samples Reverted:      {reverted}")
        print(f"Initial Mean Error:  {initial_errors.mean():.2f} m")
        print(f"Refined Mean Error:  {refined_errors.mean():.2f} m")
        print(f"Improvement:         {initial_errors.mean() - refined_errors.mean():.2f} m")

    return initial_pos, refined_pos, initial_errors, refined_errors



def visualize_refinement(batch, initial_pos, refined_pos, sample_idx=0, scene_extent=512.0, save_path=None):
    """Visualize the position refinement trajectory."""
    true_pos = batch['position'][sample_idx].cpu().numpy()
    init_pos = initial_pos[sample_idx].cpu().numpy()
    ref_pos = refined_pos[sample_idx].cpu().numpy()

    radio_map = batch['radio_map'][sample_idx, 0].cpu().numpy()
    h, w = radio_map.shape

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    ax = axes[0]
    im = ax.imshow(radio_map, cmap='viridis', origin='lower', extent=[0, 1, 0, 1])

    ax.annotate('', xy=(ref_pos[0], ref_pos[1]), xytext=(init_pos[0], init_pos[1]),
                arrowprops=dict(arrowstyle='->', color='cyan', lw=2))

    ax.scatter(true_pos[0], true_pos[1], c='lime', s=150, marker='*', edgecolors='black', label='Ground Truth', zorder=10)
    ax.scatter(init_pos[0], init_pos[1], c='red', s=80, marker='o', edgecolors='black', label='Initial Pred', zorder=9)
    ax.scatter(ref_pos[0], ref_pos[1], c='blue', s=80, marker='s', edgecolors='black', label='Refined Pred', zorder=9)

    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.set_title(f"Position Refinement (Sample {sample_idx})", fontsize=12)
    ax.set_xlabel('X (normalized)')
    ax.set_ylabel('Y (normalized)')
    plt.colorbar(im, ax=ax, label='Path Gain')
    ax.legend(loc='upper right')

    ax = axes[1]

    if isinstance(scene_extent, (list, tuple, np.ndarray)):
        scene_size = np.array(scene_extent, dtype=np.float32)
    else:
        scene_size = np.array([scene_extent, scene_extent], dtype=np.float32)
    init_error = np.linalg.norm((init_pos - true_pos) * scene_size)
    ref_error = np.linalg.norm((ref_pos - true_pos) * scene_size)

    df_err = pd.DataFrame({
        'stage': ['Initial', 'Refined'],
        'error_m': [init_error, ref_error],
    })
    sns.barplot(data=df_err, x='stage', y='error_m', palette=['coral', 'steelblue'], ax=ax)

    improvement = init_error - ref_error
    improvement_pct = (improvement / init_error) * 100 if init_error > 0 else 0

    ax.text(0.5, max(init_error, ref_error) * 0.95, f"Improvement: {improvement:.1f}m ({improvement_pct:.1f}%)",
            ha='center', fontsize=11, color='darkgreen')

    ax.set_ylabel('Error (m)')
    ax.set_title('Error Before vs After Refinement', fontsize=12)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    _save_figure(fig, save_path)
    plt.show()



def create_summary_visualization(metrics, losses, initial_errors, refined_errors, scene_extent=512.0):
    """Create a comprehensive summary visualization."""
    fig = plt.figure(figsize=(16, 10))

    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)

    # 1. Metrics table
    ax = fig.add_subplot(gs[0, 0])
    ax.axis('off')

    table_data = [
        ['Metric', 'Value'],
        ['Median Error', f"{metrics['Median Error (m)']:.2f} m"],
        ['RMSE', f"{metrics['RMSE (m)']:.2f} m"],
        ['67th Percentile', f"{metrics['67th Percentile (m)']:.2f} m"],
        ['90th Percentile', f"{metrics['90th Percentile (m)']:.2f} m"],
    ]

    table = ax.table(cellText=table_data, loc='center', cellLoc='center', colWidths=[0.5, 0.5])
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 1.5)

    for i in range(2):
        table[(0, i)].set_facecolor('#4472C4')
        table[(0, i)].set_text_props(color='white', weight='bold')

    ax.set_title('Evaluation Metrics', fontsize=14, pad=20)

    # 2. Loss breakdown pie chart
    ax = fig.add_subplot(gs[0, 1])
    loss_values = [losses['coarse_loss'].item(), losses['fine_loss'].item()]
    loss_labels = ['Coarse (CE)', 'Fine (Smooth-L1)']
    colors = ['#FF6B6B', '#4ECDC4']

    ax.pie(
        loss_values,
        labels=loss_labels,
        colors=colors,
        autopct='%1.1f%%',
        startangle=90,
        explode=(0.05, 0.05),
    )
    ax.set_title('Loss Breakdown', fontsize=14)

    # 3. Before/After refinement comparison
    ax = fig.add_subplot(gs[0, 2])

    df_refine = pd.DataFrame({
        'sample': np.arange(len(initial_errors)),
        'Initial': initial_errors,
        'Refined': refined_errors,
    }).melt(id_vars='sample', var_name='stage', value_name='error_m')
    sns.barplot(data=df_refine, x='sample', y='error_m', hue='stage', ax=ax, palette=['coral', 'steelblue'])

    ax.set_xlabel('Sample Index')
    ax.set_ylabel('Error (m)')
    ax.set_title('Per-Sample Refinement Improvement', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # 4. CDF comparison
    ax = fig.add_subplot(gs[1, :2])

    sorted_init = np.sort(initial_errors)
    sorted_ref = np.sort(refined_errors)
    cdf = np.arange(1, len(sorted_init) + 1) / len(sorted_init) * 100

    sns.lineplot(x=sorted_init, y=cdf, linewidth=2, label='Initial', color='coral', ax=ax)
    sns.lineplot(x=sorted_ref, y=cdf, linewidth=2, label='After Refinement', color='steelblue', ax=ax)

    ax.axhline(67, color='gray', linestyle='--', alpha=0.5)
    ax.axhline(90, color='gray', linestyle='--', alpha=0.5)
    ax.text(ax.get_xlim()[1] * 0.98, 67, '67%', ha='right', va='bottom', fontsize=10)
    ax.text(ax.get_xlim()[1] * 0.98, 90, '90%', ha='right', va='bottom', fontsize=10)

    ax.set_xlabel('Localization Error (m)', fontsize=12)
    ax.set_ylabel('CDF (%)', fontsize=12)
    ax.set_title('Error CDF: Before vs After Physics Refinement', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 105])

    # 5. Model architecture summary
    ax = fig.add_subplot(gs[1, 2])
    ax.axis('off')

    arch_text = (
        "Model Architecture\n"
        "==================\n\n"
        "1. Radio Encoder\n"
        "   - Transformer (CLS pooling)\n\n"
        "2. Map Encoder\n"
        "   - ViT (E2 optional)\n\n"
        "3. Cross-Attention Fusion\n"
        "   - Multi-head attention\n\n"
        "4. Coarse Head\n"
        "   - Grid cell classifier\n\n"
        "5. Fine Head\n"
        "   - Top-K offset + score re-ranking\n\n"
        "6. Physics Regularization\n"
        "   - Differentiable lookup\n"
    )

    ax.text(
        0.1,
        0.95,
        arch_text,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment='top',
        fontfamily='monospace',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
    )

    plt.suptitle('UE Localization Pipeline - Summary', fontsize=16, y=1.02)
    plt.tight_layout()
    plt.show()

