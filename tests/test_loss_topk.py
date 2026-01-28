"""
Tests for Top-K fine loss behavior and hypothesis weighting.
"""

import torch

from src.models.ue_localization_model import UELocalizationModel


def _minimal_config(grid_size: int = 4, top_k: int = 3) -> dict:
    return {
        'dataset': {
            'scene_extent': 512.0,
        },
        'model': {
            'radio_encoder': {
                'num_cells': 2, 'num_beams': 4, 'd_model': 32, 'nhead': 2,
                'num_layers': 1, 'dropout': 0.0, 'max_seq_len': 8,
                'rt_features_dim': 8, 'phy_features_dim': 4, 'mac_features_dim': 4
            },
            'map_encoder': {
                'img_size': 64, 'patch_size': 8, 'in_channels': 10,
                'd_model': 32, 'nhead': 2, 'num_layers': 1, 'dropout': 0.0,
                'radio_map_channels': 5, 'osm_map_channels': 5
            },
            'fusion': {'d_fusion': 32, 'nhead': 2, 'dropout': 0.0},
            'coarse_head': {'grid_size': grid_size, 'dropout': 0.0},
            'fine_head': {'top_k': top_k, 'd_hidden': 32, 'dropout': 0.0},
        },
    }


def _build_model(grid_size: int = 4, top_k: int = 3) -> UELocalizationModel:
    model = UELocalizationModel(_minimal_config(grid_size=grid_size, top_k=top_k))
    model.eval()
    return model


def _manual_fine_loss(
    model: UELocalizationModel,
    outputs: dict,
    targets: dict,
) -> torch.Tensor:
    eps = 1e-6
    top_k_centers = model.coarse_head.indices_to_coords(
        outputs['top_k_indices'],
        model.cell_size,
        origin=model.origin,
    )
    top_k_probs = outputs['top_k_probs']
    fine_scores = outputs['fine_scores']
    log_pi = torch.log(top_k_probs.clamp(min=eps))
    logits = log_pi + fine_scores
    weights = torch.softmax(logits, dim=-1)

    candidate_pos = top_k_centers + outputs['fine_offsets']
    target_pos = targets['position'].unsqueeze(1).expand_as(candidate_pos)
    per_candidate = torch.nn.functional.smooth_l1_loss(
        candidate_pos,
        target_pos,
        reduction='none',
    ).sum(dim=-1)

    return (weights * per_candidate).sum(dim=-1).mean()


def test_fine_loss_uses_renormalized_top_k_probs():
    model = _build_model(grid_size=4, top_k=3)
    num_cells = model.grid_size ** 2

    top_k_indices = torch.tensor([[0, 1, 2]])
    top_k_centers = model.coarse_head.indices_to_coords(top_k_indices, model.cell_size, origin=model.origin)
    true_pos = top_k_centers[:, 0, :]  # Exact center of best cell

    # Make all components perfect so per-candidate loss is identical across k.
    target_offsets = true_pos.unsqueeze(1) - top_k_centers
    fine_offsets = target_offsets.clone()
    fine_scores = torch.zeros((1, top_k_indices.shape[1]))

    outputs = {
        'coarse_logits': torch.zeros(1, num_cells),
        'top_k_indices': top_k_indices,
        'top_k_probs': torch.tensor([[10.0, 1.0, 1.0]]),  # Not normalized
        'fine_offsets': fine_offsets,
        'fine_scores': fine_scores,
        'predicted_position': true_pos,
    }
    targets = {
        'position': true_pos,
        'cell_grid': torch.tensor([0]),
    }
    loss_weights = {
        'coarse_weight': 0.0,
        'fine_weight': 1.0,
    }

    losses = model.compute_loss(outputs, targets, loss_weights)
    manual = _manual_fine_loss(model, outputs, targets)

    assert torch.allclose(losses['fine_loss'], manual, rtol=1e-5, atol=1e-6)


def test_fine_loss_prefers_high_weight_on_correct_component():
    model = _build_model(grid_size=4, top_k=3)
    num_cells = model.grid_size ** 2

    top_k_indices = torch.tensor([[0, 1, 2]])
    top_k_centers = model.coarse_head.indices_to_coords(top_k_indices, model.cell_size, origin=model.origin)
    true_pos = top_k_centers[:, 0, :] + torch.tensor([[0.02, -0.01]])

    target_offsets = true_pos.unsqueeze(1) - top_k_centers
    # Correct component matches target; others predict zero offset (their centers)
    fine_offsets = target_offsets.clone()
    fine_offsets[:, 1:, :] = 0.0
    fine_scores = torch.zeros((1, top_k_indices.shape[1]))

    targets = {
        'position': true_pos,
        'cell_grid': torch.tensor([0]),
    }
    base_outputs = {
        'coarse_logits': torch.zeros(1, num_cells),
        'top_k_indices': top_k_indices,
        'fine_offsets': fine_offsets,
        'fine_scores': fine_scores,
        'predicted_position': true_pos,
    }
    loss_weights = {
        'coarse_weight': 0.0,
        'fine_weight': 1.0,
    }

    outputs_high = dict(base_outputs)
    outputs_high['top_k_probs'] = torch.tensor([[0.9, 0.05, 0.05]])
    outputs_low = dict(base_outputs)
    outputs_low['top_k_probs'] = torch.tensor([[0.1, 0.45, 0.45]])

    loss_high = model.compute_loss(outputs_high, targets, loss_weights)['fine_loss']
    loss_low = model.compute_loss(outputs_low, targets, loss_weights)['fine_loss']

    assert loss_high < loss_low
