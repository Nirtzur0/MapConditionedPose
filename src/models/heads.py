"""
Prediction Heads: Coarse and Fine

Coarse Head: Predicts probability distribution over grid cells (32x32)
Fine Head: Predicts offset within cell with uncertainty for top-K candidates
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class CoarseHead(nn.Module):
    """Coarse positioning head - predicts grid cell probabilities.
    
    Outputs a heatmap over a coarse grid (e.g., 32x32) representing
    the probability of UE being in each cell.
    
    Args:
        d_input: Input dimension from fusion module
        grid_size: Size of the grid (32 -> 32x32 cells)
        dropout: Dropout rate
    """
    
    def __init__(
        self,
        d_input: int = 768,
        grid_size: int = 32,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.grid_size = grid_size
        self.num_cells = grid_size ** 2
        
        # MLP for classification
        self.classifier = nn.Sequential(
            nn.Linear(d_input, d_input),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_input, d_input // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_input // 2, self.num_cells),
        )
    
    def forward(self, fused: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            fused: [batch, d_input] fused representation
        
        Returns:
            logits: [batch, num_cells] raw logits
            heatmap: [batch, grid_size, grid_size] probability distribution
        """
        # Predict logits for each cell
        logits = self.classifier(fused)  # [B, num_cells]
        
        # Convert to probability distribution (softmax)
        probs = F.softmax(logits, dim=-1)  # [B, num_cells]
        
        # Reshape to 2D heatmap
        heatmap = probs.view(-1, self.grid_size, self.grid_size)  # [B, H, W]
        
        return logits, heatmap
    
    def get_top_k_cells(
        self,
        heatmap: torch.Tensor,
        k: int = 5,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get top-K cells by probability.
        
        Args:
            heatmap: [batch, grid_size, grid_size]
            k: Number of top cells to return
        
        Returns:
            top_k_indices: [batch, k] flattened cell indices
            top_k_probs: [batch, k] corresponding probabilities
        """
        batch_size = heatmap.shape[0]
        
        # Flatten heatmap
        probs_flat = heatmap.view(batch_size, -1)  # [B, num_cells]
        
        # Get top-K
        top_k_probs, top_k_indices = torch.topk(probs_flat, k, dim=-1)
        
        return top_k_indices, top_k_probs
    
    def indices_to_coords(
        self,
        indices: torch.Tensor,
        cell_size: float,
    ) -> torch.Tensor:
        """Convert grid indices to (x, y) coordinates.
        
        Args:
            indices: [batch, k] cell indices
            cell_size: Size of each cell in meters
        
        Returns:
            coords: [batch, k, 2] (x, y) cell centers in meters
        """
        # Convert flat index to (row, col)
        rows = indices // self.grid_size
        cols = indices % self.grid_size
        
        # Convert to metric coordinates (cell centers)
        y = (rows.float() + 0.5) * cell_size
        x = (cols.float() + 0.5) * cell_size
        
        coords = torch.stack([x, y], dim=-1)  # [B, k, 2]
        
        return coords


class FineHead(nn.Module):
    """Fine positioning head - predicts offset within cell.
    
    For each top-K candidate cell, predicts:
    - (Δx, Δy): Offset from cell center
    - (σx, σy): Uncertainty estimates
    
    Uses heteroscedastic uncertainty (predicts both mean and variance).
    
    Args:
        d_input: Input dimension from fusion module
        d_hidden: Hidden dimension for refinement network
        top_k: Number of candidate cells to refine
        dropout: Dropout rate
    """
    
    def __init__(
        self,
        d_input: int = 768,
        d_hidden: int = 256,
        top_k: int = 5,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.top_k = top_k
        
        # Per-candidate refinement network
        # Input: fused representation + cell embedding
        self.cell_embedding = nn.Parameter(torch.randn(1, d_hidden))
        
        self.refiner = nn.Sequential(
            nn.Linear(d_input + d_hidden, d_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_hidden, d_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        
        # Predict mean offset (Δx, Δy)
        self.mean_head = nn.Linear(d_hidden, 2)
        
        # Predict log variance (for numerical stability)
        self.logvar_head = nn.Linear(d_hidden, 2)
    
    def forward(
        self,
        fused: torch.Tensor,
        top_k_indices: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            fused: [batch, d_input] fused representation
            top_k_indices: [batch, k] top-K cell indices
        
        Returns:
            offsets: [batch, k, 2] (Δx, Δy) in meters
            uncertainties: [batch, k, 2] (σx, σy) in meters
        """
        batch_size, k = top_k_indices.shape
        
        # Expand fused representation for each candidate
        fused_expanded = fused.unsqueeze(1).expand(-1, k, -1)  # [B, k, d_input]
        
        # Add cell-specific embedding
        cell_emb = self.cell_embedding.expand(batch_size, k, -1)  # [B, k, d_hidden]
        
        # Concatenate
        combined = torch.cat([fused_expanded, cell_emb], dim=-1)  # [B, k, d_input+d_hidden]
        
        # Refine
        refined = self.refiner(combined)  # [B, k, d_hidden]
        
        # Predict offset (mean)
        offsets = self.mean_head(refined)  # [B, k, 2]
        
        # Predict uncertainty (log variance -> std dev)
        log_var = self.logvar_head(refined)  # [B, k, 2]
        uncertainties = torch.exp(0.5 * log_var)  # [B, k, 2]
        
        return offsets, uncertainties
    
    def nll_loss(
        self,
        predicted_offsets: torch.Tensor,
        predicted_uncert: torch.Tensor,
        true_offsets: torch.Tensor,
        top_k_probs: torch.Tensor,
    ) -> torch.Tensor:
        """Negative log-likelihood loss for Gaussian predictions.
        
        Args:
            predicted_offsets: [batch, k, 2] predicted (Δx, Δy)
            predicted_uncert: [batch, k, 2] predicted (σx, σy)
            true_offsets: [batch, 2] ground truth offset from best cell
            top_k_probs: [batch, k] probabilities of each candidate
        
        Returns:
            loss: Scalar NLL loss
        """
        batch_size, k, _ = predicted_offsets.shape
        
        # Expand true offsets for each candidate
        true_offsets_expanded = true_offsets.unsqueeze(1).expand(-1, k, -1)  # [B, k, 2]
        
        # Compute NLL for Gaussian: -log p(y|μ,σ) = 0.5*(log(2πσ²) + (y-μ)²/σ²)
        residuals = true_offsets_expanded - predicted_offsets  # [B, k, 2]
        
        # Prevent division by zero
        var = predicted_uncert ** 2 + 1e-6  # [B, k, 2]
        
        # NLL per dimension
        nll = 0.5 * (torch.log(2 * torch.pi * var) + residuals ** 2 / var)  # [B, k, 2]
        
        # Sum over dimensions (x, y)
        nll = nll.sum(dim=-1)  # [B, k]
        
        # Weight by candidate probabilities (focus on likely cells)
        weighted_nll = nll * top_k_probs  # [B, k]
        
        # Average over candidates and batch
        loss = weighted_nll.mean()
        
        return loss
