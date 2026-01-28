"""
Prediction Heads: Coarse and Fine

Coarse Head: Predicts probability distribution over grid cells (32x32)
Fine Head: Predicts offset within cell and a score for top-K candidates
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


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
        origin: Tuple[float, float] = (0.0, 0.0),
    ) -> torch.Tensor:
        """Convert grid indices to (x, y) coordinates.
        
        Args:
            indices: [batch, k] cell indices
            cell_size: Size of each cell in meters
            origin: (x_min, y_min) coordinate of the grid origin
        
        Returns:
            coords: [batch, k, 2] (x, y) cell centers in meters
        """
        # Convert flat index to (row, col)
        rows = indices // self.grid_size
        cols = indices % self.grid_size
        
        # Convert to metric coordinates (cell centers) in bottom-left origin
        y = (rows.float() + 0.5) * cell_size
        x = (cols.float() + 0.5) * cell_size
        
        # Stack coordinates
        coords = torch.stack([x, y], dim=-1)  # [B, k, 2]
        
        # Apply origin offset
        origin_tensor = torch.tensor(origin, device=indices.device, dtype=coords.dtype)
        coords = coords + origin_tensor
        
        return coords


class FineHead(nn.Module):
    """Fine positioning head - predicts offset within cell and a candidate score.

    For each top-K candidate cell, predicts:
    - (Δx, Δy): Offset from cell center
    - s: Candidate score used for re-ranking hypotheses

    Uses 2D sinusoidal positional encoding instead of learned embeddings
    to better capture spatial structure and enable generalization.

    Args:
        d_input: Input dimension from fusion module
        d_hidden: Hidden dimension for refinement network
        top_k: Number of candidate cells to refine
        num_cells: Total number of cells in the grid (for computing grid_size)
        d_map: Dimension of local map patch embeddings (if used)
        use_local_map: Whether to condition refinement on local map patches
        offset_scale: Max offset scale relative to cell size
        dropout: Dropout rate
    """
    
    def __init__(
        self,
        d_input: int = 768,
        d_hidden: int = 256,
        top_k: int = 5,
        num_cells: int = 1024,  # Default 32x32
        d_map: int = 0,
        use_local_map: bool = False,
        offset_scale: float = 1.5,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.top_k = top_k
        self.offset_scale = offset_scale
        self.grid_size = int(num_cells ** 0.5)  # Infer grid size (32 for 1024 cells)
        self.d_hidden = d_hidden
        self.use_local_map = use_local_map and d_map > 0
        
        # 2D positional encoding projection (instead of learned embeddings)
        # We'll use sinusoidal 2D encoding: sin/cos of x and y coordinates
        # This captures spatial structure better than arbitrary embeddings
        self.pos_proj = nn.Sequential(
            nn.Linear(d_hidden, d_hidden),
            nn.LayerNorm(d_hidden),
            nn.GELU(),
        )
        
        map_dim = d_hidden if self.use_local_map else 0
        self.map_proj = None
        if self.use_local_map:
            self.map_proj = nn.Sequential(
                nn.Linear(d_map, d_hidden),
                nn.LayerNorm(d_hidden),
                nn.GELU(),
            )

        self.refiner = nn.Sequential(
            nn.Linear(d_input + d_hidden + map_dim, d_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_hidden, d_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        
        # Predict mean offset (Δx, Δy)
        self.mean_head = nn.Linear(d_hidden, 2)
        
        # Predict candidate score for re-ranking
        self.score_head = nn.Linear(d_hidden, 1)
        
        # Precompute sinusoidal basis for positional encoding
        # Using d_hidden/2 frequencies for x and y each
        freqs = torch.exp(torch.arange(0, d_hidden // 4, dtype=torch.float32) * 
                          (-torch.log(torch.tensor(10000.0)) / (d_hidden // 4)))
        self.register_buffer('pos_freqs', freqs)
    
    def _compute_2d_pos_encoding(self, indices: torch.Tensor) -> torch.Tensor:
        """Compute 2D sinusoidal positional encoding for cell indices.
        
        Args:
            indices: [batch, k] flat cell indices
            
        Returns:
            pos_encoding: [batch, k, d_hidden] positional encodings
        """
        # Convert flat index to (row, col)
        rows = indices // self.grid_size  # [B, k]
        cols = indices % self.grid_size   # [B, k]
        
        # Normalize to [0, 1]
        rows_norm = rows.float() / self.grid_size  # [B, k]
        cols_norm = cols.float() / self.grid_size  # [B, k]
        
        # Compute sinusoidal encoding
        # freqs: [d_hidden/4]
        # We want [B, k, d_hidden]
        
        # Compute phases: [B, k, 1] * [d_hidden/4] = [B, k, d_hidden/4]
        phase_x = cols_norm.unsqueeze(-1) * self.pos_freqs * 2 * torch.pi
        phase_y = rows_norm.unsqueeze(-1) * self.pos_freqs * 2 * torch.pi
        
        # Sin and cos for x and y: each [B, k, d_hidden/4]
        enc_x = torch.cat([torch.sin(phase_x), torch.cos(phase_x)], dim=-1)  # [B, k, d_hidden/2]
        enc_y = torch.cat([torch.sin(phase_y), torch.cos(phase_y)], dim=-1)  # [B, k, d_hidden/2]
        
        # Concatenate x and y encodings
        pos_enc = torch.cat([enc_x, enc_y], dim=-1)  # [B, k, d_hidden]
        
        return pos_enc
    
    def forward(
        self,
        fused: torch.Tensor,
        top_k_indices: torch.Tensor,
        cell_size: float,
        local_map_embeddings: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            fused: [batch, d_input] fused representation
            top_k_indices: [batch, k] top-K cell indices
            cell_size: Size of each grid cell in normalized coords
            local_map_embeddings: [batch, k, d_map] optional patch embeddings
        
        Returns:
            offsets: [batch, k, 2] (Δx, Δy) in normalized coords
            scores: [batch, k] candidate scores (unnormalized)
        """
        batch_size, k = top_k_indices.shape
        
        # Expand fused representation for each candidate
        fused_expanded = fused.unsqueeze(1).expand(-1, k, -1)  # [B, k, d_input]
        
        # Get 2D positional encoding for cell positions
        pos_enc = self._compute_2d_pos_encoding(top_k_indices)  # [B, k, d_hidden]
        pos_enc = self.pos_proj(pos_enc)  # [B, k, d_hidden]
        
        combined_parts = [fused_expanded, pos_enc]

        if self.use_local_map:
            if local_map_embeddings is None:
                local_map_embeddings = fused_expanded.new_zeros(
                    (batch_size, k, self.map_proj[0].in_features)
                )
            if local_map_embeddings.shape[:2] != (batch_size, k):
                raise ValueError("local_map_embeddings must have shape [batch, k, d_map]")
            if local_map_embeddings.shape[2] != self.map_proj[0].in_features:
                raise ValueError("local_map_embeddings last dimension must match d_map")
            map_enc = self.map_proj(local_map_embeddings)
            combined_parts.append(map_enc)

        combined = torch.cat(combined_parts, dim=-1)  # [B, k, d_input+d_hidden(+d_hidden)]
        
        # Refine
        refined = self.refiner(combined)  # [B, k, d_hidden]
        
        # Predict offset (mean) - bounded to cell size using tanh
        offset_raw = self.mean_head(refined)  # [B, k, 2]
        cell_size_t = torch.as_tensor(cell_size, device=refined.device, dtype=refined.dtype)
        max_offset = cell_size_t * self.offset_scale
        offsets = torch.tanh(offset_raw) * max_offset  # [B, k, 2] bounded by cell size
        
        scores = self.score_head(refined).squeeze(-1)  # [B, k]

        return offsets, scores
