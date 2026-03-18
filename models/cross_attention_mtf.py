"""
ScalForex — Cross-Attention Multi-Timeframe
M5 (query) attends to H1/H4 (key/value) context.
CAUSAL MASK: M5 bar at time T can only see completed HTF bars (time < T).
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


class CrossAttentionMTF(nn.Module):
    """Cross-attention for multi-timeframe fusion.

    M5 features (query) attend to H1/H4 features (key/value).
    Causal masking ensures no look-ahead bias.

    Args:
        d_model: Hidden dimension.
        nhead: Number of attention heads.
        context_window: Max HTF context bars.
        use_causal_mask: Enable causal masking.
    """

    def __init__(
        self,
        d_model: int = 64,
        nhead: int = 4,
        context_window: int = 24,
        use_causal_mask: bool = True,
    ) -> None:
        super().__init__()

        self.d_model = d_model
        self.use_causal_mask = use_causal_mask
        self.context_window = context_window

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, d_model),
        )

    def _build_causal_mask(
        self,
        query_times: torch.Tensor,
        context_times: torch.Tensor,
    ) -> torch.Tensor:
        """Build causal attention mask from timestamps.

        M5 bar at time T can only attend to HTF bars with time < T
        (i.e., completed bars only — no look-ahead).

        Args:
            query_times: (batch, q_len) — M5 bar timestamps.
            context_times: (batch, kv_len) — HTF bar timestamps.

        Returns:
            Bool mask (batch * nhead, q_len, kv_len) where True = blocked.
        """
        # (batch, q_len, 1) vs (batch, 1, kv_len) → (batch, q_len, kv_len)
        q = query_times.unsqueeze(2)   # (B, Q, 1)
        c = context_times.unsqueeze(1) # (B, 1, K)

        # Block if context_time > query_time (strictly future)
        # Same-time HTF bar IS allowed (it's already completed when M5 reads it)
        mask = c > q  # True = strictly future = BLOCKED

        return mask

    def forward(
        self,
        query: torch.Tensor,
        context: torch.Tensor,
        query_times: Optional[torch.Tensor] = None,
        context_times: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Cross-attention forward pass.

        Args:
            query: M5 features (batch, m5_seq, d_model).
            context: HTF features (batch, htf_seq, d_model).
            query_times: M5 timestamps for causal masking.
            context_times: HTF timestamps for causal masking.

        Returns:
            (batch, m5_seq, d_model) — enriched M5 features.
        """
        attn_mask = None

        if self.use_causal_mask and query_times is not None and context_times is not None:
            # Build causal mask from timestamps
            key_padding_mask_2d = self._build_causal_mask(query_times, context_times)
            # nn.MultiheadAttention expects: attn_mask (L, S) or (N*h, L, S)
            # We need to expand for num_heads
            nhead = self.cross_attn.num_heads
            batch_size = query.size(0)
            # (batch, q, k) → (batch*nhead, q, k)
            attn_mask = key_padding_mask_2d.unsqueeze(1).expand(
                -1, nhead, -1, -1
            ).reshape(batch_size * nhead, query.size(1), context.size(1))
            # Convert bool mask to float: True→-inf, False→0
            attn_mask = attn_mask.float().masked_fill(attn_mask, float("-inf"))

        elif self.use_causal_mask:
            # Default: simple sequential causal mask (assume 1:1 time mapping)
            q_len = query.size(1)
            k_len = context.size(1)
            # Scale indices
            q_idx = torch.arange(q_len, device=query.device).float()
            k_idx = torch.arange(k_len, device=query.device).float() * (q_len / max(k_len, 1))
            mask = k_idx.unsqueeze(0) >= q_idx.unsqueeze(1)
            attn_mask = mask.float().masked_fill(mask, float("-inf"))

        # Cross-attention with residual
        residual = query
        query_normed = self.norm1(query)

        attn_out, _ = self.cross_attn(
            query=query_normed,
            key=context,
            value=context,
            attn_mask=attn_mask,
        )
        h = residual + attn_out

        # FFN with residual
        h = h + self.ffn(self.norm2(h))

        return h
