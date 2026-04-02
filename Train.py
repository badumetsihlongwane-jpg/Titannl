"""
TITAN-NL v5.0: Real-PnL, Triple-Barrier SL/TP, Dual-Head Architecture
======================================================================

ARCHITECTURE:
1. SelfModifyingDeltaMemory  — stateful delta-memory cell, M threaded across chunks
2. ContinuumMemoryMLP (CMS)  — multi-frequency MLP hierarchy
3. MarketRegimeMemory        — regime-aware graph attention
4. NestedGraphTitanNL v5     — dual-head: direction [-1,1] × gate [0,1]
5. RealPnLLoss               — cost-aware PnL: spread + CVaR + vol-scaling
6. TripleBarrier             — realistic SL/TP realized return targets
7. Online Evolution Engine   — bar-by-bar weight update after live candle close
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import RobustScaler
import math
import pickle
import os
import io, sys
from typing import Optional, Tuple, List

try:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
except Exception:
    pass

# ==========================================
# 1. CONFIGURATION
# ==========================================
CHUNK_LEN   = 16         # Bars per TBPTT chunk
PAIRS       = ['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD']
NUM_NODES   = len(PAIRS)
D_MODEL     = 96         # Scaled up from 64 for 2-year dataset (~450K params)
EPOCHS      = 50         # More epochs for larger dataset (was 30)
PATIENCE    = 12         # Wider patience (was 8) for steady convergence
LR          = 2e-4       # Slightly lower LR for stability over full year
NOISE_STD   = 0.01       # Lighter noise on larger, more diverse dataset
ONLINE_LR   = 5e-6

# ── Date ranges — Titan30M_Dataset.csv (Feb 2025 – Feb 2026) ─────────────────
# Dataset: Titan30M_Dataset.csv  |  12,356 rows  |  2025-02-26 -> 2026-02-25
# Split: Train 8mo / Val 2mo / Calib 1mo / Backtest 1mo
TRAIN_START = "2025-02-26"
TRAIN_END   = "2025-10-31"   # ~8 months training
VAL_START   = "2025-11-01"
VAL_END     = "2025-12-31"   # 2 months val

CALIB_START    = "2026-01-01"
CALIB_END      = "2026-01-31"
BACKTEST_START = "2026-02-01"
BACKTEST_END   = "2026-02-25"   # exact period we failed on before
CALIB_DAYS     = 30

# ── Bar timing (set to match your dataset) ───────────────────────────────────
BARSPERYEAR_15M = 22176    # 15m bars per trading year
BARSPERYEAR_30M = 11088    # 30m bars per trading year

# Change this to '30m' once you have Titan30M_Dataset.csv
DATASET_INTERVAL  = '30m'  # '15m' or '30m'
BARSPERYEAR       = BARSPERYEAR_30M if DATASET_INTERVAL == '30m' else BARSPERYEAR_15M

CMS_CHUNK_SIZES  = [16, 64, 256]

# ── Real PnL Loss hyperparams ─────────────────────────────────────────────────
SPREAD_BPS   = 1.0       # Round-trip spread cost in bps per trade
LAMBDA_TC    = 0.5       # Turnover penalty weight
LAMBDA_CVAR  = 0.1       # CVaR tail-risk penalty weight
TARGET_VOL   = 0.001     # Target per-pair volatility (~10 bps / 15m bar)
CVAR_QUANTILE = 0.10     # Penalise worst-10% PnL outcomes

# ── Triple-Barrier SL/TP hyperparams ─────────────────────────────────────────
ATR_PERIOD   = 14        # Bars for ATR rolling window
K_TP         = 2.0       # TP multiple of ATR
K_SL         = 1.5       # SL multiple of ATR
# MAX_HOLD in bars: 6×30m = 3h horizon (same wall-clock as 12×15m)
MAX_HOLD     = 6  if DATASET_INTERVAL == '30m' else 12

try:
    _here = os.path.dirname(os.path.abspath(__file__))
    # Try 30m dataset first, fall back to 15m
    for _fname in ('Titan30M_Dataset.csv', 'Titan15M_Dataset.csv',
                   'TitanForexDataset.csv', 'Titan_Dataset.csv'):
        _candidate = os.path.join(_here, _fname)
        if os.path.exists(_candidate):
            DATASET_PATH = _candidate
            break
    else:
        DATASET_PATH = os.path.join(_here, 'Titan30M_Dataset.csv')  # will trigger auto-search
except NameError:
    # Kaggle / notebook: try both known paths
    for _kpath in ('/kaggle/input/datasets/zackhlongwane/new30m/Titan30M_Dataset.csv',
                   '/kaggle/input/datasets/zackhlongwane/titanv2/Titan15M_Dataset.csv',
                   '/kaggle/input/titanfx/Titan30M_Dataset.csv',
                   '/kaggle/input/titanfx/TitanForexDataset.csv',
                   '/kaggle/input/titanfx/Titan15M_Dataset.csv',
                   '/kaggle/input/datasets/zackhlongwane/fxtitan/TitanForexDataset.csv'):
        if os.path.exists(_kpath):
            DATASET_PATH = _kpath
            break
    else:
        DATASET_PATH = '/kaggle/input/titanfx/Titan30M_Dataset.csv'

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"TITAN-NL v5.0: Real-PnL | Triple-Barrier SL/TP | Dual-Head Architecture")
print(f"Device: {DEVICE}")


# ==========================================
# 2. M3 OPTIMIZER (Multi-scale Momentum Muon)
# ==========================================
class M3Optimizer(optim.Optimizer):
    def __init__(self, params, lr=3e-4, betas=(0.9, 0.95, 0.999), eps=1e-8,
                 weight_decay=1e-2, ns_steps=5, slow_momentum_freq=10, alpha_slow=0.1):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay,
                        ns_steps=ns_steps, slow_momentum_freq=slow_momentum_freq, alpha_slow=alpha_slow)
        super().__init__(params, defaults)
        self.step_count = 0

    @staticmethod
    def newton_schulz(M: torch.Tensor, steps: int = 5) -> torch.Tensor:
        if M.dim() != 2 or M.shape[0] > M.shape[1]:
            return M
        norm = M.norm()
        if norm < 1e-8:
            return M
        X = M / max(norm.item(), 1e-6)
        for _ in range(steps):
            A = X @ X.T
            if A.norm().item() > 1e4:
                return M
            X = 1.5 * X - 0.5 * A @ X
        return X * norm

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        self.step_count += 1
        for group in self.param_groups:
            beta1_fast, beta1_slow, beta2 = group['betas']
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad
                if group['weight_decay'] != 0:
                    p.data.mul_(1 - group['lr'] * group['weight_decay'])
                state = self.state[p]
                if len(state) == 0:
                    state['step'] = 0
                    state['m1_fast'] = torch.zeros_like(p)
                    state['m1_slow'] = torch.zeros_like(p)
                    state['v'] = torch.zeros_like(p)
                    state['grad_running_avg'] = torch.zeros_like(p)
                    state['grad_count'] = 0
                state['step'] += 1
                m1_fast = state['m1_fast']
                m1_slow = state['m1_slow']
                v = state['v']
                m1_fast.mul_(beta1_fast).add_(grad, alpha=1 - beta1_fast)
                v.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                state['grad_count'] += 1
                count = state['grad_count']
                state['grad_running_avg'].mul_((count - 1) / count).add_(grad, alpha=1.0 / count)
                if self.step_count % group['slow_momentum_freq'] == 0:
                    m1_slow.mul_(beta1_slow).add_(state['grad_running_avg'], alpha=1 - beta1_slow)
                    state['grad_running_avg'].zero_()
                    state['grad_count'] = 0
                bias_c1 = 1 - beta1_fast ** state['step']
                bias_c2 = 1 - beta2 ** state['step']
                m1_fast_c = m1_fast / bias_c1
                v_corrected = v / bias_c2
                if m1_fast_c.dim() == 2 and min(m1_fast_c.shape) > 1:
                    m1_orth = self.newton_schulz(m1_fast_c, group['ns_steps'])
                    m1_slow_orth = self.newton_schulz(m1_slow, group['ns_steps'])
                else:
                    m1_orth = m1_fast_c
                    m1_slow_orth = m1_slow
                combined = m1_orth + group['alpha_slow'] * m1_slow_orth
                denom = v_corrected.sqrt().add_(group['eps'])
                p.data.addcdiv_(combined, denom, value=-group['lr'])
        return loss


# ==========================================
# 3. SELF-MODIFYING DELTA MEMORY (Stateful)
# ==========================================
class SelfModifyingDeltaMemory(nn.Module):
    """
    Delta Gradient Descent memory cell.
    BUG FIX: accepts prev_M from outside; returns updated M so it can be
    threaded across sequential chunks — this is what makes evolution real.
    """

    def __init__(self, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.proj_q  = nn.Linear(d_model, d_model, bias=False)
        self.proj_k  = nn.Linear(d_model, d_model, bias=False)
        self.proj_v  = nn.Linear(d_model, d_model, bias=False)
        self.value_generator = nn.Sequential(
            nn.Linear(d_model, d_model), nn.SiLU(), nn.Linear(d_model, d_model)
        )
        self.eta_proj = nn.Sequential(
            nn.Linear(d_model, d_model // 4), nn.SiLU(), nn.Linear(d_model // 4, 1), nn.Sigmoid()
        )
        self.alpha_proj = nn.Sequential(
            nn.Linear(d_model, d_model // 4), nn.SiLU(), nn.Linear(d_model // 4, 1), nn.Sigmoid()
        )
        self.out_proj = nn.Linear(d_model, d_model)
        self.norm     = nn.LayerNorm(d_model)
        self.dropout  = nn.Dropout(dropout)
        self.register_buffer('init_memory', torch.zeros(d_model, d_model))

    def forward(
        self,
        x: torch.Tensor,                       # [B, S, N, D]
        prev_M: Optional[torch.Tensor] = None  # [B*N, D, D] — persistent carry-over
    ) -> Tuple[torch.Tensor, torch.Tensor]:    # output [B,S,N,D], M [B*N,D,D]

        b, s, n, f = x.shape
        x_flat = x.view(b * n, s, f)
        residual = x_flat

        q     = self.proj_q(x_flat)          # [B*N, S, D]
        k     = self.proj_k(x_flat)
        v     = self.proj_v(x_flat)
        v_hat = self.value_generator(v)
        eta   = self.eta_proj(x_flat)   * 0.1 + 0.01   # [B*N, S, 1]
        alpha = self.alpha_proj(x_flat) * 0.5 + 0.5    # [B*N, S, 1]

        # ── Seed from external state if provided, else start fresh ──────
        if prev_M is not None:
            M = prev_M
        else:
            M = self.init_memory.unsqueeze(0).expand(b * n, -1, -1).clone()

        outputs = []
        for t in range(s):
            q_t      = q[:, t, :]
            k_t_norm = F.normalize(k[:, t, :], dim=-1)
            v_t      = v_hat[:, t, :]
            eta_t    = eta[:, t, :].unsqueeze(-1)    # [B*N, 1, 1]
            alpha_t  = alpha[:, t, :].unsqueeze(-1)  # [B*N, 1, 1]

            # Read
            out_t = torch.bmm(M, q_t.unsqueeze(-1)).squeeze(-1)

            # Delta rule write: M_t = alpha*M - eta*M*kk^T + eta*v_hat*k^T
            Mk = torch.bmm(M, k_t_norm.unsqueeze(-1))  # [B*N, D, 1]
            M  = (alpha_t * M
                  - eta_t * torch.bmm(Mk, k_t_norm.unsqueeze(-2))
                  + eta_t * torch.bmm(v_t.unsqueeze(-1), k_t_norm.unsqueeze(-2)))
            outputs.append(out_t)

        output = torch.stack(outputs, dim=1)   # [B*N, S, D]
        output = self.norm(self.dropout(self.out_proj(output)) + residual)
        return output.view(b, s, n, f), M       # ← return final M for next chunk


# ==========================================
# 4. CONTINUUM MEMORY SYSTEM (CMS) — FIXED
# ==========================================
class ContinuumMemoryMLP(nn.Module):
    """
    Multi-frequency MLP hierarchy.
    BUG FIX v4.1: removed stray `M` return — CMS is stateless in terms of a
    matrix. Its 'memory' lives entirely in its learned weights across chunks.
    Returns only the processed tensor.
    """

    def __init__(self, d_model: int, chunk_sizes: List[int] = [16, 64, 256],
                 expansion: int = 4, dropout: float = 0.1):
        super().__init__()
        self.chunk_sizes = chunk_sizes
        self.num_levels  = len(chunk_sizes)
        self.mlps = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_model * expansion),
                nn.SiLU(),
                nn.Dropout(dropout),
                nn.Linear(d_model * expansion, d_model),
                nn.Dropout(dropout)
            ) for _ in range(self.num_levels)
        ])
        self.level_weights = nn.Parameter(torch.ones(self.num_levels))
        self.level_norms   = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(self.num_levels)])
        self.final_norm    = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, step: int = 0) -> torch.Tensor:   # [B,S,N,D] → [B,S,N,D]
        b, s, n, f = x.shape
        x_flat = x.view(b * s * n, f)
        level_outputs = []
        for level_idx, (mlp, norm) in enumerate(zip(self.mlps, self.level_norms)):
            out = mlp(x_flat)
            # Stochastic depth during training: lower freqs drop less
            if self.training:
                drop_p = [0.3, 0.15, 0.0][level_idx]
                out = F.dropout(out, p=drop_p, training=True)
            level_outputs.append(norm(out + x_flat))
        weights    = F.softmax(self.level_weights, dim=0)
        aggregated = sum(w * o for w, o in zip(weights, level_outputs))
        return self.final_norm(aggregated).view(b, s, n, f)   # ← no M here


# ==========================================
# 5. MARKET REGIME ADAPTIVE MEMORY
# ==========================================
class MarketRegimeMemory(nn.Module):
    """
    Regime-aware graph attention that collapses the temporal dim.
    Returns [B, N, D] summary for prediction head.
    """

    def __init__(self, num_nodes: int, d_model: int, dropout: float = 0.2):
        super().__init__()
        self.num_nodes = num_nodes
        self.d_model   = d_model
        self.regime_detector = nn.Sequential(
            nn.Linear(d_model * 2, d_model), nn.SiLU(),
            nn.Linear(d_model, 3), nn.Softmax(dim=-1)
        )
        self.regime_eta   = nn.Parameter(torch.tensor([0.1, 0.05, 0.2]))
        self.regime_alpha = nn.Parameter(torch.tensor([0.8, 0.9, 0.6]))
        self.q_graph      = nn.Linear(d_model, d_model)
        self.k_graph      = nn.Linear(d_model, d_model)
        self.v_graph      = nn.Linear(d_model, d_model)
        self.gate_net     = nn.Sequential(
            nn.LayerNorm(d_model * 3 + 3),
            nn.Linear(d_model * 3 + 3, 64), nn.ReLU(),
            nn.Linear(64, 1)
        )
        nn.init.constant_(self.gate_net[-1].bias, 1.5)
        self.gate_act = nn.Sigmoid()
        self.norm     = nn.LayerNorm(d_model)
        self.dropout  = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # Summarise last few timesteps as 'state'
        state    = x[:, -3:, :, :].mean(dim=1)  # [B, N, D]
        b, n, d  = state.shape
        residual = state

        global_mean = state.mean(dim=1, keepdim=True)
        global_std  = state.std(dim=1, keepdim=True)

        regime_input = torch.cat([state, global_mean.expand(-1, n, -1)], dim=-1)
        regime_probs = self.regime_detector(regime_input)  # [B, N, 3]

        gate_input   = torch.cat([state, global_mean.expand(-1, n, -1),
                                  global_std.expand(-1, n, -1), regime_probs], dim=-1)
        alpha        = self.gate_act(self.gate_net(gate_input))  # [B, N, 1]

        Q = self.q_graph(state)
        K = self.k_graph(state)
        V = self.v_graph(state)

        attn_scores  = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d)
        attn_weights = F.softmax(attn_scores, dim=-1)

        I            = torch.eye(n, device=x.device).unsqueeze(0).expand(b, -1, -1)
        mixed_weights = (alpha * I) + ((1 - alpha) * attn_weights)

        out = self.norm(self.dropout(torch.matmul(mixed_weights, V)) + residual)
        return out, alpha, attn_weights, regime_probs


# ==========================================
# 6. NESTED GRAPH TITAN-NL v5 — DUAL-HEAD
# ==========================================
class NestedGraphTitanNL(nn.Module):
    """
    v5.0: Dual-head output:
      • direction_head  → tanh → [-1, 1]   (long / short strength)
      • gate_head       → sigmoid → [0, 1] (trade confidence / "do nothing" gate)

    Effective signal = direction * gate.
    Gate ≈ 0 means flat — the model learns to skip low-conviction bars.
    Output shape is unchanged: [B, N, 1], compatible with all downstream code.
    """

    def __init__(self, num_nodes: int = NUM_NODES, feats_per_node: int = 34,
                 d_model: int = D_MODEL, num_layers: int = 2, dropout: float = 0.3,
                 cms_chunk_sizes: List[int] = CMS_CHUNK_SIZES):
        super().__init__()
        mid_dim = min(d_model, max(64, feats_per_node // 3))
        self.embedding  = nn.Sequential(nn.Linear(feats_per_node, mid_dim), nn.SiLU(),
                                        nn.Linear(mid_dim, d_model))
        self.input_norm = nn.LayerNorm(d_model)

        # Sinusoidal positional encoding
        max_len  = 512
        pos_emb  = torch.zeros(1, max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        pos_emb[0, :, 0::2] = torch.sin(position * div_term)
        pos_emb[0, :, 1::2] = torch.cos(position * div_term)
        self.pos_emb = nn.Parameter(pos_emb)

        # Stateful temporal layers
        self.temporal_layers = nn.ModuleList([
            SelfModifyingDeltaMemory(d_model, dropout) for _ in range(num_layers)
        ])
        self.cms           = ContinuumMemoryMLP(d_model, cms_chunk_sizes, expansion=3, dropout=dropout)
        self.regime_memory = MarketRegimeMemory(num_nodes, d_model, dropout)

        # Shared trunk for both heads
        self.trunk = nn.Sequential(
            nn.Linear(d_model, d_model // 2), nn.GELU(), nn.Dropout(dropout)
        )

        # Head 1: direction signal  [-1, 1]
        self.direction_head = nn.Sequential(
            nn.Linear(d_model // 2, 1), nn.Tanh()
        )
        # Head 2: trade gate / confidence  [0, 1]
        # Bias init > 0 so gate starts slightly open, preventing dead-gate at step 0
        self.gate_head = nn.Sequential(
            nn.Linear(d_model // 2, 1), nn.Sigmoid()
        )
        nn.init.constant_(self.gate_head[0].bias, 0.5)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.5)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        # Re-apply gate bias after global zero-init
        nn.init.constant_(self.gate_head[0].bias, 0.5)

    def forward(
        self,
        x: torch.Tensor,            # [B, S, N, F]
        prev_states=None,           # Optional[List[Tensor]]
        return_attn: bool = False,
        step: int = 0
    ):
        """
        Returns
        -------
        signal         : [B, N, 1]  gated signal = direction * gate
        current_states : list of M tensors (one per delta-memory layer)
        """
        b, s, n, f = x.shape
        x   = self.embedding(x)                                   # [B, S, N, D]
        pos = self.pos_emb[:, :s, :].unsqueeze(2).expand(b, s, n, -1)
        x   = self.input_norm(x + pos)

        # Thread state through each delta-memory layer
        current_states = []
        for i, layer in enumerate(self.temporal_layers):
            p_M      = prev_states[i] if prev_states is not None else None
            x, new_M = layer(x, p_M)
            current_states.append(new_M)

        x = self.cms(x, step=step)                                # [B, S, N, D]

        graph_out, alpha, attn_weights, regime_probs = self.regime_memory(x)
        # graph_out: [B, N, D]

        trunk     = self.trunk(graph_out)                         # [B, N, D//2]
        direction = self.direction_head(trunk)                    # [B, N, 1]
        gate      = self.gate_head(trunk)                         # [B, N, 1]
        signal    = direction * gate                              # [B, N, 1]  effective position

        if return_attn:
            return signal, current_states, attn_weights, alpha, gate
        return signal, current_states


# ==========================================
# 7. REAL PnL LOSS  (v5.0)
# ==========================================
class RealPnLLoss(nn.Module):
    """
    Cost-aware PnL loss combining:
      1. Core PnL on vol-scaled positions      — -mean(pos * realized_return)
      2. Transaction-cost / turnover penalty   — λ_tc  * mean(|Δpos|) * spread
      3. CVaR tail-risk penalty               — λ_cvar * CVaR_q(per-bar pnl)
      4. L2 regulariser on raw direction head  — λ_l2  * mean(dir²)

    Vol-scaling:  pos = dir * gate * (TARGET_VOL / realized_vol).clamp(0.1, 3.0)
    This makes EURUSD and USDJPY contribute equal risk per unit signal.

    Triple-barrier targets (pre-computed in load_titan_dataset) replace
    raw log-returns so the model trains on realistic trade outcomes.
    """
    def __init__(
        self,
        spread_bps:     float = SPREAD_BPS,
        lambda_tc:      float = LAMBDA_TC,
        lambda_cvar:    float = LAMBDA_CVAR,
        target_vol:     float = TARGET_VOL,
        cvar_quantile:  float = CVAR_QUANTILE,
        lambda_l2:      float = 0.02,
    ):
        super().__init__()
        self.spread      = spread_bps * 1e-4   # convert bps to decimal
        self.lambda_tc   = lambda_tc
        self.lambda_cvar = lambda_cvar
        self.target_vol  = target_vol
        self.quantile    = cvar_quantile
        self.lambda_l2   = lambda_l2

    def forward(
        self,
        signal:   torch.Tensor,                     # [B, N, 1]  raw gated signal
        targets:  torch.Tensor,                     # [B, S, N]  per-bar returns
        prev_sig: Optional[torch.Tensor] = None,    # [B, N]     signal from prev chunk
    ) -> torch.Tensor:
        sig = signal.squeeze(-1)                     # [B, N]

        # ── 1. Vol-scale position ──────────────────────────────────────────
        realized_vol = targets.std(dim=1).clamp(min=1e-8)   # [B, N]
        scale        = (self.target_vol / realized_vol).clamp(0.1, 3.0)
        pos          = sig * scale                           # [B, N]

        # ── 2. Core PnL (on pre-computed targets) ─────────────────────────
        r_net = targets.sum(dim=1)                          # [B, N]
        pnl   = pos * r_net                                 # [B, N]

        # ── 3. Transaction cost (turnover) ────────────────────────────────
        if prev_sig is not None:
            turnover = (sig - prev_sig).abs().mean()        # scalar
        else:
            turnover = sig.abs().mean()                     # first chunk: full entry cost
        tc_cost = self.lambda_tc * turnover * self.spread

        # ── 4. CVaR tail-risk penalty ──────────────────────────────────────
        # Compute per-bar PnL proxy: pos (chunk-level) × per-bar return
        pos_expanded = pos.unsqueeze(1).expand_as(targets)  # [B, S, N]
        bar_pnl      = (pos_expanded * targets).view(-1)    # flatten all bars
        # CVaR = mean of worst-q% outcomes (lower is more negative → bigger penalty)
        k        = max(1, int(self.quantile * bar_pnl.numel()))
        worst_k  = torch.topk(bar_pnl, k, largest=False).values
        cvar     = -worst_k.mean()                          # positive scalar
        cvar_pen = self.lambda_cvar * cvar

        # ── 5. L2 on direction signal (prevents ±1 saturation) ────────────
        l2_pen = self.lambda_l2 * (sig ** 2).mean()

        return -pnl.mean() + tc_cost + cvar_pen + l2_pen


# ==========================================
# 8. SEQUENTIAL (CHUNK) DATASET
# ==========================================
class SequentialForexDataset(Dataset):
    """
    Non-overlapping, strictly chronological chunks.
    Each item returns (features [S, N, F], future_returns [S, N]).
    """
    def __init__(self, X: np.ndarray, returns: np.ndarray, chunk_len: int):
        self.X         = torch.FloatTensor(X)
        self.returns   = torch.FloatTensor(returns)
        self.chunk_len = chunk_len
        self.n_chunks  = len(X) // chunk_len

    def __len__(self) -> int:
        return self.n_chunks

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        s = idx * self.chunk_len
        e = s + self.chunk_len
        return self.X[s:e], self.returns[s:e]


# ==========================================
# 9. METRICS
# ==========================================
def calculate_sharpe(signals: np.ndarray, returns: np.ndarray,
                     periods_per_year: Optional[int] = None) -> float:
    # Auto-detect annualization: caller can override, otherwise we infer from global
    if periods_per_year is None:
        periods_per_year = BARSPERYEAR_15M
    pnl = signals * returns
    mu  = np.mean(pnl)
    sig = np.std(pnl)
    return 0.0 if sig < 1e-8 else (mu / sig) * np.sqrt(periods_per_year)


# ==========================================
# 10. TRAINING — SEQUENTIAL WITH TBPTT
# ==========================================
def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    amp_scaler,
    device: torch.device,
    step_counter: int,
) -> Tuple[float, float, int]:
    """One full pass through training tape; carries state + prev_sig for TC penalty."""
    model.train()
    total_loss  = 0.0
    total_pnl   = 0.0
    n_samples   = 0
    prev_states = None
    prev_sig    = None          # for transaction-cost turnover term
    use_amp     = (device.type == 'cuda')

    for x, r in loader:
        x, r = x.to(device), r.to(device)
        x    = torch.clamp(x + torch.randn_like(x) * NOISE_STD, -10.0, 10.0)

        optimizer.zero_grad()
        if use_amp:
            with torch.amp.autocast('cuda'):
                signal, current_states = model(x, prev_states=prev_states, step=step_counter)
                loss = criterion(signal, r, prev_sig=prev_sig)
            amp_scaler.scale(loss).backward()
            amp_scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            amp_scaler.step(optimizer)
            amp_scaler.update()
        else:
            signal, current_states = model(x, prev_states=prev_states, step=step_counter)
            loss = criterion(signal, r, prev_sig=prev_sig)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        prev_states  = [s.detach() for s in current_states]
        prev_sig     = signal.squeeze(-1).detach()
        total_loss  += loss.item()
        with torch.no_grad():
            r_net      = r.sum(dim=1)            # [B, N]
            total_pnl += (signal.squeeze(-1) * r_net).sum().item()
        n_samples   += r.shape[0] * r.shape[2]
        step_counter += 1

    return total_loss / len(loader), total_pnl / max(n_samples, 1), step_counter


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    initial_states=None,
    periods_per_year: int = BARSPERYEAR_15M,
):
    """Evaluate sequentially; passes prev_sig for transaction-cost tracking."""
    model.eval()
    total_loss  = 0.0
    all_signals = []
    all_returns = []
    all_gates   = []
    prev_states = initial_states
    prev_sig    = None

    for x, r in loader:
        x, r = x.to(device), r.to(device)
        x    = torch.clamp(x, -10.0, 10.0)
        signal, current_states = model(x, prev_states=prev_states)
        loss         = criterion(signal, r, prev_sig=prev_sig)
        prev_states  = current_states
        prev_sig     = signal.squeeze(-1)
        total_loss  += loss.item()
        sig_chunk    = signal.squeeze(-1)    # [B, N]
        r_net        = r.sum(dim=1)          # [B, N]
        # Recover gate magnitude: |gated_sig| / (|direction| + eps)
        # We can't separate direction from gate here, so just store |sig| as gate proxy
        all_signals.append(sig_chunk.cpu().numpy().reshape(-1, NUM_NODES))
        all_returns.append(r_net.cpu().numpy().reshape(-1, NUM_NODES))
        all_gates.append(sig_chunk.abs().cpu().numpy().reshape(-1, NUM_NODES))

    if not all_signals:
        empty = np.zeros((0, NUM_NODES), dtype=np.float32)
        return 0.0, 0.0, prev_states, empty, empty

    sig_arr  = np.concatenate([a.reshape(-1, NUM_NODES) for a in all_signals], axis=0)
    ret_arr  = np.concatenate([a.reshape(-1, NUM_NODES) for a in all_returns], axis=0)
    gate_arr = np.concatenate([a.reshape(-1, NUM_NODES) for a in all_gates],   axis=0)
    sharpe   = calculate_sharpe(sig_arr.flatten(), ret_arr.flatten(), periods_per_year)
    return total_loss / max(len(loader), 1), sharpe, prev_states, sig_arr, ret_arr


# ==========================================
# 11. TRUE ONLINE EVOLUTION (bar-by-bar)
# ==========================================
def online_evolve(
    model: nn.Module,
    x_bar: torch.Tensor,                        # [1, 1, N, F]
    observed_return: torch.Tensor,              # [1, 1, N]
    prev_states: Optional[List[torch.Tensor]],
    online_optimizer: optim.Optimizer,
    device: torch.device,
    prev_sig: Optional[torch.Tensor] = None,    # [1, N] — for turnover cost
) -> Tuple[float, torch.Tensor, List[torch.Tensor], torch.Tensor]:
    """
    One-bar online weight update using the same RealPnLLoss objective,
    but applied at single-bar granularity. Call after every live candle close.
    """
    model.train()
    x_bar           = torch.clamp(x_bar, -10.0, 10.0).to(device)
    observed_return = observed_return.to(device)

    online_optimizer.zero_grad()
    signal, current_states = model(x_bar, prev_states=prev_states)
    sig = signal.squeeze(-1)                             # [1, N]

    # Vol-scale position
    rv    = observed_return.squeeze(1).abs().clamp(min=1e-8)  # [1, N] crude single-bar vol
    scale = (TARGET_VOL / rv).clamp(0.1, 3.0)
    pos   = sig * scale

    # Core PnL
    pnl  = (pos * observed_return.squeeze(1)).mean()

    # Turnover cost
    if prev_sig is not None:
        tc = LAMBDA_TC * (sig - prev_sig).abs().mean() * (SPREAD_BPS * 1e-4)
    else:
        tc = LAMBDA_TC * sig.abs().mean() * (SPREAD_BPS * 1e-4)

    loss = -pnl + tc
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.25)
    online_optimizer.step()

    next_states = [s.detach() for s in current_states]
    model.eval()
    return loss.item(), signal.detach(), next_states, sig.detach()


# ==========================================
# 12. TRIPLE-BARRIER TARGET COMPUTATION
# ==========================================
def compute_triple_barrier_returns(
    close: np.ndarray,       # [T] closing prices for one pair
    k_tp:  float = K_TP,
    k_sl:  float = K_SL,
    max_hold: int = MAX_HOLD,
    atr_period: int = ATR_PERIOD,
) -> np.ndarray:
    """
    For each bar t, compute the realized return as:
      +TP  if price hits take-profit barrier first      (long perspective)
      -SL  if price hits stop-loss barrier first
      ret[t+max_hold]  if neither barrier is hit within max_hold bars

    Long TP  = entry + k_tp * ATR
    Long SL  = entry - k_sl * ATR
    The model signal is signed, so we return the LONG-perspective realized
    return: multiply by -1 for shorts.

    Returns float32 array of shape [T].
    """
    T = len(close)
    if T < atr_period + 2:
        return np.zeros(T, dtype=np.float32)

    # ── ATR (simple TR-based, no High/Low in dataset so use bar-to-bar range) ──
    returns = np.diff(close, prepend=close[0])
    atr = pd.Series(np.abs(returns)).ewm(span=atr_period, adjust=False).mean().values.astype(np.float64)
    atr = np.maximum(atr, 1e-8)

    realized = np.zeros(T, dtype=np.float32)
    close_f   = close.astype(np.float64)

    for t in range(T - 1):
        entry  = close_f[t]
        tp_lvl = entry + k_tp * atr[t]
        sl_lvl = entry - k_sl * atr[t]
        end_t  = min(t + max_hold, T - 1)

        outcome = (close_f[end_t] - entry) / (entry + 1e-12)  # default: expire
        for j in range(t + 1, end_t + 1):
            price = close_f[j]
            if price >= tp_lvl:
                outcome = k_tp * atr[t] / (entry + 1e-12)   # TP hit
                break
            if price <= sl_lvl:
                outcome = -k_sl * atr[t] / (entry + 1e-12)  # SL hit
                break
        realized[t] = float(outcome)

    return realized


# ==========================================
# 13. DATA LOADING  (auto-detects dataset & columns)
# ==========================================
def _find_dataset() -> str:
    """Return the first CSV found in the script directory."""
    here = os.path.dirname(os.path.abspath(__file__)) if '__file__' in dir() else '.'
    candidates = [
        'Titan15M_Dataset.csv',
        'TitanForexDataset.csv',
        'Titan_Dataset.csv',
    ]
    for c in candidates:
        p = os.path.join(here, c)
        if os.path.exists(p):
            return p
    # fallback: pick any CSV in the folder
    csvs = [f for f in os.listdir(here) if f.endswith('.csv')]
    if csvs:
        return os.path.join(here, csvs[0])
    raise FileNotFoundError(f"No CSV dataset found in {here}")


def load_titan_dataset(path: str):
    # ── auto-find file if the specified path doesn't exist ──────────────
    if not os.path.exists(path):
        path = _find_dataset()

    print(f"\n>>> Loading {os.path.basename(path)} ...")
    df = pd.read_csv(path, index_col=0, parse_dates=True).sort_index()
    print(f"    Shape: {df.shape} | {df.index.min()} → {df.index.max()}")

    # ── auto-detect target return column ────────────────────────────────
    # We now strictly expect the 12-bar target from the new dataset builder
    target_col = "target_EURUSD_ret_12"
    if target_col in df.columns:
        print(f"    Target column: {target_col}")
        df = df.dropna(subset=[target_col])
    else:
        print(f"    [WARNING] {target_col} not found! Falling back to raw processing.")

    df = df.fillna(0)

    # ── Keep only numeric columns (drops string columns like news/event names) ──
    numeric_cols = set(df.select_dtypes(include=[np.number]).columns)
    shared_cols = [
        c for c in df.columns
        if c in numeric_cols
        and not any(c.startswith(p) for p in PAIRS)
        and not c.startswith('target_')
    ]

    node_arrays = []
    for pair in PAIRS:
        pair_cols = [c for c in df.columns
                     if c in numeric_cols
                     and c.startswith(pair)
                     and not c.startswith('target_')]
        if not pair_cols:
            pair_lower = pair.lower()
            pair_cols = [c for c in df.columns
                         if c in numeric_cols
                         and c.startswith(pair_lower)
                         and not c.startswith('target_')]
        node_arrays.append(df[pair_cols + shared_cols].values)

    # Guard: all nodes must have same feature count
    min_feats = min(a.shape[1] for a in node_arrays)
    node_arrays = [a[:, :min_feats] for a in node_arrays]

    feats_per_node = min_feats
    master         = np.stack(node_arrays, axis=1).astype(np.float32)  # [T, N, F]

    # ── Future returns: triple-barrier OR fallback to dataset targets ──────
    # Priority: compute triple-barrier returns from Close prices.
    # Fallback: use pre-built target columns from the 15M dataset builder.
    future_rets = []
    tp_hits_total, sl_hits_total, expire_total = 0, 0, 0

    for pair_idx, p in enumerate(PAIRS):
        close_col = f"{p}_Close"
        if close_col in df.columns:
            close_arr  = df[close_col].ffill().bfill().values.astype(np.float64)
            tb_rets    = compute_triple_barrier_returns(close_arr, K_TP, K_SL, MAX_HOLD, ATR_PERIOD)
            # Stats
            raw_per_bar = np.diff(close_arr, prepend=close_arr[0]) / (close_arr + 1e-12)
            atr_est     = np.abs(raw_per_bar).mean()
            tp_threshold = K_TP * atr_est
            sl_threshold = K_SL * atr_est
            tp_hits = int(np.sum(np.abs(tb_rets) >= tp_threshold * 0.9))
            sl_hits = int(np.sum(tb_rets <= -sl_threshold * 0.9))
            expires = len(tb_rets) - tp_hits - sl_hits
            tp_hits_total += tp_hits; sl_hits_total += sl_hits; expire_total += expires
            found = tb_rets
            print(f"    [{p}] Triple-barrier targets: std={found.std():.5f}  "
                  f"TP%={tp_hits/max(len(tb_rets),1)*100:.1f}  "
                  f"SL%={sl_hits/max(len(tb_rets),1)*100:.1f}")
        else:
            cand = f"target_{p}_ret_12"
            if cand in df.columns:
                found = df[cand].fillna(0).values.astype(np.float32)
                print(f"    [{p}] Fallback: {cand} (std={found.std():.5f})")
            else:
                print(f"    [{p}] WARNING: no Close col or target col found. Zeros.")
                found = np.zeros(len(df), dtype=np.float32)
        future_rets.append(found)

    future_rets = np.stack(future_rets, axis=1).astype(np.float32)  # [T, N]
    T_total = len(future_rets)
    print(f"  Triple-barrier summary across all pairs:  "
          f"TP%={tp_hits_total/(T_total*NUM_NODES+1)*100:.1f}  "
          f"SL%={sl_hits_total/(T_total*NUM_NODES+1)*100:.1f}  "
          f"Expired%={expire_total/(T_total*NUM_NODES+1)*100:.1f}")


    # ── Save feature schema for live inference ───────────────────────────
    import json
    schema_cols = {}
    for pair_i, p in enumerate(PAIRS):
        p_lo = p.lower()
        p_cols = [c for c in df.columns
                  if c in numeric_cols
                  and (c.startswith(p) or c.startswith(p_lo))
                  and not c.startswith('target_')]
        schema_cols[p] = (p_cols + shared_cols)[:min_feats]
    schema = {'pairs': PAIRS, 'feats_per_node': min_feats,
              'shared_cols': shared_cols, 'node_cols': schema_cols}
    with open('titan_feature_schema.json', 'w') as _sf:
        json.dump(schema, _sf)

    print(f"    Master tensor: {master.shape} | feats/node: {feats_per_node}")
    print(f"    Pairs: {PAIRS} | shared cols: {len(shared_cols)}")
    return master, future_rets, feats_per_node, df.index


# ==========================================
# 13. MAIN — EVOLVING TRAINING LOOP
# ==========================================
if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("TITAN-NL v4.1: Evolving Stateful Profit-Maximization")
    print("=" * 60)

    # ── Load ─────────────────────────────────────────────────────────────
    master, future_returns, feats_per_node, dates = load_titan_dataset(DATASET_PATH)

    train_mask = (dates >= TRAIN_START) & (dates <= TRAIN_END)
    val_mask   = (dates >= VAL_START)   & (dates <= VAL_END)
    train_idx, val_idx = [np.where(m)[0] for m in [train_mask, val_mask]]

    print(f"\n  Splits → Train: {len(train_idx):,}  Val: {len(val_idx):,} bars")

    # ── Auto-detect bars-per-year for correct Sharpe annualization ───────
    total_years = max((dates[-1] - dates[0]).days / 365.25, 0.1)
    raw_bpy     = int(len(dates) / total_years)
    if raw_bpy < 1500:
        bars_per_year = 252          # daily
    elif raw_bpy < 5000:
        bars_per_year = 1440         # hourly
    else:
        bars_per_year = BARSPERYEAR_15M  # 15-minute
    print(f"  Detected periodicity: {bars_per_year} bars/year  (daily=252, 15M=22176)")

    # ── Scale (train stats only) ─────────────────────────────────────────
    N, Nodes, Feats = master.shape
    scaler = RobustScaler().fit(master[train_idx].reshape(-1, Feats))
    scaled = np.nan_to_num(
        scaler.transform(master.reshape(-1, Feats)).reshape(N, Nodes, Feats),
        nan=0.0, posinf=5.0, neginf=-5.0
    )
    with open('titan_scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)

    # ── Datasets ─────────────────────────────────────────────────────────
    train_ds = SequentialForexDataset(scaled[train_idx], future_returns[train_idx], CHUNK_LEN)
    val_ds   = SequentialForexDataset(scaled[val_idx],   future_returns[val_idx],   CHUNK_LEN)

    print(f"  Chunks → Train: {len(train_ds)}  Val: {len(val_ds)}  (chunk_len={CHUNK_LEN})")

    # ── Data sanity check ─────────────────────────────────────────────────
    tr_rets = future_returns[train_idx]
    nonzero_pct = np.count_nonzero(tr_rets) / max(tr_rets.size, 1) * 100
    print(f"\n  Return stats (train):")
    print(f"    mean={tr_rets.mean():.6f}  std={tr_rets.std():.6f}")
    print(f"    min={tr_rets.min():.4f}  max={tr_rets.max():.4f}")
    print(f"    non-zero: {nonzero_pct:.1f}%")
    if nonzero_pct < 1.0:
        raise ValueError("future_returns are nearly all zero! Check _log_ret column in dataset.")
    if len(train_ds) == 0:
        raise ValueError(f"Train split has {len(train_idx)} bars but CHUNK_LEN={CHUNK_LEN}. "
                         f"Reduce CHUNK_LEN to at most {len(train_idx) // 4}.")

    # Split the ~60 final trading days into calibration + backtest
    calib_mask   = (dates >= CALIB_START)   & (dates <= CALIB_END)
    backtest_mask= (dates >= BACKTEST_START) & (dates <= BACKTEST_END)
    calib_idx    = np.where(calib_mask)[0]
    backtest_idx = np.where(backtest_mask)[0]

    calib_ds    = SequentialForexDataset(scaled[calib_idx],    future_returns[calib_idx],    CHUNK_LEN)
    backtest_ds = SequentialForexDataset(scaled[backtest_idx], future_returns[backtest_idx], CHUNK_LEN)
    calib_loader    = DataLoader(calib_ds,    batch_size=1, shuffle=False)
    backtest_loader = DataLoader(backtest_ds, batch_size=1, shuffle=False)

    print(f"  Calib bars: {len(calib_idx)}  ({len(calib_ds)} chunks)  "
          f"| Backtest bars: {len(backtest_idx)}  ({len(backtest_ds)} chunks)")

    # DataLoaders for train/val priming
    train_loader    = DataLoader(train_ds, batch_size=1, shuffle=False, drop_last=True)
    val_loader      = DataLoader(val_ds,   batch_size=1, shuffle=False)

    # ── Model ─────────────────────────────────────────────────────────────
    model = NestedGraphTitanNL(
        num_nodes=NUM_NODES, feats_per_node=feats_per_node,
        d_model=D_MODEL, num_layers=2, dropout=0.3,
        cms_chunk_sizes=CMS_CHUNK_SIZES
    ).to(DEVICE)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n  Model Parameters: {total_params:,}")

    criterion = RealPnLLoss()
    optimizer = M3Optimizer(model.parameters(), lr=LR, betas=(0.9, 0.95, 0.999), weight_decay=1e-3)
    amp_scaler = torch.amp.GradScaler('cuda') if DEVICE.type == 'cuda' else None

    # ── Training ──────────────────────────────────────────────────────────
    best_val_loss = float('inf')
    patience_ctr  = 0
    step_ctr      = 0

    print(f"\n{'='*60}")
    print(f"PHASE 1: Historical Calibration ({EPOCHS} epochs)")
    print(f"{'='*60}\n")

    for epoch in range(EPOCHS):
        tr_loss, tr_pnl, step_ctr = train_epoch(
            model, train_loader, criterion, optimizer, amp_scaler, DEVICE, step_ctr
        )
        val_loss, val_sharpe, _, _, _ = evaluate(model, val_loader, criterion, DEVICE,
                                                  periods_per_year=bars_per_year)

        print(f"Epoch {epoch+1:02d}/{EPOCHS} | "
              f"Loss {tr_loss:.4e}/{val_loss:.4e} | "
              f"PnL/chunk {tr_pnl*100:.4f}% | "
              f"Val Sharpe {val_sharpe:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_ctr  = 0
            torch.save(model.state_dict(), 'Best_TITAN_EVOLVING.pth')
            print("  >> Best model saved.")
        else:
            patience_ctr += 1
            if patience_ctr >= PATIENCE:
                print(f"\n  Early stop at epoch {epoch+1}")
                break

    # ── Phase 2a: OOS Calibration (first ~30 days of Q4 2024) ────────────
    print("\n" + "=" * 60)
    print("PHASE 2: Walk-Forward OOS + Calibration + Backtest")
    print("=" * 60)

    model.load_state_dict(torch.load('Best_TITAN_EVOLVING.pth', weights_only=True))

    # Prime memory through full history chronologically
    print("  [1/3] Priming on train data...")
    _, _, states, _, _  = evaluate(model, train_loader, criterion, DEVICE,
                                   periods_per_year=bars_per_year)
    print("  [2/3] Priming on val data  (Jan–Sep 2024)...")
    _, _, states, _, _  = evaluate(model, val_loader, criterion, DEVICE, states,
                                   periods_per_year=bars_per_year)

    # ── Calibration window: Oct 1 – Nov 14 2024 ──────────────────────────
    print(f"  [3/3] Calibration window ({CALIB_START} – {CALIB_END})...")
    calib_opt = optim.AdamW(model.parameters(), lr=ONLINE_LR * 10, weight_decay=1e-4)
    calib_pnl_hist  = []
    calib_prev_sig  = None
    for x_c, r_c in calib_loader:
        x_c, r_c = x_c.to(DEVICE), r_c.to(DEVICE)
        model.train()
        calib_opt.zero_grad()
        sig_c, states = model(x_c, prev_states=states)
        states  = [s.detach() for s in states]
        loss_c  = criterion(sig_c, r_c, prev_sig=calib_prev_sig)
        loss_c.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        calib_opt.step()
        calib_prev_sig = sig_c.squeeze(-1).detach()
        with torch.no_grad():
            pnl_c = (sig_c.squeeze(-1) * r_c.sum(dim=1)).mean().item()
            calib_pnl_hist.append(pnl_c)
    calib_pnl_mean = np.mean(calib_pnl_hist) * 100 if calib_pnl_hist else 0.0
    print(f"    Calibration avg PnL/chunk: {calib_pnl_mean:.4f}%")

    # ── Backtest: Nov 15 – Dec 31 2024 ───────────────────────────────────
    print(f"\n  Backtesting ({BACKTEST_START} – {BACKTEST_END})...")
    _, bt_sharpe, _, sig_arr, ret_arr = evaluate(
        model, backtest_loader, criterion, DEVICE, states, periods_per_year=bars_per_year
    )

    n_bt = sig_arr.shape[0]
    gate_util = (np.abs(sig_arr) > 0.05).mean() * 100   # bars where gate was meaningfully open
    print(f"\nBacktest: {n_bt} chunks  |  Portfolio Sharpe: {bt_sharpe:.4f}  "
          f"|  Gate utilisation: {gate_util:.1f}%\n")
    print(f"  {'Pair':<8} {'Sharpe':>8} {'WinRate':>9} {'AvgSig':>8} {'PnL%':>9}")
    print(f"  {'-'*46}")
    for i, pair in enumerate(PAIRS):
        ps  = calculate_sharpe(sig_arr[:, i], ret_arr[:, i], bars_per_year)
        wr  = np.mean(np.sign(sig_arr[:, i]) == np.sign(ret_arr[:, i])) * 100
        avg = np.abs(sig_arr[:, i]).mean()
        cum_pnl_pct = (sig_arr[:, i] * ret_arr[:, i]).sum() * 100
        print(f"  {pair:<8} {ps:>8.3f} {wr:>8.1f}% {avg:>8.4f} {cum_pnl_pct:>8.4f}%")

    final_states = states  # carry calibrated state forward

    # ── Persist state + model for live trading ────────────────────────────
    torch.save(final_states, 'titan_final_memory_state.pt')
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': {
            'num_nodes': NUM_NODES, 'd_model': D_MODEL,
            'cms_chunk_sizes': CMS_CHUNK_SIZES,
            'feats_per_node': feats_per_node,
            'pairs': PAIRS,
        }
    }, 'titan_nl_complete.pth')

    print("\n" + "=" * 60)
    print("PHASE 3: True Online Evolution Demo (last 500 test bars)")
    print("=" * 60)
    print("(In production: call online_evolve() after every live candle close)")
    print()

    # Small AdamW for the online phase — we want tiny, clean nudges
    online_opt = optim.AdamW(model.parameters(), lr=ONLINE_LR, weight_decay=1e-4)

    x_data  = torch.FloatTensor(scaled[backtest_idx])         # [T, N, F]
    r_data  = torch.FloatTensor(future_returns[backtest_idx]) # [T, N]
    live_states = final_states

    total_live_pnl = 0.0
    demo_bars  = min(500, len(backtest_idx) - 1)
    live_prev_sig = None

    for bar_idx in range(demo_bars):
        x_bar   = x_data[bar_idx].unsqueeze(0).unsqueeze(0).to(DEVICE)   # [1,1,N,F]
        ret_bar = r_data[bar_idx].unsqueeze(0).unsqueeze(0).to(DEVICE)   # [1,1,N]

        loss_val, sig, live_states, live_prev_sig = online_evolve(
            model, x_bar, ret_bar, live_states, online_opt, DEVICE,
            prev_sig=live_prev_sig
        )
        bar_pnl = (sig.squeeze() * ret_bar.squeeze()).mean().item()
        total_live_pnl += bar_pnl

    avg_live_pnl = total_live_pnl / demo_bars
    print(f"  Live simulation over {demo_bars} bars complete.")
    print(f"  Average PnL/bar: {avg_live_pnl:.6f}")
    print(f"\nSave `Best_TITAN_EVOLVING.pth` + `titan_final_memory_state.pt` for live deployment.")
    print("\nTITAN-NL v5.0 COMPLETE.")
