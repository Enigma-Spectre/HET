# het_train.py — Hypercomplex Eulerian Transformer trainer (UPDATED)
#
# Additions in this revision (flags default-off to preserve your baselines):
#   • --qk_norm : Per-head L2 normalization for Q and K before dot-product (stabilizes quat attention)
#   • --norm {layernorm,rmsnorm} : Choose RMSNorm instead of LayerNorm
#   • --attn_logit_scale_init : Initialize a learnable scalar that multiplies attention logits (exp(logit_scale))
#   • --rope_base : Expose RoPE base frequency
#   • --load_weights : Warm-start from model-only checkpoint (already present)
#   • No default behavior changed vs prior script unless you pass the new flags.
#
# Example (baseline unchanged):
#   python het_train.py --corpus ./Corpus.txt --d_model 256 --n_layers 12 --n_heads 8 \
#     --seq_len 512 --batch_size 48 --epochs 2 --eval_every 500 \
#     --lr 7e-5 --min_lr 1e-5 --warmup 300 --amp --device cuda
#
# Example (quat attention with the stabilizers on):
#   python het_train.py --corpus ./Corpus.txt --quat_attention --quat_rope --qk_norm \
#     --d_model 512 --n_heads 8 --n_layers 12 --seq_len 512 --batch_size 8 \
#     --attn_logit_scale_init -0.2 --rope_base 10000 --lr 7e-4 --min_lr 1e-4 --warmup 600 --grad_clip 1.0

import os, math, time, argparse, random, sys
from dataclasses import dataclass, asdict
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# --------------------- Utils ---------------------

def set_matmul_high_precision():
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass

def seed_all(seed: int):
    random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

def causal_mask(B, S, device):
    m = torch.ones(S, S, dtype=torch.bool, device=device).triu(1)
    return m[None, :, :].expand(B, -1, -1)

def cosine_lr(step, total, warmup, base_lr, min_lr):
    if step < warmup:
        return base_lr * step / max(1, warmup)
    t = (step - warmup) / max(1, total - warmup)
    t = min(max(t, 0.0), 1.0)
    return min_lr + 0.5 * (base_lr - min_lr) * (1 + math.cos(math.pi * t))

def _amp_setup(device_str: str, requested_dtype: str):
    """
    Decide autocast dtype/enable for torch.amp.autocast('cuda', ...).
    Returns (enabled: bool, autocast_dtype: Optional[torch.dtype]).
    """
    if not device_str.startswith("cuda"):
        return False, None
    req = requested_dtype.lower()
    if req == "fp32":
        return False, None
    if req == "bf16":
        try:
            if torch.cuda.is_bf16_supported():
                return True, torch.bfloat16
            else:
                print("[warn] bf16 not supported on this GPU; using fp16 autocast instead.", file=sys.stderr)
                return True, torch.float16
        except Exception:
            print("[warn] bf16 support query failed; using fp16 autocast.", file=sys.stderr)
            return True, torch.float16
    # default to fp16
    return True, torch.float16

# --------------------- Tokenizer ---------------------

class ByteTokenizer:
    vocab_size = 256
    def encode(self, text: str) -> torch.Tensor:
        return torch.tensor(list(text.encode("utf-8")), dtype=torch.long)
    def decode(self, ids: torch.Tensor) -> str:
        return bytes([int(x) for x in ids]).decode("utf-8", errors="ignore")

# --------------------- Rotary / Q-RoPE ---------------------

def _rotate_half(x):
    x_even = x[..., ::2]; x_odd = x[..., 1::2]
    out = torch.empty_like(x); out[..., ::2] = -x_odd; out[..., 1::2] = x_even
    return out

class RoPE(nn.Module):
    """Standard complex RoPE on pairs."""
    def __init__(self, head_dim, base=10000.0):
        super().__init__()
        assert head_dim % 2 == 0
        self.head_dim = head_dim; self.base = base
        self.max_seq = 0; self._cos=None; self._sin=None; self._dev=None; self._dt=None

    @torch.no_grad()
    def _maybe(self, S, device, dtype):
        need = (self._cos is None or S > self.max_seq or device != self._dev or dtype != self._dt)
        if not need: return
        self.max_seq = S; self._dev=device; self._dt=dtype
        t = torch.arange(S, device=device, dtype=dtype)
        freqs = 1.0 / (self.base ** (torch.arange(0, self.head_dim, 2, device=device, dtype=dtype) / self.head_dim))
        ang = torch.einsum("s,d->sd", t, freqs)
        emb = torch.repeat_interleave(ang, 2, dim=-1)
        self._cos = emb.cos()[None, None, :, :]
        self._sin = emb.sin()[None, None, :, :]

    def apply(self, q, k):
        B,H,S,D = q.shape
        self._maybe(S, q.device, q.dtype)
        cos, sin = self._cos, self._sin
        return (q * cos) + (_rotate_half(q) * sin), (k * cos) + (_rotate_half(k) * sin)

class QRoPE(nn.Module):
    """
    Quaternion RoPE: rotate 4-ch groups with a unit quaternion u(theta) = cos(theta) + axis * sin(theta).
    axis_mode: 'cycle' (i,j,k repeating) or 'i' (all i-axis).
    conjugate: if True, apply u ⊗ q ⊗ u^{-1}; else left multiply u ⊗ q (faster).
    """
    def __init__(self, n_groups: int, base: float = 10000.0, axis_mode: str = "cycle", conjugate: bool = False):
        super().__init__()
        assert n_groups > 0
        self.n_groups = n_groups
        self.base = base
        self.axis_mode = axis_mode
        self.conjugate = conjugate

        if axis_mode == "cycle":
            idx = torch.arange(n_groups)
            ax_i = (idx % 3 == 0).float()
            ax_j = (idx % 3 == 1).float()
            ax_k = (idx % 3 == 2).float()
        elif axis_mode == "i":
            ax_i = torch.ones(n_groups); ax_j = torch.zeros(n_groups); ax_k = torch.zeros(n_groups)
        else:
            raise ValueError("axis_mode must be 'cycle' or 'i'")

        self.register_buffer("_ax_i_buf", ax_i)
        self.register_buffer("_ax_j_buf", ax_j)
        self.register_buffer("_ax_k_buf", ax_k)

        self.max_seq = 0
        self._cos = None; self._sin = None
        self._ax_i = None; self._ax_j = None; self._ax_k = None
        self._dev = None; self._dt = None

    @torch.no_grad()
    def _maybe(self, S, device, dtype):
        need = (self._cos is None or S > self.max_seq or device != self._dev or dtype != self._dt)
        if not need: return
        self.max_seq = S; self._dev = device; self._dt = dtype
        t = torch.arange(S, device=device, dtype=dtype)  # [S]
        inv = 1.0 / (self.base ** (torch.arange(0, self.n_groups, device=device, dtype=dtype) / self.n_groups))  # [G]
        ang = torch.einsum("s,g->sg", t, inv)  # [S,G]
        self._cos = ang.cos()[None, None, :, :]  # [1,1,S,G]
        self._sin = ang.sin()[None, None, :, :]  # [1,1,S,G]
        self._ax_i = self._ax_i_buf.to(device=device, dtype=dtype)[None, None, None, :]
        self._ax_j = self._ax_j_buf.to(device=device, dtype=dtype)[None, None, None, :]
        self._ax_k = self._ax_k_buf.to(device=device, dtype=dtype)[None, None, None, :]

    @staticmethod
    def _lmul(ur, ui, uj, uk, qr, qi, qj, qk):
        pr = ur*qr - ui*qi - uj*qj - uk*qk
        pi = ur*qi + ui*qr + uj*qk - uk*qj
        pj = ur*qj - ui*qk + uj*qr + uk*qi
        pk = ur*qk + ui*qj - uj*qi + uk*qr
        return pr, pi, pj, pk

    @staticmethod
    def _rmul(qr, qi, qj, qk, vr, vi, vj, vk):
        pr = qr*vr - qi*vi - qj*vj - qk*vk
        pi = qr*vi + qi*vr + qj*vk - qk*vj
        pj = qr*vj - qi*vk + qj*vr + qk*vi
        pk = qr*vk + qi*vj - qj*vi + qk*vr
        return pr, pi, pj, pk

    def apply(self, q, k):
        """
        q,k: [B, H, S, Dh], Dh % 4 == 0
        Operates on groups of 4 channels as quaternions.
        """
        B,H,S,Dh = q.shape
        assert Dh % 4 == 0
        G = Dh // 4
        self._maybe(S, q.device, q.dtype)

        def rot(x):
            x = x.view(B, H, S, G, 4)
            xr, xi, xj, xk = x[...,0], x[...,1], x[...,2], x[...,3]  # [B,H,S,G]

            ur = self._cos[..., :S, :G].expand_as(xr)
            s  = self._sin[..., :S, :G].expand_as(xr)
            ui = s * self._ax_i.expand_as(xr)
            uj = s * self._ax_j.expand_as(xr)
            uk = s * self._ax_k.expand_as(xr)

            # Left multiply
            pr, pi, pj, pk = self._lmul(ur, ui, uj, uk, xr, xi, xj, xk)

            if self.conjugate:
                # Right multiply by u^{-1} = (ur, -ui, -uj, -uk)
                pr, pi, pj, pk = self._rmul(pr, pi, pj, pk, ur, -ui, -uj, -uk)

            y = torch.stack([pr, pi, pj, pk], dim=-1).reshape(B, H, S, Dh)
            return y

        return rot(q), rot(k)

# --------------------- Quaternion Linear ---------------------

class QuaternionLinear(nn.Module):
    """Parameter-light quaternion linear: 4 weights of size (out_q, in_q)."""
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        assert in_features % 4 == 0 and out_features % 4 == 0
        self.in_quat = in_features // 4
        self.out_quat = out_features // 4
        def p():
            w = nn.Parameter(torch.empty(self.out_quat, self.in_quat))
            nn.init.xavier_uniform_(w); return w
        self.wr = p(); self.wi = p(); self.wj = p(); self.wk = p()
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None

    def forward(self, x):
        B,S,_ = x.shape; iq=self.in_quat; oq=self.out_quat
        xr = x[...,0::4].reshape(B*S, iq)
        xi = x[...,1::4].reshape(B*S, iq)
        xj = x[...,2::4].reshape(B*S, iq)
        xk = x[...,3::4].reshape(B*S, iq)
        WrT,WiT,WjT,WkT = self.wr.t(), self.wi.t(), self.wj.t(), self.wk.t()

        Ar = xr@WrT; Br = xi@WrT; Cr = xj@WrT; Dr = xk@WrT
        Ai = xr@WiT; Bi = xi@WiT; Ci = xj@WiT; Di = xk@WiT
        Aj = xr@WjT; Bj = xi@WjT; Cj = xj@WjT; Dj = xk@WjT
        Ak = xr@WkT; Bk = xi@WkT; Ck = xj@WkT; Dk = xk@WkT

        or_ = (Ar - Bi - Cj - Dk)
        oi_ = (Br + Ai + Dk - Cj)
        oj_ = (Cr - Dk + Aj + Bi)
        ok_ = (Dr + Cj - Bi + Ak)

        out = torch.empty(B*S, 4*oq, device=x.device, dtype=x.dtype)
        out[:,0::4]=or_; out[:,1::4]=oi_; out[:,2::4]=oj_; out[:,3::4]=ok_
        out = out.view(B,S,4*oq)
        if self.bias is not None: out = out + self.bias.view(1,1,-1)
        return out

class FastQuaternionLinear(QuaternionLinear):
    """Fused real-matmul form for inference/export speed."""
    def __init__(self, in_features, out_features, bias=True, cache_block_in_eval=True):
        super().__init__(in_features, out_features, bias)
        self.cache_block_in_eval = cache_block_in_eval
        self._W = None; self._dev=None; self._dt=None

    def train(self, mode: bool = True):
        self._W=None; return super().train(mode)

    @torch.no_grad()
    def _assemble(self, device, dtype):
        Wr,Wi,Wj,Wk = self.wr.to(device,dtype), self.wi.to(device,dtype), self.wj.to(device,dtype), self.wk.to(device,dtype)
        top  = torch.cat([ Wr, -Wi, -Wj, -Wk], dim=1)
        row2 = torch.cat([ Wi,  Wr,  Wk, -Wj], dim=1)
        row3 = torch.cat([ Wj, -Wk,  Wr,  Wi], dim=1)
        row4 = torch.cat([ Wk,  Wj, -Wi,  Wr], dim=1)
        return torch.cat([top,row2,row3,row4], dim=0)  # [4*out_q, 4*in_q]

    def _getW(self, device, dtype):
        if not self.cache_block_in_eval or self.training:
            return self._assemble(device, dtype)
        if self._W is None or self._dev!=device or self._dt!=dtype:
            self._W = self._assemble(device, dtype); self._dev=device; self._dt=dtype
        return self._W

    def forward(self, x):
        B,S,_ = x.shape; iq=self.in_quat
        X = torch.cat([x[...,0::4], x[...,1::4], x[...,2::4], x[...,3::4]], dim=-1)  # [B,S,4*iq]
        W = self._getW(x.device, x.dtype)  # [4*out_q, 4*in_q]
        Y = X.reshape(B*S, 4*iq) @ W.t()
        Y = Y.view(B,S,-1)
        if self.bias is not None: Y = Y + self.bias.view(1,1,-1)
        return Y

# --------------------- Norms ---------------------

class RMSNorm(nn.Module):
    def __init__(self, dim, eps: float = 1e-8):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps
    def forward(self, x):
        # x: [..., D]
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).sqrt()
        x = x / rms
        return x * self.weight

# --------------------- Attention / FFN / Model ---------------------

class StandardAttention(nn.Module):
    def __init__(self, dim, n_heads=8, dropout=0.0, rope_base=10000.0, qk_norm: bool = False,
                 attn_logit_scale_init: Optional[float] = None):
        super().__init__()
        assert dim % n_heads == 0
        self.n_heads = n_heads; self.dim = dim; self.head_dim = dim // n_heads
        self.scale = self.head_dim ** -0.5
        self.q = nn.Linear(dim, dim); self.k = nn.Linear(dim, dim); self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim); self.drop = nn.Dropout(dropout)
        self.rope = RoPE(self.head_dim, rope_base)
        self.qk_norm = qk_norm
        self.logit_scale = nn.Parameter(torch.tensor(attn_logit_scale_init)) if (attn_logit_scale_init is not None) else None

    @staticmethod
    def _l2norm_last(x, eps=1e-6):
        return x / (x.pow(2).sum(dim=-1, keepdim=True).add(eps).sqrt())

    def forward(self, x, mask=None):
        B,S,Dm = x.shape; H=self.n_heads; Dh=self.head_dim
        q = self.q(x).view(B,S,H,Dh).transpose(1,2)
        k = self.k(x).view(B,S,H,Dh).transpose(1,2)
        v = self.v(x).view(B,S,H,Dh).transpose(1,2)
        q,k = self.rope.apply(q,k)
        if self.qk_norm:
            q = self._l2norm_last(q); k = self._l2norm_last(k)
        scores = torch.einsum("bhsd,bhtd->bhst", q, k) * self.scale
        if self.logit_scale is not None:
            scores = scores * torch.exp(self.logit_scale)
        if mask is not None: scores = scores.masked_fill(mask[:,None,:,:], float("-inf"))
        a = self.drop(F.softmax(scores, dim=-1))
        out = torch.einsum("bhst,bhtd->bhsd", a, v).transpose(1,2).contiguous().view(B,S,Dm)
        return self.o(out)

class QAttention(nn.Module):
    def __init__(self, dim, n_heads=8, dropout=0.0, linear_cls=QuaternionLinear,
                 quat_rope: bool = False, qrope_axis: str = "cycle", qrope_conjugate: bool = False,
                 rope_base: float = 10000.0, qk_norm: bool = False,
                 attn_logit_scale_init: Optional[float] = None):
        super().__init__()
        assert dim % n_heads == 0
        head_dim = dim // n_heads; assert head_dim % 4 == 0
        self.n_heads=n_heads; self.dim=dim; self.head_dim=head_dim; self.scale = (head_dim) ** -0.5
        self.q = linear_cls(dim, dim); self.k = linear_cls(dim, dim); self.v = linear_cls(dim, dim); self.o = linear_cls(dim, dim)
        self.drop = nn.Dropout(dropout)
        self.use_qrope = quat_rope
        self.rope = QRoPE(head_dim // 4, base=rope_base, axis_mode=qrope_axis, conjugate=qrope_conjugate) if quat_rope else RoPE(head_dim, base=rope_base)
        self.qk_norm = qk_norm
        self.logit_scale = nn.Parameter(torch.tensor(attn_logit_scale_init)) if (attn_logit_scale_init is not None) else None

    @staticmethod
    def _l2norm_last(x, eps=1e-6):
        return x / (x.pow(2).sum(dim=-1, keepdim=True).add(eps).sqrt())

    def forward(self, x, mask=None):
        B,S,Dm = x.shape; H=self.n_heads; Dh=self.head_dim
        q = self.q(x).view(B,S,H,Dh).transpose(1,2)
        k = self.k(x).view(B,S,H,Dh).transpose(1,2)
        v = self.v(x).view(B,S,H,Dh).transpose(1,2)
        q,k = self.rope.apply(q,k)
        if self.qk_norm:
            q = self._l2norm_last(q); k = self._l2norm_last(k)
        scores = torch.einsum("bhsd,bhtd->bhst", q, k) * self.scale
        if self.logit_scale is not None:
            scores = scores * torch.exp(self.logit_scale)
        if mask is not None: scores = scores.masked_fill(mask[:,None,:,:], float("-inf"))
        a = self.drop(F.softmax(scores, dim=-1))
        out = torch.einsum("bhst,bhtd->bhsd", a, v).transpose(1,2).contiguous().view(B,S,Dm)
        return self.o(out)

class QuatFFN(nn.Module):
    def __init__(self, dim, mult=4, dropout=0.0, linear_cls=QuaternionLinear):
        super().__init__()
        hidden = dim * mult; assert hidden % 4 == 0
        self.fc1 = linear_cls(dim, hidden); self.fc2 = linear_cls(hidden, dim)
        self.act = nn.GELU(); self.drop = nn.Dropout(dropout)
    def forward(self, x):
        x = self.fc1(x); x = self.drop(self.act(x)); return self.fc2(x)

class HETBlock(nn.Module):
    def __init__(self, dim, n_heads, dropout, mult, quat_attention=False,
                 linear_cls=QuaternionLinear, quat_rope=False, qrope_axis="cycle", qrope_conjugate=False,
                 rope_base: float = 10000.0, qk_norm: bool = False, attn_logit_scale_init: Optional[float] = None,
                 norm_type: str = "layernorm"):
        super().__init__()
        Norm = nn.LayerNorm if norm_type == "layernorm" else RMSNorm
        self.norm1 = Norm(dim); self.norm2 = Norm(dim); self.drop = nn.Dropout(dropout)
        if quat_attention:
            self.attn = QAttention(dim, n_heads, dropout, linear_cls=linear_cls,
                                   quat_rope=quat_rope, qrope_axis=qrope_axis, qrope_conjugate=qrope_conjugate,
                                   rope_base=rope_base, qk_norm=qk_norm, attn_logit_scale_init=attn_logit_scale_init)
        else:
            self.attn = StandardAttention(dim, n_heads, dropout, rope_base=rope_base, qk_norm=qk_norm,
                                          attn_logit_scale_init=attn_logit_scale_init)
        self.ffn  = QuatFFN(dim, mult, dropout, linear_cls=linear_cls)
    def forward(self, x, mask):
        x = x + self.drop(self.attn(self.norm1(x), mask)); x = x + self.drop(self.ffn(self.norm2(x))); return x

class CausalLM(nn.Module):
    def __init__(self, vocab_size, d_model=512, n_layers=8, n_heads=8,
                 dropout=0.0, ffn_mult=4, quat_attention=False, tie_weights=True,
                 linear_cls=QuaternionLinear, quat_rope=False, qrope_axis="cycle", qrope_conjugate=False,
                 rope_base: float = 10000.0, qk_norm: bool = False, attn_logit_scale_init: Optional[float] = None,
                 norm_type: str = "layernorm"):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([
            HETBlock(d_model, n_heads, dropout, ffn_mult, quat_attention, linear_cls,
                     quat_rope=quat_rope, qrope_axis=qrope_axis, qrope_conjugate=qrope_conjugate,
                     rope_base=rope_base, qk_norm=qk_norm, attn_logit_scale_init=attn_logit_scale_init,
                     norm_type=norm_type)
            for _ in range(n_layers)
        ])
        self.norm_f = nn.LayerNorm(d_model) if norm_type == "layernorm" else RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        if tie_weights: self.lm_head.weight = self.tok_emb.weight

    def forward(self, input_ids, mask=None):
        x = self.drop(self.tok_emb(input_ids))
        for b in self.blocks: x = b(x, mask)
        x = self.norm_f(x)
        return self.lm_head(x)

# --------------------- Datasets ---------------------

class SlidingCorpusDataset(Dataset):
    """
    Full-corpus sliding windows:
      stride = seq_len - keep, where keep = floor(seq_len * sliding_keep_pct)
    """
    def __init__(self, ids: torch.Tensor, seq_len: int, sliding_keep_pct: float):
        assert ids.dim() == 1
        self.ids = ids
        self.seq_len = seq_len
        self.keep = int(seq_len * max(0.0, min(0.95, sliding_keep_pct)))
        self.stride = max(1, seq_len - self.keep)
        n = ids.numel()
        max_start = n - 1 - seq_len
        if max_start < 0:
            raise ValueError(f"Corpus too short for seq_len={seq_len}")
        starts = torch.arange(0, max_start + 1, self.stride)
        if starts[-1].item() != max_start:
            starts = torch.cat([starts, torch.tensor([max_start])])
        self.starts = starts

    def __len__(self): return self.starts.numel()

    def __getitem__(self, idx):
        s = int(self.starts[idx])
        x = self.ids[s : s + self.seq_len]
        y = self.ids[s + 1 : s + self.seq_len + 1]
        return {"input_ids": x, "labels": y}

class EvalCorpusDataset(Dataset):
    """Non-overlapping windows for evaluation (stride=seq_len)."""
    def __init__(self, ids: torch.Tensor, seq_len: int):
        assert ids.dim() == 1
        self.ids = ids
        self.seq_len = seq_len
        n = ids.numel()
        max_start = n - 1 - seq_len
        if max_start < 0:
            self.starts = torch.empty(0, dtype=torch.long)
        else:
            starts = torch.arange(0, max_start + 1, seq_len)
            if starts[-1].item() != max_start:
                starts = torch.cat([starts, torch.tensor([max_start])])
            self.starts = starts

    def __len__(self): return self.starts.numel()

    def __getitem__(self, idx):
        s = int(self.starts[idx])
        x = self.ids[s : s + self.seq_len]
        y = self.ids[s + 1 : s + self.seq_len + 1]
        return {"input_ids": x, "labels": y}

# --------------------- Export (fused) ---------------------

def swap_to_fast(module: nn.Module, ffn_only: bool = False):
    for name, child in list(module.named_children()):
        if isinstance(child, QuaternionLinear):
            if ffn_only:
                # Only replace if parent is a QuatFFN
                if isinstance(module, QuatFFN):
                    fast = FastQuaternionLinear(child.wr.size(1)*4, child.wr.size(0)*4, bias=(child.bias is not None))
                    fast.wr.data.copy_(child.wr.data); fast.wi.data.copy_(child.wi.data)
                    fast.wj.data.copy_(child.wj.data); fast.wk.data.copy_(child.wk.data)
                    if child.bias is not None: fast.bias.data.copy_(child.bias.data)
                    setattr(module, name, fast)
            else:
                fast = FastQuaternionLinear(child.wr.size(1)*4, child.wr.size(0)*4, bias=(child.bias is not None))
                fast.wr.data.copy_(child.wr.data); fast.wi.data.copy_(child.wi.data)
                fast.wj.data.copy_(child.wj.data); fast.wk.data.copy_(child.wk.data)
                if child.bias is not None: fast.bias.data.copy_(child.bias.data)
                setattr(module, name, fast)
        else:
            swap_to_fast(child, ffn_only)

# --------------------- Config ---------------------

@dataclass
class TrainConfig:
    corpus: str
    out_dir: str = "ckpts_quat"
    d_model: int = 512
    n_layers: int = 8
    n_heads: int = 8
    ffn_mult: int = 4
    dropout: float = 0.0
    quat_attention: bool = False
    quat_rope: bool = False
    qrope_axis: str = "cycle"  # 'cycle' or 'i'
    qrope_conjugate: bool = False
    rope_base: float = 10000.0
    qk_norm: bool = False
    attn_logit_scale_init: Optional[float] = None
    norm: str = "layernorm"  # or 'rmsnorm'
    seq_len: int = 512
    batch_size: int = 8
    accum_steps: int = 8
    # Training horizon: either steps OR epochs; if steps<=0 we compute from epochs.
    steps: int = 0
    epochs: float = 1.0
    eval_every: int = 200
    lr: float = 1e-3
    min_lr: float = 1e-4
    warmup: int = 200
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    amp: bool = True
    dtype: str = "fp16"  # fp32|fp16|bf16 ; AMP handles compute dtype
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 1337
    compile: bool = False
    train_split: float = 0.9
    sliding_keep_pct: float = 0.14
    num_workers: int = 2
    pin_memory: bool = True
    fast_export: bool = False
    ffn_only_fast_export: bool = False
    # Warm start / resume:
    load_weights: Optional[str] = None   # load 'model' weights only
    resume_ckpt: Optional[str] = None    # load model+optimizer+step

# --------------------- Checkpoint I/O ---------------------

def save_checkpoint(path, model, opt, cfg, step, best_val):
    obj = {
        "model": model.state_dict(),
        "optimizer": opt.state_dict(),
        "config": asdict(cfg),
        "step": step,
        "best_val": best_val,
        "vocab_size": ByteTokenizer.vocab_size
    }
    torch.save(obj, path)

def load_model_weights(model, path, map_location):
    obj = torch.load(path, map_location=map_location)
    state = obj["model"] if isinstance(obj, dict) and "model" in obj else obj
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        print(f"[warn] load_weights: missing={len(missing)} unexpected={len(unexpected)}", file=sys.stderr)

def load_full_resume(model, optimizer, path, map_location):
    obj = torch.load(path, map_location=map_location)
    model.load_state_dict(obj["model"])
    optimizer.load_state_dict(obj["optimizer"])
    step = obj.get("step", 0)
    best_val = obj.get("best_val", float("inf"))
    cfg_loaded = obj.get("config", None)
    return step, best_val, cfg_loaded

# --------------------- Training ---------------------

def cycle(loader):
    while True:
        for batch in loader:
            yield batch

def run_eval(model, val_dl, device, use_amp, amp_dtype):
    if val_dl is None:
        return float("nan")
    model.eval()
    total_loss = 0.0; count = 0
    with torch.no_grad():
        for batch in val_dl:
            inp = batch["input_ids"].to(device)
            tgt = batch["labels"].to(device)
            mask = causal_mask(inp.size(0), inp.size(1), device)
            with torch.amp.autocast('cuda', enabled=use_amp, dtype=amp_dtype):
                logits = model(inp, mask)
                loss = F.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    tgt.view(-1)
                )
            total_loss += loss.item(); count += 1
    return total_loss / max(1, count)

def train(cfg: TrainConfig):
    set_matmul_high_precision()
    seed_all(cfg.seed)
    os.makedirs(cfg.out_dir, exist_ok=True)

    # Load corpus
    raw = open(cfg.corpus, "r", encoding="utf-8").read()
    tok = ByteTokenizer()
    ids = tok.encode(raw)

    # Split
    n = ids.numel()
    split = int(n * cfg.train_split)
    train_ids = ids[:split].clone()
    val_ids   = ids[split:].clone() if split < n else ids[:0]

    # Datasets / loaders
    train_ds = SlidingCorpusDataset(train_ids, cfg.seq_len, cfg.sliding_keep_pct)
    val_ds   = EvalCorpusDataset(val_ids, cfg.seq_len) if val_ids.numel() > 0 else None

    train_dl = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True, drop_last=True,
        num_workers=cfg.num_workers, pin_memory=(cfg.pin_memory and cfg.device.startswith("cuda"))
    )
    val_dl   = DataLoader(
        val_ds, batch_size=cfg.batch_size, shuffle=False, drop_last=True,
        num_workers=cfg.num_workers, pin_memory=(cfg.pin_memory and cfg.device.startswith("cuda"))
    ) if val_ds else None

    # Model
    linear_cls = QuaternionLinear
    if cfg.quat_rope and not cfg.quat_attention:
        raise SystemExit("Q-RoPE requires --quat_attention and d_model divisible by 4*n_heads.")
    model = CausalLM(
        vocab_size=ByteTokenizer.vocab_size,
        d_model=cfg.d_model,
        n_layers=cfg.n_layers,
        n_heads=cfg.n_heads,
        dropout=cfg.dropout,
        ffn_mult=cfg.ffn_mult,
        quat_attention=cfg.quat_attention,
        linear_cls=linear_cls,
        quat_rope=cfg.quat_rope,
        qrope_axis=cfg.qrope_axis,
        qrope_conjugate=cfg.qrope_conjugate,
        rope_base=cfg.rope_base,
        qk_norm=cfg.qk_norm,
        attn_logit_scale_init=cfg.attn_logit_scale_init,
        norm_type=("layernorm" if cfg.norm=="layernorm" else "rmsnorm")
    ).to(cfg.device)

    if cfg.quat_attention and (cfg.d_model % (4*cfg.n_heads) != 0):
        raise SystemExit(f"d_model must be divisible by 4*n_heads when --quat_attention is set. Got {cfg.d_model} and {cfg.n_heads}.")

    if cfg.compile and hasattr(torch, "compile"):
        model = torch.compile(model)

    # Warm-start weights
    if cfg.load_weights:
        print(f"[info] loading weights from {cfg.load_weights}")
        load_model_weights(model, cfg.load_weights, map_location=cfg.device)

    # Optimizer
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr,
                            betas=(cfg.beta1, cfg.beta2), weight_decay=cfg.weight_decay)

    # AMP
    use_amp, amp_dtype = _amp_setup(cfg.device, cfg.dtype)
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)

    # Resume
    global_step = 0
    best_val = float("inf")
    if cfg.resume_ckpt:
        print(f"[info] resuming from {cfg.resume_ckpt}")
        global_step, best_val, _ = load_full_resume(model, opt, cfg.resume_ckpt, map_location=cfg.device)

    # Steps / epochs planning
    batches_per_epoch = math.ceil(len(train_ds) / cfg.batch_size)
    plan_steps = cfg.steps if cfg.steps > 0 else int(math.ceil(batches_per_epoch * cfg.epochs))
    stride = train_ds.stride; keep = train_ds.keep
    effective_batch = cfg.batch_size * cfg.accum_steps
    print(f"[info] tokens(train)={train_ids.numel():,} | windows={len(train_ds):,} | "
          f"stride={stride} (keep={keep}/{cfg.seq_len}, {100.0*keep/cfg.seq_len:.1f}%) | "
          f"batches/epoch≈{batches_per_epoch:,} | steps={plan_steps:,} (epochs≈{plan_steps/max(1,batches_per_epoch):.3f}) | "
          f"effective batch={effective_batch}")

    # Train loop
    t0 = time.time()
    train_iter = cycle(train_dl)

    # Initial zero grad
    opt.zero_grad(set_to_none=True)

    while global_step < plan_steps:
        model.train()
        batch = next(train_iter)

        # LR schedule
        lr = cosine_lr(global_step, plan_steps, cfg.warmup, cfg.lr, cfg.min_lr)
        for pg in opt.param_groups: pg["lr"] = lr

        inp = batch["input_ids"].to(cfg.device)
        tgt = batch["labels"].to(cfg.device)
        mask = causal_mask(inp.size(0), inp.size(1), cfg.device)

        with torch.amp.autocast('cuda', enabled=use_amp, dtype=amp_dtype):
            logits = model(inp, mask)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                tgt.view(-1)
            )
            # Scale loss for accumulation
            loss = loss / cfg.accum_steps

        scaler.scale(loss).backward()
        
        # Accumulation step: only update weights every accum_steps
        if ((global_step + 1) % cfg.accum_steps == 0) or ((global_step + 1) == plan_steps):
            if cfg.grad_clip is not None:
                if use_amp:
                    scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            
            if use_amp:
                scaler.step(opt)
                scaler.update()
            else:
                opt.step()

            opt.zero_grad(set_to_none=True)

        global_step += 1

        if (global_step % 10 == 0) or (global_step == 1):
            toks = cfg.batch_size * cfg.seq_len * 10
            dt = time.time() - t0; t0 = time.time()
            # Note: loss.item() is the scaled loss from the last micro-batch
            print(f"step {global_step:6d} | loss {loss.item() * cfg.accum_steps:.4f} | lr {lr:.2e} | ~tok/s {int(toks/max(dt,1e-6))}")

        if (global_step % cfg.eval_every == 0) or (global_step == plan_steps):
            val_loss = run_eval(model, val_dl, cfg.device, use_amp, amp_dtype)
            print(f"[eval] step {global_step} | val_loss {val_loss:.4f}")
            if val_loss < best_val:
                best_val = val_loss
                ckpt_path = os.path.join(cfg.out_dir, f"model_step{global_step}.pt")
                save_checkpoint(ckpt_path, model, opt, cfg, global_step, best_val)
                print(f"saved {ckpt_path}")

    # Optional fused export for inference
    if cfg.fast_export:
        model.eval()
        swap_to_fast(model, ffn_only=cfg.ffn_only_fast_export)
        exp_path = os.path.join(cfg.out_dir, "model_fused_infer.pt")
        torch.save({"model": model.state_dict(), "config": asdict(cfg), "vocab_size": ByteTokenizer.vocab_size}, exp_path)
        print(f"Exported fused-inference weights to {exp_path}")

# --------------------- CLI ---------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--corpus", type=str, required=True, help="Path to .txt file")
    p.add_argument("--out_dir", type=str, default="ckpts_quat")
    p.add_argument("--d_model", type=int, default=512)
    p.add_argument("--n_layers", type=int, default=8)
    p.add_argument("--n_heads", type=int, default=8)
    p.add_argument("--ffn_mult", type=int, default=4)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--quat_attention", action="store_true", help="Use quaternion Q/K/V/O projections")
    p.add_argument("--quat_rope", action="store_true", help="Use quaternion RoPE (requires --quat_attention)")
    p.add_argument("--qrope_axis", type=str, default="cycle", choices=["cycle","i"], help="Axis policy for Q-RoPE")
    p.add_argument("--qrope_conjugate", action="store_true", help="Use conjugation u*q*u^{-1} instead of left-mul")
    p.add_argument("--rope_base", type=float, default=10000.0, help="Base frequency for RoPE / QRoPE")
    p.add_argument("--qk_norm", action="store_true", help="L2-normalize Q and K per head before dot-product")
    p.add_argument("--attn_logit_scale_init", type=float, default=None, help="If set, creates a learnable scalar logit scale initialized to this value (multiplicative factor = exp(value))")
    p.add_argument("--norm", type=str, default="layernorm", choices=["layernorm","rmsnorm"], help="Block/Final norm type")
    p.add_argument("--seq_len", type=int, default=512)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--accum_steps", type=int, default=8, help="Gradient accumulation. Effective batch size = batch_size * accum_steps")
    p.add_argument("--steps", type=int, default=0, help="If >0, train for exactly this many steps")
    p.add_argument("--epochs", type=float, default=1.0, help="If steps==0, derive total steps from epochs")
    p.add_argument("--eval_every", type=int, default=200)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--min_lr", type=float, default=1e-4)
    p.add_argument("--warmup", type=int, default=200)
    p.add_argument("--weight_decay", type=float, default=0.1)
    p.add_argument("--beta1", type=float, default=0.9)
    p.add_argument("--beta2", type=float, default=0.95)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--amp", action="store_true")
    p.add_argument("--dtype", type=str, default="fp16", choices=["fp32","fp16","bf16"])
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--compile", action="store_true")
    p.add_argument("--train_split", type=float, default=0.9)
    p.add_argument("--sliding_keep_pct", type=float, default=0.14, help="Fraction of previous context kept between windows")
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--pin_memory", action="store_true")
    p.add_argument("--fast_export", action="store_true", help="Export fused FastQuaternionLinear model for inference")
    p.add_argument("--ffn_only_fast_export", action="store_true", help="Only fuse FFN layers on export")
    p.add_argument("--load_weights", type=str, default=None, help="Warm-start: load weights from checkpoint (model key or raw state_dict)")
    p.add_argument("--resume_ckpt", type=str, default=None, help="Resume full state (model+optimizer+step) from checkpoint")
    args = p.parse_args()

    cfg = TrainConfig(
        corpus=args.corpus, out_dir=args.out_dir, d_model=args.d_model, n_layers=args.n_layers,
        n_heads=args.n_heads, ffn_mult=args.ffn_mult, dropout=args.dropout,
        quat_attention=args.quat_attention, quat_rope=args.quat_rope,
        qrope_axis=args.qrope_axis, qrope_conjugate=args.qrope_conjugate, rope_base=args.rope_base,
        qk_norm=args.qk_norm, attn_logit_scale_init=args.attn_logit_scale_init, norm=args.norm,
        seq_len=args.seq_len, batch_size=args.batch_size, accum_steps=args.accum_steps, 
        steps=args.steps, epochs=args.epochs,
        eval_every=args.eval_every, lr=args.lr, min_lr=args.min_lr, warmup=args.warmup,
        weight_decay=args.weight_decay, beta1=args.beta1, beta2=args.beta2, grad_clip=args.grad_clip,
        amp=args.amp, dtype=args.dtype, device=args.device, seed=args.seed, compile=args.compile,
        train_split=args.train_split, sliding_keep_pct=args.sliding_keep_pct,
        num_workers=args.num_workers, pin_memory=args.pin_memory,
        fast_export=args.fast_export, ffn_only_fast_export=args.ffn_only_fast_export,
        load_weights=args.load_weights, resume_ckpt=args.resume_ckpt
    )
    return cfg

if __name__ == "__main__":
    cfg = parse_args()
    train(cfg)