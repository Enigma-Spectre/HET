# het_inference.py
# Load a checkpoint from het_train.py and chat in a REPL.
# Byte-level tokenizer, temperature/top-k/top-p sampling, optional fused FastQuaternionLinear.
# Uses torch.amp.autocast('cuda', ...). Includes repetition penalty and sliding_keep_pct.
# RoPE.apply slices cached cos/sin to current S to avoid shape mismatches.

import os, sys, argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

# ===== Byte-level tokenizer =====
class ByteTokenizer:
    vocab_size = 256
    def encode(self, text: str) -> torch.Tensor:
        return torch.tensor(list(text.encode("utf-8")), dtype=torch.long)
    def decode(self, ids: torch.Tensor) -> str:
        return bytes([int(x) for x in ids]).decode("utf-8", errors="ignore")

# ===== RoPE and quaternion layers (subset needed for inference) =====
def _rotate_half(x):
    x_even = x[..., ::2]; x_odd = x[..., 1::2]
    out = torch.empty_like(x); out[..., ::2] = -x_odd; out[..., 1::2] = x_even
    return out

class RoPE(nn.Module):
    def __init__(self, head_dim, base=10000.0):
        super().__init__()
        assert head_dim % 2 == 0
        self.head_dim = head_dim
        self.base = base
        self.max_seq = 0
        self._cos = None
        self._sin = None
        self._dev = None
        self._dt = None

    @torch.no_grad()
    def _maybe(self, S, device, dtype):
        need = (self._cos is None or S > self.max_seq or device != self._dev or dtype != self._dt)
        if not need:
            return
        self.max_seq = S
        self._dev = device
        self._dt = dtype
        t = torch.arange(S, device=device, dtype=dtype)
        freqs = 1.0 / (self.base ** (torch.arange(0, self.head_dim, 2, device=device, dtype=dtype) / self.head_dim))
        ang = torch.einsum("s,d->sd", t, freqs)
        emb = torch.repeat_interleave(ang, 2, dim=-1)
        self._cos = emb.cos()[None, None, :, :]
        self._sin = emb.sin()[None, None, :, :]

    def apply(self, q, k):
        # q,k: [B,H,S,D]
        B, H, S, D = q.shape
        self._maybe(S, q.device, q.dtype)
        # Slice caches to current S to avoid mismatches after truncation or sliding windows
        cos = self._cos[:, :, :S, :].to(device=q.device, dtype=q.dtype)
        sin = self._sin[:, :, :S, :].to(device=q.device, dtype=q.dtype)
        return (q * cos) + (_rotate_half(q) * sin), (k * cos) + (_rotate_half(k) * sin)

class QuaternionLinear(nn.Module):
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
        B, S, _ = x.shape; iq = self.in_quat; oq = self.out_quat
        xr = x[..., 0::4].reshape(B*S, iq)
        xi = x[..., 1::4].reshape(B*S, iq)
        xj = x[..., 2::4].reshape(B*S, iq)
        xk = x[..., 3::4].reshape(B*S, iq)
        WrT, WiT, WjT, WkT = self.wr.t(), self.wi.t(), self.wj.t(), self.wk.t()
        Ar = xr @ WrT; Br = xi @ WrT; Cr = xj @ WrT; Dr = xk @ WrT
        Ai = xr @ WiT; Bi = xi @ WiT; Ci = xj @ WiT; Di = xk @ WiT
        Aj = xr @ WjT; Bj = xi @ WjT; Cj = xj @ WjT; Dj = xk @ WjT
        Ak = xr @ WkT; Bk = xi @ WkT; Ck = xj @ WkT; Dk = xk @ WkT
        or_ = (Ar - Bi - Cj - Dk)
        oi_ = (Br + Ai + Dk - Cj)
        oj_ = (Cr - Dk + Aj + Bi)
        ok_ = (Dr + Cj - Bi + Ak)
        out = torch.empty(B*S, 4*oq, device=x.device, dtype=x.dtype)
        out[:, 0::4] = or_; out[:, 1::4] = oi_; out[:, 2::4] = oj_; out[:, 3::4] = ok_
        out = out.view(B, S, 4*oq)
        if self.bias is not None:
            out = out + self.bias.view(1, 1, -1)
        return out

class FastQuaternionLinear(QuaternionLinear):
    def __init__(self, in_features, out_features, bias=True, cache_block_in_eval=True):
        super().__init__(in_features, out_features, bias)
        self.cache_block_in_eval = cache_block_in_eval
        self._W = None; self._dev = None; self._dt = None

    def train(self, mode: bool = True):
        self._W = None
        return super().train(mode)

    @torch.no_grad()
    def _assemble(self, device, dtype):
        Wr, Wi, Wj, Wk = self.wr.to(device, dtype), self.wi.to(device, dtype), self.wj.to(device, dtype), self.wk.to(device, dtype)
        top  = torch.cat([ Wr, -Wi, -Wj, -Wk], dim=1)
        row2 = torch.cat([ Wi,  Wr,  Wk, -Wj], dim=1)
        row3 = torch.cat([ Wj, -Wk,  Wr,  Wi], dim=1)
        row4 = torch.cat([ Wk,  Wj, -Wi,  Wr], dim=1)
        return torch.cat([top, row2, row3, row4], dim=0)  # [4*out_q, 4*in_q]

    def _getW(self, device, dtype):
        if not self.cache_block_in_eval or self.training:
            return self._assemble(device, dtype)
        if self._W is None or self._dev != device or self._dt != dtype:
            self._W = self._assemble(device, dtype); self._dev = device; self._dt = dtype
        return self._W

    def forward(self, x):
        B, S, _ = x.shape; iq = self.in_quat
        X = torch.cat([x[..., 0::4], x[..., 1::4], x[..., 2::4], x[..., 3::4]], dim=-1)  # [B,S,4*in_q]
        W = self._getW(x.device, x.dtype)  # [4*out_q, 4*in_q]
        Y = X.reshape(B*S, 4*iq) @ W.t()
        Y = Y.view(B, S, -1)
        if self.bias is not None:
            Y = Y + self.bias.view(1, 1, -1)
        return Y

# ===== Attention/FFN/minimal model (matches train script) =====
class StandardAttention(nn.Module):
    def __init__(self, dim, n_heads=8, dropout=0.0, rope_base=10000.0):
        super().__init__()
        assert dim % n_heads == 0
        self.dim = dim; self.n_heads = n_heads; self.hd = dim // n_heads; self.scale = self.hd ** -0.5
        self.q = nn.Linear(dim, dim); self.k = nn.Linear(dim, dim); self.v = nn.Linear(dim, dim); self.o = nn.Linear(dim, dim)
        self.drop = nn.Dropout(dropout); self.rope = RoPE(self.hd, rope_base)

    def forward(self, x, mask=None):
        B, S, Dm = x.shape; H = self.n_heads; Dh = self.hd
        q = self.q(x).view(B, S, H, Dh).transpose(1, 2)
        k = self.k(x).view(B, S, H, Dh).transpose(1, 2)
        v = self.v(x).view(B, S, H, Dh).transpose(1, 2)
        q, k = self.rope.apply(q, k)
        scores = torch.einsum("bhsd,bhtd->bhst", q, k) * self.scale
        if mask is not None:
            scores = scores.masked_fill(mask[:, None, :, :], float("-inf"))
        a = self.drop(F.softmax(scores, dim=-1))
        out = torch.einsum("bhst,bhtd->bhsd", a, v).transpose(1, 2).contiguous().view(B, S, Dm)
        return self.o(out)

class QAttention(nn.Module):
    def __init__(self, dim, n_heads=8, dropout=0.0, linear_cls=QuaternionLinear):
        super().__init__()
        assert dim % n_heads == 0
        hd = dim // n_heads; assert hd % 4 == 0
        self.dim = dim; self.n_heads = n_heads; self.hd = hd; self.scale = hd ** -0.5
        self.q = linear_cls(dim, dim); self.k = linear_cls(dim, dim); self.v = linear_cls(dim, dim); self.o = linear_cls(dim, dim)
        self.drop = nn.Dropout(dropout); self.rope = RoPE(self.hd)

    def forward(self, x, mask=None):
        B, S, Dm = x.shape; H = self.n_heads; Dh = self.hd
        q = self.q(x).view(B, S, H, Dh).transpose(1, 2)
        k = self.k(x).view(B, S, H, Dh).transpose(1, 2)
        v = self.v(x).view(B, S, H, Dh).transpose(1, 2)
        q, k = self.rope.apply(q, k)
        scores = torch.einsum("bhsd,bhtd->bhst", q, k) * self.scale
        if mask is not None:
            scores = scores.masked_fill(mask[:, None, :, :], float("-inf"))
        a = self.drop(F.softmax(scores, dim=-1))
        out = torch.einsum("bhst,bhtd->bhsd", a, v).transpose(1, 2).contiguous().view(B, S, Dm)
        return self.o(out)

class QuatFFN(nn.Module):
    def __init__(self, dim, mult=4, dropout=0.0, linear_cls=QuaternionLinear):
        super().__init__()
        hid = dim * mult; assert hid % 4 == 0
        self.fc1 = linear_cls(dim, hid); self.fc2 = linear_cls(hid, dim)
        self.act = nn.GELU(); self.drop = nn.Dropout(dropout)
    def forward(self, x):
        x = self.fc1(x); x = self.drop(self.act(x)); return self.fc2(x)

class HETBlock(nn.Module):
    def __init__(self, dim, n_heads, dropout, mult, quat_attention=False, linear_cls=QuaternionLinear):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim); self.norm2 = nn.LayerNorm(dim); self.drop = nn.Dropout(dropout)
        self.attn = (QAttention(dim, n_heads, dropout, linear_cls=linear_cls)
                     if quat_attention else StandardAttention(dim, n_heads, dropout))
        self.ffn = QuatFFN(dim, mult, dropout, linear_cls=linear_cls)
    def forward(self, x, mask):
        x = x + self.drop(self.attn(self.norm1(x), mask))
        x = x + self.drop(self.ffn(self.norm2(x)))
        return x

class CausalLM(nn.Module):
    def __init__(self, vocab_size, d_model=512, n_layers=8, n_heads=8,
                 dropout=0.0, ffn_mult=4, quat_attention=False, tie_weights=True,
                 linear_cls=QuaternionLinear):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([
            HETBlock(d_model, n_heads, dropout, ffn_mult, quat_attention, linear_cls=linear_cls)
            for _ in range(n_layers)
        ])
        self.norm_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        if tie_weights:
            self.lm_head.weight = self.tok_emb.weight
    def forward(self, input_ids, mask=None):
        x = self.drop(self.tok_emb(input_ids))
        for b in self.blocks:
            x = b(x, mask)
        x = self.norm_f(x)
        return self.lm_head(x)

# ===== helpers =====
def causal_mask(B, S, device):
    m = torch.ones(S, S, dtype=torch.bool, device=device).triu(1)
    return m[None, :, :].expand(B, -1, -1)

def _autocast_dtype_and_flag(device_str: str, requested: torch.dtype):
    """Return (enabled, dtype) for torch.amp.autocast('cuda', ...)."""
    if not device_str.startswith("cuda"):
        return False, None
    # BF16 fallback on T4 and similar
    want_bf16 = (requested == torch.bfloat16)
    try:
        bf16_ok = torch.cuda.is_bf16_supported()
    except Exception:
        bf16_ok = False
    if want_bf16 and not bf16_ok:
        print("[warn] bf16 not supported on this GPU; using fp16 autocast instead.", file=sys.stderr)
        return True, torch.float16
    if requested == torch.float16:
        return True, torch.float16
    # fp32 => disable autocast
    return False, None

# ===== Sampling =====
@torch.no_grad()
def sample(model, ids, max_new_tokens, temperature=1.0, top_k=0, top_p=1.0,
           device="cuda", amp_dtype: Optional[torch.dtype]=None, amp_enabled: bool=False,
           repetition_penalty: float = 1.0, eos_id: Optional[int]=None, max_seq_len=2048):
    model.eval()
    for _ in range(max_new_tokens):
        # Context window truncate (left)
        if ids.size(1) > max_seq_len:
            ids = ids[:, -max_seq_len:]
        B, S = ids.shape
        mask = torch.ones(S, S, dtype=torch.bool, device=device).triu(1)[None, :, :].expand(B, -1, -1)

        with torch.amp.autocast('cuda', enabled=amp_enabled, dtype=amp_dtype):
            logits = model(ids, mask=mask)[:, -1, :]  # [B,V]

        # simple repetition penalty (batch=1)
        if repetition_penalty != 1.0 and S > 0:
            seen = torch.bincount(ids[0], minlength=logits.size(-1)).to(logits.dtype)
            logits = logits / (1.0 + (repetition_penalty - 1.0) * (seen > 0))

        logits = logits / max(temperature, 1e-6)

        # top-k
        if top_k > 0:
            k = min(top_k, logits.size(-1))
            vals, idx = torch.topk(logits, k=k)
            filt = torch.full_like(logits, float('-inf'))
            filt.scatter_(1, idx, vals)
            logits = filt

        # top-p
        if top_p < 1.0:
            sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
            probs = torch.softmax(sorted_logits, dim=-1)
            cum = torch.cumsum(probs, dim=-1)
            mask_p = cum > top_p
            mask_p[:, 0] = False
            sorted_logits[mask_p] = float('-inf')
            logits = torch.gather(sorted_logits, 1, torch.argsort(sorted_idx, dim=-1))

        probs = torch.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)  # [B,1]
        ids = torch.cat([ids, next_id], dim=1)

        if eos_id is not None and int(next_id[0, 0]) == eos_id:
            break
    return ids

# ===== Checkpoint load =====
def load_checkpoint(ckpt_path, device="cuda"):
    obj = torch.load(ckpt_path, map_location=device)
    if "model" not in obj or "config" not in obj:
        raise SystemExit("Checkpoint missing required keys: 'model' and 'config'.")
    return obj["model"], obj["config"]

def build_model_from_config(cfg_dict, fast=False):
    # cfg_dict was produced by TrainConfig.asdict()
    d_model = cfg_dict.get("d_model", 512)
    n_layers = cfg_dict.get("n_layers", 8)
    n_heads = cfg_dict.get("n_heads", 8)
    dropout = cfg_dict.get("dropout", 0.0)
    ffn_mult = cfg_dict.get("ffn_mult", 4)
    quat_attention = cfg_dict.get("quat_attention", False)
    vocab_size = 256  # ByteTokenizer
    linear_cls = FastQuaternionLinear if fast else QuaternionLinear
    return CausalLM(vocab_size, d_model, n_layers, n_heads, dropout, ffn_mult,
                    quat_attention=quat_attention, linear_cls=linear_cls)

# ===== REPL driver =====
def run_repl(model, tok: ByteTokenizer, device, requested_dtype,
             max_seq_len, max_new, temperature, top_k, top_p,
             repetition_penalty,
             sys_prefix: str, user_prefix: str, asst_prefix: str, stop_str: str,
             sliding_keep_pct: float):
    print("Enter text. Ctrl+C or /quit to exit.")
    history = ""  # raw byte-level transcript

    amp_enabled, amp_dtype = _autocast_dtype_and_flag(device, requested_dtype)

    def encode_history_with_sliding(h: str, new_user: str):
        # Keep a fraction of previous history to reduce runaway context (pre-encode trim).
        if not h:
            h2 = ""
        else:
            if sliding_keep_pct >= 1.0:
                h2 = h
            else:
                keep = int(len(h) * max(0.0, min(0.95, sliding_keep_pct)))
                h2 = h[-keep:] if keep > 0 else ""
        if sys_prefix and not h2:
            h2 += f"{sys_prefix}\n"
        h2 += f"{user_prefix}{new_user}\n{asst_prefix}"
        ids = tok.encode(h2).unsqueeze(0).to(device)
        return ids, h2

    while True:
        try:
            user = input("> ").rstrip("\n")
        except KeyboardInterrupt:
            print("\nbye."); return
        if user.strip() in {"/quit", "/exit"}:
            print("bye."); return

        ids, history = encode_history_with_sliding(history, user)

        out = sample(
            model, ids, max_new_tokens=max_new, temperature=temperature,
            top_k=top_k, top_p=top_p, device=device,
            amp_dtype=amp_dtype, amp_enabled=amp_enabled,
            repetition_penalty=repetition_penalty, max_seq_len=max_seq_len
        )
        text = tok.decode(out[0].tolist())

        # Take only the new tail after the assistant prefix
        start = text.rfind(asst_prefix)
        gen = text[start+len(asst_prefix):] if start != -1 else text
        stop_pos = len(gen)
        if stop_str:
            p = gen.find(stop_str)
            if p != -1:
                stop_pos = p
        gen_piece = gen[:stop_pos]
        print(gen_piece.strip())

        # Append assistant output + newline to full history buffer
        history += gen_piece + "\n"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True, help="Path to checkpoint .pt from het_train.py")
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--dtype", type=str, default="fp16", choices=["fp32", "fp16", "bf16"])
    ap.add_argument("--fast", action="store_true", help="Use FastQuaternionLinear modules")
    ap.add_argument("--max_seq_len", type=int, default=2048)
    ap.add_argument("--max_new", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top_k", type=int, default=50)
    ap.add_argument("--top_p", type=float, default=0.9)
    ap.add_argument("--repetition_penalty", type=float, default=1.1)
    ap.add_argument("--sliding_keep_pct", type=float, default=1.0, help="Keep this fraction of prior history each turn (<=1.0).")
    ap.add_argument("--prompt", type=str, default=None, help="Optional one-shot prompt then exit (no REPL)")
    ap.add_argument("--sys_prefix", type=str, default="You are a helpful model.")
    ap.add_argument("--user_prefix", type=str, default="User: ")
    ap.add_argument("--asst_prefix", type=str, default="Assistant: ")
    ap.add_argument("--stop_str", type=str, default="\nUser: ")
    args = ap.parse_args()

    dtype_map = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}
    requested_dtype = dtype_map[args.dtype]

    # Load
    state, cfg = load_checkpoint(args.ckpt, device=args.device)
    model = build_model_from_config(cfg, fast=args.fast).to(args.device)
    # Keep weights FP32 for stability; autocast handles compute dtypes.
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        print(f"[warn] missing keys: {len(missing)}, unexpected keys: {len(unexpected)}", file=sys.stderr)

    tok = ByteTokenizer()

    # One-shot mode
    if args.prompt is not None:
        amp_enabled, amp_dtype = _autocast_dtype_and_flag(args.device, requested_dtype)
        ids = tok.encode(args.prompt).unsqueeze(0).to(args.device)
        out = sample(model, ids, max_new_tokens=args.max_new, temperature=args.temperature,
                     top_k=args.top_k, top_p=args.top_p, device=args.device,
                     amp_dtype=amp_dtype, amp_enabled=amp_enabled,
                     repetition_penalty=args.repetition_penalty, max_seq_len=args.max_seq_len)
        print(tok.decode(out[0].tolist()))
        return

    # REPL
    run_repl(model, tok, args.device, requested_dtype,
             args.max_seq_len, args.max_new, args.temperature, args.top_k, args.top_p,
             args.repetition_penalty,
             args.sys_prefix, args.user_prefix, args.asst_prefix, args.stop_str,
             max(0.0, min(1.0, float(args.sliding_keep_pct))))

if __name__ == "__main__":
    main()
