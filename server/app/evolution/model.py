"""PyTorch twin of the firmware forward pass (gemma4/gemma4_engine.h).

Every operation matches the C++ exactly, so weights trained here behave the same
on the ESP32-S3 after INT16 quantization (tests/test_model_parity.py checks it
against a NumPy re-implementation of the firmware):

  x      = E[token] * sqrt(H)                         (tied table E)
  layer: h = rmsnorm(x, attn_norm)        rmsnorm(x,w) = x / sqrt(mean(x^2) + 1e-6) * (1 + w)
         q,k,v = Wq h, Wk h, Wv h ; RoPE on (q,k), interleaved pairs (2i, 2i+1),
                                    theta = pos / 10000^(2i/head_dim)
         attn  = softmax(q.k * 0.125, causal) . v ;  x += Wo attn
         h = rmsnorm(x, ffn_norm) ; g = Wg h ; u = Wu h
         x += Wd (g * sigmoid(1.702 g) * u)
  logits = rmsnorm(x, final_norm) . E^T

Requires torch (requirements-train.txt). Only imported by the torch trainer.
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .image_codec import LAYER_MATRICES, Weights, weight_shapes
from .model_spec import SPEC, ModelSpec


def rmsnorm(x: torch.Tensor, w: torch.Tensor, eps: float) -> torch.Tensor:
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * (1.0 + w)


def rope_tables(spec: ModelSpec, T: int, device=None) -> tuple[torch.Tensor, torch.Tensor]:
    i = torch.arange(0, spec.head_dim, 2, dtype=torch.float32, device=device)
    freq = 1.0 / torch.pow(torch.tensor(spec.rope_theta, device=device), i / spec.head_dim)
    theta = torch.arange(T, dtype=torch.float32, device=device)[:, None] * freq[None, :]
    return torch.cos(theta), torch.sin(theta)            # [T, head_dim/2]


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    # x: [B, T, heads, head_dim]; pairs (2i, 2i+1) rotate together
    x0, x1 = x[..., 0::2], x[..., 1::2]
    c, s = cos[None, :, None, :], sin[None, :, None, :]
    out = torch.empty_like(x)
    out[..., 0::2] = x0 * c - x1 * s
    out[..., 1::2] = x0 * s + x1 * c
    return out


class EdgeGemma(nn.Module):
    def __init__(self, weights: Weights, spec: ModelSpec = SPEC) -> None:
        super().__init__()
        self.spec = spec
        shapes = weight_shapes(spec)
        self.params = nn.ParameterDict({k.replace(".", "__"): nn.Parameter(torch.from_numpy(np.array(weights[k], np.float32)))
                                        for k in shapes})
        self.lora_A = nn.ParameterDict()
        self.lora_B = nn.ParameterDict()
        self.lora_scale = 0.0

    # ---- LoRA ----
    def enable_lora(self, rank: int, alpha: float | None = None, seed: int = 0) -> None:
        """Freezes the base weights and adds low-rank adapters on every layer matrix."""
        g = torch.Generator().manual_seed(seed)
        for p in self.params.values():
            p.requires_grad_(False)
        for l in range(self.spec.layers):
            for m in LAYER_MATRICES:
                key = f"layers__{l}__{m}"
                out_dim, in_dim = self.params[key].shape
                self.lora_A[key] = nn.Parameter(torch.randn(rank, in_dim, generator=g) / math.sqrt(in_dim))
                self.lora_B[key] = nn.Parameter(torch.zeros(out_dim, rank))
        self.lora_scale = (alpha or rank) / rank

    def _w(self, key: str) -> torch.Tensor:
        w = self.params[key]
        if key in self.lora_A:
            w = w + self.lora_scale * (self.lora_B[key] @ self.lora_A[key])
        return w

    def export_weights(self) -> Weights:
        """Float weights with any LoRA adapters merged in (ready to quantize)."""
        with torch.no_grad():
            return {k.replace("__", "."): self._w(k).detach().cpu().numpy().astype(np.float32) for k in self.params}

    # ---- forward ----
    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        spec = self.spec
        B, T = tokens.shape
        E = self.params["embed"]
        x = F.embedding(tokens, E) * spec.embed_scale
        cos, sin = rope_tables(spec, T, tokens.device)
        mask = torch.full((T, T), float("-inf"), device=tokens.device).triu(1)
        for l in range(spec.layers):
            p = f"layers__{l}__"
            h = rmsnorm(x, self.params[p + "attn_norm"], spec.norm_eps)
            q = (h @ self._w(p + "wq").T).view(B, T, spec.heads, spec.head_dim)
            k = (h @ self._w(p + "wk").T).view(B, T, spec.heads, spec.head_dim)
            v = (h @ self._w(p + "wv").T).view(B, T, spec.heads, spec.head_dim)
            q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
            scores = torch.einsum("bthd,bshd->bhts", q, k) * spec.query_scale + mask
            att = torch.einsum("bhts,bshd->bthd", scores.softmax(-1), v).reshape(B, T, spec.hidden)
            x = x + att @ self._w(p + "wo").T
            h = rmsnorm(x, self.params[p + "ffn_norm"], spec.norm_eps)
            g = h @ self._w(p + "w_gate").T
            u = h @ self._w(p + "w_up").T
            x = x + (g * torch.sigmoid(1.702 * g) * u) @ self._w(p + "w_down").T
        x = rmsnorm(x, self.params["final_norm"], spec.norm_eps)
        return x @ E.T

    @torch.no_grad()
    def greedy(self, prompt: list[int], max_new: int) -> list[int]:
        """Greedy decode, like gemma_generate_greedy() on the device (EOS excluded)."""
        seq = list(prompt)
        out: list[int] = []
        while len(out) < max_new and len(seq) < self.spec.max_context + 1:
            logits = self.forward(torch.tensor([seq]))[0, -1]
            nxt = int(torch.argmax(logits))
            if nxt == self.spec.eos_id:
                break
            out.append(nxt)
            seq.append(nxt)
        return out
