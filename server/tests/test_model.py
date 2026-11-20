"""The PyTorch twin must compute what the firmware computes, and training must
actually teach the edge model (the self-learning claim, tested)."""

from __future__ import annotations

import json
import math
import random

import numpy as np
import pytest

from app.evolution import image_codec
from app.evolution.model_spec import SPEC
from app.evolution.pipeline import gate
from app.evolution.prompt_format import build_prompt
from app.evolution.teacher import DeviceContext, RuleTeacher
from app.evolution.tokenizer import BpeTokenizer
from app.evolution.trainer import Example, TorchTrainer, TrainConfig

from .conftest import THERMO_MANIFEST, TOKENIZER

torch = pytest.importorskip("torch")


def firmware_forward(w: dict, tokens: list[int]) -> np.ndarray:
    """NumPy transcription of gemma4_engine.h (token-by-token with a KV cache)."""
    H, nh, hd = SPEC.hidden, SPEC.heads, SPEC.head_dim
    f32 = np.float32

    def rmsnorm(x, weight):
        return x * f32(1.0 / math.sqrt(float(np.sum(x * x, dtype=f32)) / H + 1e-6)) * (f32(1.0) + weight)

    K = [[] for _ in range(SPEC.layers)]
    V = [[] for _ in range(SPEC.layers)]
    x = None
    for pos, tok in enumerate(tokens):
        x = (w["embed"][tok] * f32(SPEC.embed_scale)).astype(f32)
        for l in range(SPEC.layers):
            p = f"layers.{l}."
            h = rmsnorm(x, w[p + "attn_norm"])
            q, k, v = w[p + "wq"] @ h, w[p + "wk"] @ h, w[p + "wv"] @ h
            for head in range(nh):
                for i in range(0, hd, 2):
                    theta = pos * (1.0 / 10000.0 ** (i / hd))
                    c, s = math.cos(theta), math.sin(theta)
                    for vec in (q, k):
                        a, b = vec[head * hd + i], vec[head * hd + i + 1]
                        vec[head * hd + i], vec[head * hd + i + 1] = a * c - b * s, a * s + b * c
            K[l].append(k)
            V[l].append(v)
            att = np.zeros(H, f32)
            for head in range(nh):
                sl = slice(head * hd, (head + 1) * hd)
                scores = np.array([q[sl] @ kk[sl] * SPEC.query_scale for kk in K[l]], f32)
                e = np.exp(scores - scores.max())
                att[sl] = sum(wt * vv[sl] for wt, vv in zip(e / e.sum(), V[l]))
            x = x + w[p + "wo"] @ att
            h = rmsnorm(x, w[p + "ffn_norm"])
            g, u = w[p + "w_gate"] @ h, w[p + "w_up"] @ h
            x = x + w[p + "w_down"] @ (g / (1.0 + np.exp(-1.702 * g)) * u)
    return w["embed"] @ rmsnorm(x, w["final_norm"])


def test_torch_twin_matches_the_firmware_math(base_image):
    from app.evolution.model import EdgeGemma

    w = image_codec.decode_image(base_image)
    w["layers.0.attn_norm"] = np.linspace(-0.5, 0.5, SPEC.hidden).astype(np.float32)   # exercise (1 + w)
    tokens = [1, 300, 1200, 77, 3000, 15]
    expected = firmware_forward(w, tokens)
    got = EdgeGemma(w)(torch.tensor([tokens]))[0, -1].detach().numpy()
    assert np.allclose(got, expected, rtol=1e-3, atol=1e-4), np.max(np.abs(got - expected))


def _thermo_dataset(n: int, seed: int) -> list[Example]:
    rng = random.Random(seed)
    dev = DeviceContext.from_row({"device_key": "thermo-01", "identified": True, "capability_manifest": THERMO_MANIFEST})
    teacher = RuleTeacher()
    out = []
    for _ in range(n):
        ev = json.dumps({"ev": "temp", "v": round(rng.uniform(0, 45), 1)}, separators=(",", ":"))
        out.append(Example(build_prompt("thermo-01", "thermostat", ev), teacher.label(dev, ev).target, THERMO_MANIFEST))
    return out


@pytest.mark.slow
def test_training_teaches_the_edge_model_and_survives_int16(base_image):
    tok = BpeTokenizer.from_file(TOKENIZER)
    train, val = _thermo_dataset(400, seed=1), _thermo_dataset(40, seed=2)
    cfg = TrainConfig(steps=400, batch_size=16, lr=3e-3, freeze_embeddings=True, eval_limit=40)
    trainer = TorchTrainer()
    result = trainer.train(image_codec.decode_image(base_image), train, val, tok, cfg)
    print("baseline", result.baseline, "\ncandidate", result.metrics)

    assert result.metrics["exact_match"] >= result.baseline["exact_match"] + 0.5
    assert result.metrics["valid_action_rate"] >= 0.9
    assert gate(result.baseline, result.metrics, 0.0, force=False)[0]

    # What ships is the INT16 image; the learned behavior must survive quantization.
    from app.evolution.model import EdgeGemma

    shipped = image_codec.decode_image(image_codec.encode_image(result.weights, 2))
    after = trainer.evaluate(EdgeGemma(shipped), val, tok, cfg)
    assert after["exact_match"] >= result.metrics["exact_match"] - 0.1
