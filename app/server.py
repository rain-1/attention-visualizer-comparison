import os
from functools import lru_cache
from typing import Dict, List

import torch
from flask import Flask, jsonify, request, send_from_directory

APP_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(APP_DIR, "static")
DEFAULT_MODEL_A = os.environ.get("MODEL_A_PATH", os.path.join(APP_DIR, "data", "model_a.pt"))
DEFAULT_MODEL_B = os.environ.get("MODEL_B_PATH", os.path.join(APP_DIR, "data", "model_b.pt"))

app = Flask(__name__, static_folder=STATIC_DIR, static_url_path="")


class AttentionBundle:
    def __init__(self, path: str) -> None:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Could not find attention file at {path}")
        payload = torch.load(path, map_location="cpu")
        if "attention" not in payload:
            raise KeyError(f"Attention tensor missing in {path}")
        self.tokens: List[str] = payload.get("tokens")
        self.prompt: str = payload.get("prompt", "")
        self.num_layers: int = payload.get("num_layers") or payload["attention"].shape[0]
        self.num_heads: int = payload.get("num_heads") or payload["attention"].shape[1]
        attention_tensor = payload["attention"].float()
        if attention_tensor.ndim != 4:
            raise ValueError("Attention tensor must have shape [layers, heads, tokens, tokens]")
        self.attention = attention_tensor
        self.tokens = list(self.tokens) if self.tokens is not None else [str(i) for i in range(attention_tensor.shape[-1])]

    def get_slice(self, layer: int, head: int) -> List[List[float]]:
        if not (0 <= layer < self.attention.shape[0]):
            raise IndexError("Layer index out of range")
        if not (0 <= head < self.attention.shape[1]):
            raise IndexError("Head index out of range")
        matrix = self.attention[layer, head].detach().cpu().tolist()
        return matrix


@lru_cache(maxsize=1)
def load_models() -> Dict[str, AttentionBundle]:
    model_a = AttentionBundle(DEFAULT_MODEL_A)
    model_b = AttentionBundle(DEFAULT_MODEL_B)
    if model_a.num_layers != model_b.num_layers:
        raise ValueError("Attention bundles must share the same number of layers")
    if model_a.num_heads != model_b.num_heads:
        raise ValueError("Attention bundles must share the same number of heads")
    return {"A": model_a, "B": model_b}


@app.route("/")
def index():
    return send_from_directory(STATIC_DIR, "index.html")


@app.route("/api/metadata")
def metadata():
    models = load_models()
    model_a = models["A"]
    model_b = models["B"]
    return jsonify({
        "num_layers": model_a.num_layers,
        "num_heads": model_a.num_heads,
        "models": {
            "A": {
                "tokens": model_a.tokens,
                "prompt": model_a.prompt,
            },
            "B": {
                "tokens": model_b.tokens,
                "prompt": model_b.prompt,
            },
        },
    })


@app.route("/api/attention")
def attention():
    layer = request.args.get("layer", type=int)
    head = request.args.get("head", type=int)
    if layer is None or head is None:
        return jsonify({"error": "layer and head query parameters are required"}), 400

    models = load_models()
    try:
        matrix_a = models["A"].get_slice(layer, head)
        matrix_b = models["B"].get_slice(layer, head)
    except (IndexError, ValueError) as exc:
        return jsonify({"error": str(exc)}), 400

    return jsonify({
        "layer": layer,
        "head": head,
        "model_a": matrix_a,
        "model_b": matrix_b,
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8000"))
    app.run(host="0.0.0.0", port=port, debug=True)
