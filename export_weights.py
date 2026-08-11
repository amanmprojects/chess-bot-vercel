#!/usr/bin/env python3
"""Export a trained ChessNet checkpoint to a browser-friendly fp16 blob.

The Vercel deployment is a static site: there is no Python process to run the
model in, so the weights are shipped to the browser and inference runs in the
game's Web Worker (see chess/src/nn.js). This script converts the torch
checkpoint into:

    chess/model.json   manifest: model config + per-tensor name/shape/offset
    chess/model.bin    raw little-endian float16 weights, concatenated

The float16 conversion halves the 22MB float32 weight set to ~11MB with
negligible effect on move quality (the policy head is a softmax; ±5e-4 weight
noise rarely changes the argmax, and never by much).

Run from the project root (any dir is fine, paths are resolved relative to
this file):

    python3 export_weights.py [--ckpt chess-bot/data/ckpt.pt]

Requires torch (CPU is enough). The output files are meant to be committed so
Vercel's build never needs torch.
"""

import argparse
import json
import struct
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_CKPT = HERE / "chess-bot" / "data" / "ckpt.pt"
OUT_JSON = HERE / "chess-bot" / "chess" / "model.json"
OUT_BIN = HERE / "chess-bot" / "chess" / "model.bin"

# Hard defaults matching serve_model.py / train.py; the manifest records the
# actual architecture so the JS side never assumes.
D = 256
N_LAYERS = 7
N_HEADS = 8
MLP_SCALE = 4


def export(ckpt_path, out_json, out_bin):
    import torch  # imported lazily: this script is a dev tool, not deployed

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = ckpt["model"]

    # Validate the architecture by loading into ChessNet before we trust it.
    sys.path.insert(0, str(HERE / "chess-bot"))
    from model import ChessNet

    net = ChessNet(d=D, n_layers=N_LAYERS, n_heads=N_HEADS, mlp_scale=MLP_SCALE)
    net.load_state_dict(sd)
    net.eval()

    tensors = []
    chunks = bytearray()
    offset = 0  # in fp16 elements, not bytes
    for name, t in sd.items():
        shape = list(t.shape)
        flat = t.detach().to(torch.float16).contiguous().view(-1)
        chunks += flat.numpy().tobytes()
        n = flat.numel()
        tensors.append({"name": name, "shape": shape, "offset": offset, "len": n})
        offset += n

    manifest = {
        "format": 1,
        "d": D,
        "n_layers": N_LAYERS,
        "n_heads": N_HEADS,
        "mlp_scale": MLP_SCALE,
        "step": int(ckpt.get("step", -1)),
        "best_acc": ckpt.get("best_acc"),
        "params": sum(t["len"] for t in tensors),
        "tensors": tensors,
    }

    out_json.write_text(json.dumps(manifest, indent=1) + "\n")
    out_bin.write_bytes(bytes(chunks))

    total_mb = len(chunks) / 1e6
    print(f"exported {manifest['params']:,} params ({total_mb:.1f} MB fp16) "
          f"step {manifest['step']} best_acc {manifest['best_acc']}")
    print(f"  -> {out_json}")
    print(f"  -> {out_bin}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", default=str(DEFAULT_CKPT))
    args = ap.parse_args()
    export(Path(args.ckpt), OUT_JSON, OUT_BIN)


if __name__ == "__main__":
    main()
