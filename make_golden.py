#!/usr/bin/env python3
"""Generate chess/test/golden.json — ground truth for the JS inference port.

Runs the real torch model (data/ckpt.pt) on a set of positions and records:

    slots:      every legal move's 64x73 slot (validates nn.js moveToSlot)
    inference:  the model's chosen move, value, and cp (validates nn.js forward)

The JS test (chess/test/nn.test.mjs) replays these through the browser port and
must agree. Run from the repo root:

    python3 make_golden.py [--ckpt chess-bot/data/ckpt.pt] [--val chess-bot/data/val.bin]
"""

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "chess-bot"))

from features import DTYPE, RECORD_BYTES, board_to_features, features_to_board, move_to_slot  # noqa: E402
from model import ChessNet  # noqa: E402
from play import pick_move  # noqa: E402

# Positions chosen to exercise: black to move, castling rights, en passant,
# promotions, and quiet endgames.
HAND_PICKED = [
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3",
    "rnbqkbnr/ppp1p1pp/8/3pPp2/8/8/PPPP1PPP/RNBQKBNR w KQkq f6 0 3",   # en passant
    "r3k2r/ppp2ppp/8/8/8/8/PPP2PPP/R3K2R w KQkq - 0 1",                 # castling both sides
    "rnbqkbnr/ppp1pppp/8/3p4/8/8/PPPPPPPP/RNBQKBNR b KQkq - 0 1",       # black to move
    "8/5K1k/8/8/8/8/4P3/8 w - - 0 1",                                    # pawn endgame
    "8/2P5/8/8/8/8/7k/6K1 w - - 0 1",                                    # promo race
    "8/6k1/8/8/8/5p2/6PP/7K w - - 0 1",                                  # race to stop promo
    "4k3/8/8/8/8/8/4P1P1/4K3 w - - 0 1",
    "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 10",
]


def value_to_cp(value, eval_scale=None):
    v = max(min(float(value), 0.999), -0.999)
    if eval_scale is not None:
        return int(v * eval_scale)
    # Side-to-move view (no White-flip): the JS port and renderEval handle
    # the conversion to White's view using the recorded mover.
    return int(-400 * math.log10(2 / (v + 1) - 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(HERE / "chess-bot" / "data" / "ckpt.pt"))
    ap.add_argument("--val", default=str(HERE / "chess-bot" / "data" / "val.bin"))
    ap.add_argument("--n-sampled", type=int, default=14)
    ap.add_argument("--out", default=str(HERE / "chess-bot" / "chess" / "test" / "golden.json"))
    args = ap.parse_args()

    import chess
    import torch

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    net = ChessNet(d=int(ck.get("d", 256)), n_layers=int(ck.get("n_layers", 7)),
                   n_heads=int(ck.get("n_heads", 8))).to("cpu")
    net.load_state_dict(ck["model"])
    net.eval()
    eval_scale = ck.get("eval_scale")

    # A few positions from the validation set for realism.
    fens = list(HAND_PICKED)
    val = Path(args.val)
    if val.exists():
        size = val.stat().st_size
        n = size // RECORD_BYTES
        mmap = np.memmap(val, dtype=DTYPE, mode="r", shape=(n,))
        rng = np.random.default_rng(7)
        for i in rng.choice(n, size=args.n_sampled, replace=False):
            board = features_to_board(mmap[i]["pieces"], mmap[i]["aux"])
            if not board.is_game_over():
                fens.append(board.fen())

    # Drop anything the model would refuse (no legal moves at all).
    fens = [f for f in fens if chess.Board(f).legal_moves.count() > 0]

    slots = []
    inference = []
    for fen in fens:
        board = chess.Board(fen)
        legal = list(board.legal_moves)
        slots.append({"fen": fen, "moves": [[m.uci(), move_to_slot(board, m)] for m in legal]})
        move, value = pick_move(net, board, torch.device("cpu"), 0.0)
        # Keep only positions where the top choice is not a near coin-flip, so
        # float16 weight noise cannot flip the JS port's argmax.
        top2 = sorted((masked_top2(board, net)), reverse=True)
        margin = top2[0] - top2[1] if len(top2) > 1 else 1e9
        if margin < 0.05:
            continue
        inference.append({
            "fen": fen,
            "uci": move.uci(),
            "value": round(float(value), 6),
            "cp": value_to_cp(value, eval_scale),
            "margin": round(float(margin), 3),
        })

    out = {"slots": slots, "inference": inference}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=1) + "\n")
    print(f"wrote {args.out}: {len(slots)} slot positions, {len(inference)} inference positions")


def masked_top2(board, net):
    """The two largest masked logits, so we can insist on a clear winner."""
    import numpy as np
    import torch
    import torch.nn.functional as F

    from features import policy_mask

    pieces, aux = board_to_features(board)
    p = torch.from_numpy(pieces).long().unsqueeze(0)
    a = torch.from_numpy(aux).long().unsqueeze(0)
    with torch.no_grad():
        logits, _ = net(p, a)
    mask = torch.from_numpy(policy_mask(board))
    masked = logits[0] + torch.where(mask, torch.zeros_like(logits[0]),
                                     torch.full_like(logits[0], -1e9))
    vals, _ = torch.topk(masked, 2)
    return [float(v) for v in vals]


if __name__ == "__main__":
    main()
