# Chess Bot on Vercel

This directory is the deployable project. It contains the training repo
(`chess-bot/`) with the web UI (`chess-bot/chess/`, a git submodule), plus the
Vercel plumbing: `vercel.json`, `.vercelignore`, and the model-export tooling.

## How it works

The old setup ran the 5.58M-param PyTorch model in a local Python process
(`serve_model.py`) and the UI called it over HTTP — impossible on Vercel,
which has no persistent Python runtime.

Instead, the checkpoint is converted to a compact float16 blob and the
transformer forward pass is ported to pure JavaScript:

- `export_weights.py` — converts a checkpoint (67MB+ float32 +
  optimizer state) into committed files per model:
  - `chess-bot/chess/model.json` / `model.bin` (11.2MB fp16): the original
    5.58M-param net (2.6M records, game-result value head)
  - `chess-bot/chess/model2.json` / `model2.bin` (19.8MB fp16): the 9.92M-param
    net (12.5M records, Stockfish-eval value head, top-1 51.1% vs 43.2%)
  Pass `--ckpt`, `--out-json` and `--out-bin` to export a different
  checkpoint; architecture is read from the checkpoint itself.
- `chess-bot/chess/src/nn.js` — a dependency-free port of `model.py`: board →
  features → 7-block prenorm transformer → masked policy → best move + value.
  Numerics mirror the Python code exactly (erf GELU, LayerNorm eps 1e-5, same
  centipawn conversion).
- The game's Web Worker (`src/worker.js`) runs inference, so the UI thread
  never blocks. The model downloads once in the background (~11MB) and one
  forward pass takes ~0.4s.

The whole deployment is static files — no server, no build step, no cost.

## Deploy

Via the CLI (from this directory):

    npx vercel --prod

Or connect this directory as a git repo to Vercel and push — `vercel.json`
tells Vercel to serve `chess-bot/chess/` with no build. `.vercelignore` keeps
the ~700MB of training data and checkpoints out of the upload.

> Note: `chess-bot/chess/` is a git submodule. Commit the new files
> (`model.bin`, `model.json`, `src/nn.js`, `test/nn.test.mjs`, ...) inside the
> submodule repo as well, or the deployment will be missing them.

## Regenerating the model files

The weights are committed, so no build needs Python. To re-export after
training a new checkpoint:

    python3 export_weights.py --ckpt chess-bot/data/ckpt.pt   # needs torch (CPU ok)

Then regenerate the golden test data and run the full suite:

    python3 make_golden.py            # needs torch + python-chess + numpy
    cd chess-bot/chess && npm test    # 56 tests incl. JS-vs-torch parity

`test/nn.test.mjs` replays golden.json / golden2.json (produced by the real
torch models) through the JS port: every legal move's policy slot must match,
and the chosen move/value/cp must agree. This is what guarantees the browser
models behave like the originals.

## Local development

Unchanged: `cd chess-bot/chess && node server.mjs` serves the UI on
`http://localhost:8000`. The "Neural net" level now works with no Python
server at all.

The old `serve_model.py` still exists in `chess-bot/` for reference, but the
UI no longer calls it.
