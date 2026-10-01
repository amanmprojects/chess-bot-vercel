/**
 * Neural level support — runs the trained ChessNet in the browser.
 *
 * The Vercel deployment is a static site, so there is no Python process to
 * host the model. Instead the exported weights (chess/model.bin, 11MB fp16)
 * are downloaded once by the game's Web Worker and the forward pass runs
 * locally in pure JS (see nn.js). Replies are shaped exactly like the
 * worker's `bestmove` / `error` messages so app.js can feed them through the
 * same handler, which already discards stale replies by id and knows how to
 * render hints and evaluation.
 */

import { loadModel, runNeuralMove } from './nn.js';

/** Exported checkpoints, by level id. `neural` is the original 2.6M-record
 *  model; `neural2` the 12.5M-record Stockfish-labelled one. */
export const NEURAL_MODELS = {
  neural: { manifest: 'model.json', bin: 'model.bin' },
  neural2: { manifest: 'model2.json', bin: 'model2.bin' },
};

/** True when this level should be answered by a model instead of ai.js. */
export function isNeuralLevel(level) {
  return level in NEURAL_MODELS;
}

/**
 * Kick off the model download + decode in the background. Safe to call
 * repeatedly; the underlying fetch/parse promise is cached.
 *
 * `onStatus` receives {status:'downloading', loaded, total},
 * {status:'decoding'}, {status:'ready'}, or {status:'error', message}.
 */
export function preloadNeural(modelId = 'neural', onStatus) {
  if (typeof modelId === 'function') { onStatus = modelId; modelId = 'neural'; }
  const files = NEURAL_MODELS[modelId] ?? NEURAL_MODELS.neural;
  loadModel(undefined, onStatus, files).then(
    () => onStatus?.({ status: 'ready' }),
    (err) => onStatus?.({ status: 'error', message: String(err?.message ?? err) })
  );
}

/**
 * Ask the model for a move on a FEN position.
 *
 * Resolves to a `bestmove`-shaped result ({uci, value, cp, ms}) or throws.
 * The caller (worker.js or app.js's main-thread fallback) wraps it in the
 * message envelope; a thrown error leaves the "thinking" indicator handled
 * by the caller's error path.
 */
export async function requestNeuralMove(fen, modelId = 'neural') {
  const started = Date.now();
  const files = NEURAL_MODELS[modelId] ?? NEURAL_MODELS.neural;
  const result = await runNeuralMove(fen, files);
  return { ...result, ms: Date.now() - started };
}
