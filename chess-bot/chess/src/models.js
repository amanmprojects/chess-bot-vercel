/**
 * Descriptions of what the computer is playing against.
 *
 * Every number here is read from the thing it describes rather than typed in
 * beside it: search depths and time budgets come from `LEVELS` in ai.js, and
 * the architecture comes from the model's own `model.json`. If either changes,
 * the panel follows — a copied constant would drift silently.
 *
 * The prose is the part that cannot be derived. It exists so somebody who has
 * never written a chess engine can work out why the net misses forks and why
 * the beginner hangs a queen.
 */

import { LEVELS } from './ai.js';
import { loadManifest } from './nn.js';

/** 12,345,678 -> "12,345,678". */
const num = (n) => n.toLocaleString('en-US');

/** Compact node counts: 1190000 -> "1.19M". */
function compact(n) {
  if (!Number.isFinite(n)) return '—';
  if (n < 1000) return String(n);
  if (n < 1e6) return `${Math.round(n / 1000)}k`;
  return `${(n / 1e6).toFixed(2)}M`;
}

/**
 * The search levels all run the same engine; only three numbers differ. The
 * prose below is written once and the level's own settings are interpolated in,
 * because describing one search twice with slightly different numbers invites
 * the two copies to disagree.
 */
function searchDescription(name) {
  const level = LEVELS[name];
  const plies = `${level.depth} ${level.depth === 1 ? 'ply' : 'plies'}`;

  // Written in plies, because that is what the search counts and what the
  // depth figure above is. Converting to "moves" here would silently halve it.
  const summary = {
    beginner: 'Sees one ply ahead — its own move, plus the captures that could answer it — and then picks from the plausible ones at random. It will hang a queen, and it does so often.',
    casual: 'Searches two plies: its move, the reply, and the captures either side. It rarely loses a piece outright, but it will give away a pawn for no reason at all.',
    intermediate: 'Searches four plies — two moves apiece — and plays a club player’s game: sound tactics, loose endgames, no theory.',
    strong: 'Searches eight plies deep and plays the best move it finds, every time. Blunders are rare and always accidental.',
    expert: 'Searches as deep as four seconds of CPU allows and plays whatever that finds. How far it actually gets depends on your machine — the tile below reports it.',
  }[name];

  return {
    id: name,
    name: level.label,
    kind: 'Search',
    tagline: summary,
    stats: [
      { label: 'Search depth', value: `up to ${plies}`, note: 'A ply is one move by one side.' },
      { label: 'Time budget', value: `${level.movetime} ms`, note: 'Iterative deepening stops here.' },
      {
        label: 'Error allowance',
        value: level.randomness
          ? `±${level.randomness} centipawns`
          : 'None',
        note: level.randomness
          ? 'How far below the best move it may wander.'
          : 'It plays the best move it found.',
      },
    ],
    sections: [
      {
        heading: 'How it chooses a move',
        body: [
          'Negamax with alpha-beta pruning. The engine scores a position from the point of view of whoever is to move, then negates that score to search the reply — which means one function handles both colours.',
          'Alpha-beta discards any branch that cannot beat the best move already found, and the move most likely to be that good one is searched first, so the cutoff comes early. Nothing below the cutoff is ever looked at.',
        ],
      },
      {
        heading: 'How deep it really gets',
        body: [
          'Iterative deepening: search one ply, then two, then three, keeping the previous score as a narrow window. The depth number above is the cap. A quiet opening usually finishes well short of it; a forcing line runs to it.',
          'Each pass is kept even if the next one is cut off by the clock, so the move played has always been completely searched. This is why it never swindles into a mate it only found halfway.',
          'The depth in the tile above is what the engine reported on its last move — a real number from your machine, not a promise.',
        ],
      },
      {
        heading: 'What it thinks a position is worth',
        body: [
          'A tapered evaluation: material plus a piece-square table, written twice — once for the middlegame, once for the endgame — and interpolated by how many heavy pieces are still on the board.',
          'On top of that it scores pawn structure (passed, doubled, isolated), the bishop pair, and rooks standing on open and half-open files. Nothing here is learned; it is a PeSTO evaluation with hand-set tables.',
        ],
      },
      {
        heading: 'How it avoids running out of time',
        items: [
          'A 48 MB transposition table, so a position reached by two different move orders is searched once.',
          'Quiescence search: at the edge of the tree it keeps searching captures and promotions, so it never evaluates a position in the middle of an exchange.',
          'Null-move pruning: it hands the opponent a free move, and if the position is still winning without that move, the whole branch is skipped.',
          'Late move reductions: quiet moves tried late in the ordering are searched shallower first, and re-searched properly only if they look good.',
          'Killer moves and a history table, remembering quiet moves that recently caused a cutoff.',
          'MVV-LVA ordering, which tries captures that win material before captures that do not.',
        ],
      },
      level.randomness
        ? {
            heading: 'Why it misses things on purpose',
            body: [
              `After the search it re-scores the moves available to it and picks at random from everything within ${level.randomness} centipawns of the best. That is the error allowance above.`,
              'The point is to produce the ordinary inaccuracy a club player makes, not the nonsense a random-move generator produces. Within the window sit moves that are merely unremarkable; a real blunder usually falls outside it and is never considered.',
            ],
          }
        : {
            heading: 'Why it misses things',
            body: [
              'It plays the best move it found, so anything it misses is a genuine limitation of the search or the evaluation, not a deliberate slip. Give it a position where the only good move is ten plies deep and it will not see it.',
            ],
          },
    ],
  };
}

/**
 * The neural net. The architecture numbers arrive asynchronously from
 * model.json, so this renders twice: once from the known-good defaults if the
 * manifest has not loaded, and again for real once it has.
 */
async function neuralDescription() {
  const m = await loadManifest().catch(() => null);

  const params = m?.params ?? 5_577_034;
  const d = m?.d ?? 256;
  const layers = m?.n_layers ?? 7;
  const heads = m?.n_heads ?? 8;
  const width = heads > 0 ? Math.round(d / heads) : 32;

  const stats = [
    { label: 'Parameters', value: num(params), note: 'Float16, in model.bin.' },
    { label: 'Architecture', value: `${layers} blocks`, note: `${heads} heads of ${width}, width ${d}.` },
    { label: 'Lookahead', value: '1 ply', note: 'One forward pass. No search.' },
  ];

  if (m?.step != null) {
    stats.push({
      label: 'Training',
      value: `${num(m.step)} steps`,
      note: m.best_acc != null
        ? `Best top-1 accuracy on held-out legal moves: ${(m.best_acc * 100).toFixed(1)}%.`
        : null,
    });
  }

  const sections = [
    {
      heading: 'What it sees',
      body: [
        `The position becomes ${64 + 3} vectors of ${d} numbers. Sixty-four of them are the squares: each square looks up the piece standing on it — or an empty square — in a learned embedding table, and adds a learned position embedding.`,
        `Three more carry what is not standing on a square: whose turn it is, the four castling rights, and the en-passant target file. There is no move history. The net is shown the position and nothing else.`,
      ],
    },
    {
      heading: 'What it outputs',
      body: [
        'Two heads. The policy head is one shared matrix applied to each square, turning its 256 numbers into 73 numbers: 56 queen-style slides, 8 knight jumps, and 9 underpromotions. Across 64 squares that is 4,672 scores, one per possible move.',
        'The value head averages the 64 square vectors, normalises them, projects the result to a single number and squashes it into −1…+1. That is where the evaluation bar gets its number.',
      ],
    },
    {
      heading: 'How it picks a move',
      body: [
        'All 4,672 scores are computed, then every square that is not a legal move is thrown away. The highest score that survives wins. There is no sampling and no temperature, so the net is deterministic — ask it the same position twice and it will answer the same way twice.',
      ],
    },
    {
      heading: 'What it cannot do',
      body: [
        'It looks at one position. It has no model of what you will play next, so it cannot see a fork coming, cannot calculate a sacrifice that only pays off in twelve moves, and will walk into a tactic it helped set up.',
        'This is the trade the whole design makes: a trained evaluation of the position in front of it, in exchange for any understanding of the position after it. The search levels above make the opposite trade — they calculate nothing about a position being good, and look a long way past it.',
      ],
    },
    {
      heading: 'Where it runs',
      body: [
        'In your browser, inside the same Web Worker the search uses, so it cannot freeze the board while it thinks. The 11 MB of weights are downloaded once and cached.',
        'The training pipeline is not part of this app. Every architectural number above is read from the exported model.json, and the step count and accuracy are recorded there too.',
      ],
    },
  ];

  return {
    id: 'neural',
    name: 'Neural net',
    kind: 'Neural network',
    tagline: `A ${layers}-block transformer that scores all 4,672 candidate moves in a single pass. It never looks ahead.`,
    stats,
    sections,
  };
}

/**
 * Describe the currently selected opponent.
 *
 * @param {string} level  the `#level` select value
 * @returns {Promise<object>} content in the shape rendered by app.js
 */
export function describeModel(level) {
  return level === 'neural' ? neuralDescription() : Promise.resolve(searchDescription(level));
}

export { compact };