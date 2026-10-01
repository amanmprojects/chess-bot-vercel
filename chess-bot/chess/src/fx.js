/**
 * Win and loss effects.
 *
 * A single full-viewport canvas, driven by requestAnimationFrame, sitting above
 * everything with `pointer-events: none` — the effects are never in the way of
 * studying the finished position, which is the one thing worth looking at
 * after a game ends.
 *
 * Confetti is drawn as rotated rectangles whose vertical scale is driven by a
 * sine, so each piece appears to flutter edge-on and back. That one trick is
 * what separates confetti from falling rectangles.
 */

/** Celebration palette — saturated enough to read against a near-black page. */
const COLORS = [
  '#6f9bff', '#ffd166', '#ff6b9d', '#4fb463', '#c792ea',
  '#ff9f1c', '#4ecdc4', '#ef476f', '#f8f8f8',
];

const GRAVITY = 0.26;
const DRAG = 0.992;

/**
 * @param {HTMLCanvasElement} canvas
 * @param {HTMLElement} board  the board wrapper, shaken on a loss
 */
export function createFx(canvas, board) {
  const ctx = canvas.getContext('2d');
  /** @type {object[]} */
  let pieces = [];
  let raf = 0;
  let last = 0;
  let idleAt = 0;
  let w = 0;
  let h = 0;

  const reduced = matchMedia('(prefers-reduced-motion: reduce)');

  function resize() {
    // Backing store in device pixels, so the pieces are not soft on a HiDPI
    // screen; the CSS size stays in layout pixels.
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    w = canvas.clientWidth;
    h = canvas.clientHeight;
    canvas.width = Math.round(w * dpr);
    canvas.height = Math.round(h * dpr);
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  }

  function piece(x, y, vx, vy) {
    return {
      x, y, vx, vy,
      w: 5 + Math.random() * 6,
      h: 8 + Math.random() * 8,
      rot: Math.random() * Math.PI,
      vr: (Math.random() - 0.5) * 0.3,
      sway: Math.random() * Math.PI * 2,
      swaySpeed: 0.03 + Math.random() * 0.05,
      flutter: 1.6 + Math.random() * 0.6,
      color: COLORS[(Math.random() * COLORS.length) | 0],
      alpha: 1,
    };
  }

  /** Two cones firing up from just above the board's top corners. */
  function poppers() {
    const rect = board.getBoundingClientRect();
    const margin = Math.min(rect.width * 0.18, 110);
    for (const x of [rect.left - margin * 0.4, rect.right + margin * 0.4]) {
      const ox = Math.max(4, Math.min(w - 4, x));
      for (let i = 0; i < 46; i++) {
        // A fan pointing up and inward, so the two streams cross over the board.
        const inward = ox < w / 2 ? 1 : -1;
        const angle = -Math.PI / 2 + (Math.random() - 0.5) * 1.25 + inward * 0.22;
        const speed = 11 + Math.random() * 8;
        pieces.push(piece(
          ox + (Math.random() - 0.5) * 30,
          rect.top + 12 + (Math.random() - 0.5) * 20,
          Math.cos(angle) * speed,
          Math.sin(angle) * speed,
        ));
      }
    }
  }

  /** A slower, sparser fall over the whole viewport to fill in behind them. */
  function rain(count) {
    for (let i = 0; i < count; i++) {
      pieces.push(piece(
        Math.random() * w,
        -20 - Math.random() * h * 0.5,
        (Math.random() - 0.5) * 2.5,
        Math.random() * 2 + 1,
      ));
    }
  }

  /** Advance one step. `scale` is elapsed frames, so 120Hz and 60Hz match. */
  function step(scale) {
    for (const p of pieces) {
      p.vy += GRAVITY * scale;
      p.vx *= Math.pow(DRAG, scale);
      p.vy *= Math.pow(DRAG, scale);
      p.sway += p.swaySpeed * scale;
      p.rot += p.vr * scale;
      p.x += (p.vx + Math.sin(p.sway) * 1.1) * scale;
      p.y += p.vy * scale;
    }

    // Fade over the last stretch so pieces dissolve rather than blinking out
    // of existence at the bottom edge.
    for (const p of pieces) {
      p.alpha = p.y > h - 160 ? Math.max(0, (h - p.y) / 160) : 1;
    }
    pieces = pieces.filter((p) => p.y < h + 24 && p.alpha > 0);
  }

  function draw() {
    ctx.clearRect(0, 0, w, h);
    for (const p of pieces) {
      ctx.save();
      ctx.globalAlpha = p.alpha;
      ctx.translate(p.x, p.y);
      ctx.rotate(p.rot);
      // Edge-on every half turn: this is what makes it read as confetti.
      ctx.scale(1, Math.cos(p.sway * p.flutter));
      ctx.fillStyle = p.color;
      ctx.fillRect(-p.w / 2, -p.h / 2, p.w, p.h);
      ctx.restore();
    }
  }

  function frame(now) {
    raf = 0;
    if (!last) last = now;
    const dt = Math.min(48, now - last);
    last = now;

    step(dt / 16.67);
    draw();

    if (pieces.length > 0) raf = requestAnimationFrame(frame);
  }

  function start() {
    if (reduced.matches) return;
    if (raf) return;
    last = 0;
    raf = requestAnimationFrame(frame);
  }

  function clear() {
    if (raf) cancelAnimationFrame(raf);
    raf = 0;
    last = 0;
    pieces = [];
    ctx.clearRect(0, 0, w, h);
  }

  resize();
  window.addEventListener('resize', resize);

  return {
    /** A win: two poppers across the board, then a long fall behind them. */
    win() {
      clear();
      poppers();
      rain(90);
      start();
    },
    /** A loss: no confetti — a short shake, and the board darkening at the edges. */
    loss() {
      clear();
      if (reduced.matches) return;
      board.classList.remove('shake');
      // Reflow so the animation restarts when two losses land in a row.
      void board.offsetWidth;
      board.classList.add('shake');
      board.addEventListener('animationend', () => board.classList.remove('shake'), { once: true });
    },
    /** A draw: nothing moves, nothing flashes. */
    draw() {
      clear();
    },
    clear,
  };
}