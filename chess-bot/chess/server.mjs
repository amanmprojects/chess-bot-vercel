/**
 * A static file server for local play. No dependencies — the whole point of
 * this project is that it runs with nothing but Node and a browser.
 *
 *   node server.mjs [port]
 */

import { createServer } from 'node:http';
import { readFile, stat } from 'node:fs/promises';
import { basename, extname, join, normalize, resolve, sep } from 'node:path';
import { fileURLToPath } from 'node:url';

const ROOT = resolve(fileURLToPath(new URL('.', import.meta.url)));
const PORT = Number(process.argv[2] ?? process.env.PORT ?? 8000);

const TYPES = {
  '.html': 'text/html; charset=utf-8',
  '.js': 'text/javascript; charset=utf-8',
  '.mjs': 'text/javascript; charset=utf-8',
  '.css': 'text/css; charset=utf-8',
  '.json': 'application/json; charset=utf-8',
  '.svg': 'image/svg+xml',
  '.png': 'image/png',
  '.ico': 'image/x-icon',
};

/**
 * The model weights are immutable for a given filename and far too big to
 * re-fetch every visit: `no-store` on an 11MB download meant a full re-download
 * on every page load. Cache them hard, exactly as vercel.json does in
 * production. Everything else stays uncached so editing a module and
 * reloading shows the edit.
 */
function cacheControlFor(path) {
  const name = basename(path);
  if (/^model2?\.(bin|json)$/.test(name)) return 'public, max-age=31536000, immutable';
  return 'no-store';
}

/**
 * Map a request path to a file inside ROOT, or null if it escapes.
 * Normalising before resolving is what stops `../` traversal.
 */
function resolvePath(urlPath) {
  let decoded;
  try {
    decoded = decodeURIComponent(urlPath.split('?')[0]);
  } catch {
    return null; // malformed percent-encoding — nothing to serve
  }
  const clean = normalize(decoded).replace(/^(\.\.[/\\])+/, '');
  const full = resolve(join(ROOT, clean));
  if (full !== ROOT && !full.startsWith(ROOT + sep)) return null;
  return full;
}

const server = createServer(async (req, res) => {
  if (req.method !== 'GET' && req.method !== 'HEAD') {
    res.writeHead(405, { allow: 'GET, HEAD' }).end('Method not allowed');
    return;
  }

  let path = resolvePath(req.url === '/' ? '/index.html' : req.url);
  if (!path) {
    res.writeHead(403).end('Forbidden');
    return;
  }

  try {
    let info = await stat(path);
    if (info.isDirectory()) {
      path = join(path, 'index.html');
      info = await stat(path);
    }

    const body = await readFile(path);
    res.writeHead(200, {
      'content-type': TYPES[extname(path)] ?? 'application/octet-stream',
      'content-length': info.size,
      'cache-control': cacheControlFor(path),
    });
    res.end(req.method === 'HEAD' ? undefined : body);
  } catch {
    res.writeHead(404, { 'content-type': 'text/plain; charset=utf-8' });
    res.end('Not found');
  }
});

server.listen(PORT, () => {
  console.log(`Chess running at http://localhost:${PORT}`);
});
