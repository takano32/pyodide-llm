// tests/serving.mjs (T.391.6)
// The site as a tool serves it to a browser from this machine: plain static files under the site's base and no special
// headers, like GitHub Pages (the isolation is coi.js's work, and the page's first visit reloads once for it). One place
// for it: six tools had a server of their own, and the copies had drifted (one read its folder with fileURLToPath for
// Windows (T.242) and four with URL.pathname; one served what is outside the base; two had no type for .wasm).
//
//   const site = await serving();                       // dist/ (npm run build) under /pyodide-llm/
//   await page.goto(site.url);                          // "http://localhost:<port>/pyodide-llm/"
//   await site.close();
//
//   serving(folder, { base, instead })
//     folder    what is served: a path, or a file: URL (default: this checkout's dist/, DIST)
//     base      the path the site is under, with its slashes (default "/pyodide-llm/")
//     instead   a tool's own say in a request under the base: instead({ name, method, file }) with the name under the
//               base, and the file that would be served (undefined: there is none). It returns nothing (served as it
//               is), a status (answered with it and no body), "drop" (the line ends without an answer), or
//               { body, type } (answered 200 with that text; type is an extension of TYPES, default the file's).
//
// What it answers: a file of the folder with its Content-Type and Content-Length (a HEAD without the body); a folder
// with its index.html; 404 for what is not there, for what is not under the base, and for a path that leaves the folder
// however it is spelt ("..", "%2e%2e", "..%2f"); 400 for a path that cannot be decoded. A Range is not read (the whole
// file with 200), nothing says how long to keep a file, and there is no COOP or COEP. It listens where Node does by
// default (every interface: a browser may take "localhost" as ::1 or as 127.0.0.1) on a port the system picks.
//
// Not this: tests/gpu-check/engines.mjs serves the sources with their types taken off and its own cases, under COOP
// and COEP of its own; tests/slow-check.mjs is a proxy that throttles a deployed site.
import http from "node:http";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

export const DIST = new URL("../dist/", import.meta.url);
export const BASE = "/pyodide-llm/";
export const TYPES = { ".html": "text/html; charset=utf-8", ".js": "text/javascript", ".mjs": "text/javascript", ".css": "text/css",
  ".py": "text/plain", ".wasm": "application/wasm", ".zip": "application/x-zip-compressed" };  // (T.498: the Python zips, as GitHub Pages types them)
const typeOf = (name) => TYPES[path.extname(name)] ?? "application/octet-stream";
const stat = (file) => {
  try {
    return fs.statSync(file);
  } catch {
    return undefined;  // not there, or no name a file can have (a zero byte)
  }
};

export async function serving(folder = DIST, { base = BASE, instead } = {}) {
  // fileURLToPath, not URL.pathname: on Windows the pathname is "/D:/a/…" (T.181, T.242), and anywhere a space in it is "%20"
  const root = path.resolve(folder instanceof URL ? fileURLToPath(folder) : folder);
  const none = (res, status = 404) => {
    res.writeHead(status);
    res.end();
  };
  const server = http.createServer((req, res) => {
    let pathname;
    try {
      pathname = decodeURIComponent(new URL(req.url, "http://localhost").pathname);
    } catch {
      return none(res, 400);
    }
    if (!pathname.startsWith(base)) return none(res);
    const name = pathname.slice(base.length);
    // the path is decoded after the URL was read, so "..%2f" is ".." only here: what leaves the folder is not there
    let file = path.join(root, name);
    const inside = file === root || file.startsWith(root + path.sep);
    if (inside && stat(file)?.isDirectory()) file = path.join(file, "index.html");
    const there = inside && stat(file)?.isFile() ? file : undefined;
    const other = instead?.({ name, method: req.method, file: there });
    if (other === "drop") return req.socket.destroy();
    if (typeof other === "number") return none(res, other);
    if (other !== undefined) {
      const body = Buffer.from(other.body);
      res.writeHead(200, { "Content-Type": other.type ? TYPES[other.type] : typeOf(file), "Content-Length": body.length });
      return res.end(body);
    }
    if (!there) return none(res);
    res.writeHead(200, { "Content-Type": typeOf(there), "Content-Length": stat(there).size });
    fs.createReadStream(there).pipe(res);  // (Node sends a HEAD no body)
  });
  await new Promise((resolve, reject) => server.once("error", reject).listen(0, resolve));
  const { port } = server.address();
  return {
    port,
    url: `http://localhost:${port}${base}`,
    // the port is free when this resolves: a browser that did not close (T.141) may still hold a line open
    close: () => new Promise((resolve) => {
      server.close(resolve);
      server.closeAllConnections();
    }),
  };
}
