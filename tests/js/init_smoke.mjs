/*
 * Loads the real dashboard bundle against a minimal DOM shim and reports
 * whether init() ran to completion.
 *
 * The point is not to test the DOM. It is that init() wires the fence canvas
 * two thirds of the way down its body, so ANY exception raised above that line
 * silently unwires every click handler on the page — the drawing tools stop
 * responding with nothing in the UI to say why. This harness makes that class
 * of failure loud.
 *
 * Prints one JSON object on stdout; tests/test_dashboard_init.py asserts on it.
 */
import fs from 'node:fs';
import path from 'node:path';
import vm from 'node:vm';

const appPath = path.resolve(process.argv[2]);
const source = fs.readFileSync(appPath, 'utf8');

const listeners = new Map();          // "<element id>:<event>" -> count
const noop = () => {};

function makeElement(id) {
  const el = {
    id,
    tagName: 'DIV',
    dataset: {},
    style: {},
    value: '',
    textContent: '',
    innerHTML: '',
    title: '',
    hidden: false,
    disabled: false,
    width: 640,
    height: 384,
    classList: { add: noop, remove: noop, toggle: noop, contains: () => false },
    addEventListener(type) {
      const key = `${id}:${type}`;
      listeners.set(key, (listeners.get(key) || 0) + 1);
    },
    removeEventListener: noop,
    appendChild: noop,
    removeChild: noop,
    remove: noop,
    querySelector: () => makeElement(`${id}>child`),
    querySelectorAll: () => [],
    getBoundingClientRect: () => ({ left: 0, top: 0, width: 640, height: 384, x: 0, y: 0 }),
    getContext: () => ctx2d,
    focus: noop,
    click: noop,
    setAttribute: noop,
    scrollIntoView: noop,
  };
  return el;
}

const ctx2d = new Proxy({}, {
  get: (_t, prop) => {
    if (prop === 'getImageData') return () => ({ data: new Uint8ClampedArray(4) });
    if (prop === 'canvas') return makeElement('canvas');
    return noop;
  },
  set: () => true,
});

const elements = new Map();
const byId = (id) => {
  if (!elements.has(id)) elements.set(id, makeElement(id));
  return elements.get(id);
};

const documentStub = {
  readyState: 'loading',
  getElementById: byId,
  createElement: (tag) => ({ ...makeElement(`new:${tag}`), tagName: tag.toUpperCase() }),
  querySelector: () => null,
  querySelectorAll: () => [],
  addEventListener(type, handler) {
    const key = `document:${type}`;
    listeners.set(key, (listeners.get(key) || 0) + 1);
    if (type === 'DOMContentLoaded') documentStub.__domReady = handler;
  },
  removeEventListener: noop,
  body: makeElement('body'),
  hidden: false,
};

// Every network call resolves to an empty, well-formed payload: init() must
// survive a backend that has nothing to say, not just a happy one.
const jsonResponse = (body) => Promise.resolve({
  ok: true,
  status: 200,
  statusText: 'OK',
  headers: { get: () => 'application/json' },
  json: () => Promise.resolve(body),
  text: () => Promise.resolve(''),
});

class WebSocketStub {
  constructor() { this.readyState = 0; }
  send() {} close() {}
  static CONNECTING = 0; static OPEN = 1; static CLOSING = 2; static CLOSED = 3;
}

// 'default' is the state every operator starts in — neither granted nor
// denied. The bug this harness was written for only fired in that state.
const permission = process.argv[3] || 'default';

const sandbox = {
  console,
  document: documentStub,
  localStorage: {
    _s: {},
    getItem(k) { return k in this._s ? this._s[k] : null; },
    setItem(k, v) { this._s[k] = String(v); },
    removeItem(k) { delete this._s[k]; },
  },
  location: { protocol: 'http:', host: '127.0.0.1:8000', href: 'http://127.0.0.1:8000/dashboard' },
  fetch: () => jsonResponse({}),
  WebSocket: WebSocketStub,
  Notification: class { static permission = permission; close() {} },
  Image: class { set src(_v) {} },
  FormData: class { append() {} },
  setTimeout: (fn, ms) => setTimeout(fn, Math.min(ms || 0, 0)),
  clearTimeout,
  setInterval: () => 0,
  clearInterval: noop,
  navigator: { userAgent: 'node', onLine: true },
  AudioContext: class { constructor() { this.state = 'suspended'; } },
  // The page listens on window for scroll/resize to re-cut the MJPEG stream
  // budget, and coalesces the work into an animation frame.
  addEventListener(type) {
    const key = `window:${type}`;
    listeners.set(key, (listeners.get(key) || 0) + 1);
  },
  removeEventListener: noop,
  requestAnimationFrame: (fn) => { fn(0); return 1; },
  cancelAnimationFrame: noop,
  innerWidth: 1440,
  innerHeight: 900,
};
sandbox.window = sandbox;
sandbox.globalThis = sandbox;

const result = { loaded: false, initCompleted: false, error: null, wired: [] };
try {
  vm.createContext(sandbox);
  new vm.Script(source, { filename: appPath }).runInContext(sandbox);
  result.loaded = true;
  documentStub.readyState = 'interactive';
  if (!documentStub.__domReady) throw new Error('init() was never registered on DOMContentLoaded');
  documentStub.__domReady();
  result.initCompleted = true;
} catch (err) {
  result.error = `${err.name}: ${err.message}`;
}
result.wired = [...listeners.keys()].sort();
console.log(JSON.stringify(result));
