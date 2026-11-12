/* Runs the repo's trained MNIST CNN (models/12_cnn) in the browser.
   Weights come from website/export_cnn.py (batch norm already folded into the convolutions). */
(function () {
  "use strict";
  let NET = null;

  async function loadNet(base) {
    if (NET) return NET;
    const [meta, buf] = await Promise.all([
      fetch(base + "assets/cnn/model.json").then((r) => r.json()),
      fetch(base + "assets/cnn/weights.bin").then((r) => r.arrayBuffer()),
    ]);
    const f = new Float32Array(buf);
    meta.layers.forEach((L) => {
      if (L.type === "pool") return;
      const nw = L.type === "conv" ? L.out * L.in * 9 : L.out * L.in;
      L.W = f.subarray(L.w, L.w + nw); L.B = f.subarray(L.b, L.b + L.out);
    });
    NET = meta;
    return NET;
  }

  // x: Float32Array [C*H*W]
  function conv(x, C, H, W, L) {
    const out = new Float32Array(L.out * H * W);
    for (let o = 0; o < L.out; o++) {
      const ob = o * H * W, wb = o * C * 9;
      out.fill(L.B[o], ob, ob + H * W);
      for (let c = 0; c < C; c++) {
        const xb = c * H * W, k = wb + c * 9;
        const k0 = L.W[k], k1 = L.W[k + 1], k2 = L.W[k + 2], k3 = L.W[k + 3], k4 = L.W[k + 4], k5 = L.W[k + 5], k6 = L.W[k + 6], k7 = L.W[k + 7], k8 = L.W[k + 8];
        for (let i = 0; i < H; i++) for (let j = 0; j < W; j++) {
          let s = 0;
          const up = i > 0, dn = i < H - 1, lf = j > 0, rt = j < W - 1, p = xb + i * W + j;
          if (up) { if (lf) s += k0 * x[p - W - 1]; s += k1 * x[p - W]; if (rt) s += k2 * x[p - W + 1]; }
          if (lf) s += k3 * x[p - 1]; s += k4 * x[p]; if (rt) s += k5 * x[p + 1];
          if (dn) { if (lf) s += k6 * x[p + W - 1]; s += k7 * x[p + W]; if (rt) s += k8 * x[p + W + 1]; }
          out[ob + i * W + j] += s;
        }
      }
    }
    for (let i = 0; i < out.length; i++) if (out[i] < 0) out[i] = 0; // ReLU
    return out;
  }
  function pool(x, C, H, W) {
    const h = Math.floor(H / 2), w = Math.floor(W / 2), out = new Float32Array(C * h * w);
    for (let c = 0; c < C; c++) for (let i = 0; i < h; i++) for (let j = 0; j < w; j++) {
      const b = c * H * W + 2 * i * W + 2 * j;
      out[c * h * w + i * w + j] = Math.max(x[b], x[b + 1], x[b + W], x[b + W + 1]);
    }
    return { x: out, H: h, W: w };
  }
  function forward(net, img28) { // img28: Float32Array of 784 values in [0,1]
    let x = new Float32Array(784), C = 1, H = 28, W = 28;
    for (let i = 0; i < 784; i++) x[i] = (img28[i] - net.mean) / net.std;
    let firstMaps = null;
    for (const L of net.layers) {
      if (L.type === "conv") { x = conv(x, C, H, W, L); C = L.out; if (!firstMaps) firstMaps = { x: x.slice(), C, H, W }; }
      else if (L.type === "pool") { const r = pool(x, C, H, W); x = r.x; H = r.H; W = r.W; }
      else {
        if (x.length !== L.in) { const g = new Float32Array(C); for (let c = 0; c < C; c++) { let s = 0; for (let k = 0; k < H * W; k++) s += x[c * H * W + k]; g[c] = s / (H * W); } x = g; }
        const out = new Float32Array(L.out);
        for (let o = 0; o < L.out; o++) { let s = L.B[o]; for (let i = 0; i < L.in; i++) s += L.W[o * L.in + i] * x[i]; out[o] = s; }
        x = L === net.layers[net.layers.length - 1] ? out : out.map((v) => Math.max(0, v));
      }
    }
    const m = Math.max(...x), e = Array.from(x, (v) => Math.exp(v - m)), s = e.reduce((a, b) => a + b, 0);
    return { probs: e.map((v) => v / s), firstMaps };
  }

  /* MNIST-style preprocessing: crop the ink, scale its longest side to 20 px,
     centre it by centre of mass in a 28x28 frame. */
  function toMnist(srcCanvas) {
    const w = srcCanvas.width, h = srcCanvas.height;
    const d = srcCanvas.getContext("2d").getImageData(0, 0, w, h).data;
    let x0 = w, y0 = h, x1 = -1, y1 = -1;
    for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) if (d[4 * (y * w + x)] > 30) { if (x < x0) x0 = x; if (x > x1) x1 = x; if (y < y0) y0 = y; if (y > y1) y1 = y; }
    if (x1 < 0) return null;
    const bw = x1 - x0 + 1, bh = y1 - y0 + 1, sc = 20 / Math.max(bw, bh);
    const tw = Math.max(1, Math.round(bw * sc)), th = Math.max(1, Math.round(bh * sc));
    const tmp = document.createElement("canvas"); tmp.width = tw; tmp.height = th;
    const tc = tmp.getContext("2d"); tc.imageSmoothingEnabled = true; tc.imageSmoothingQuality = "high";
    tc.drawImage(srcCanvas, x0, y0, bw, bh, 0, 0, tw, th);
    const td = tc.getImageData(0, 0, tw, th).data;
    let mx = 0, my = 0, ms = 0;
    for (let y = 0; y < th; y++) for (let x = 0; x < tw; x++) { const v = td[4 * (y * tw + x)]; mx += x * v; my += y * v; ms += v; }
    const ox = Math.round(14 - mx / ms), oy = Math.round(14 - my / ms);
    const img = new Float32Array(784);
    for (let y = 0; y < th; y++) for (let x = 0; x < tw; x++) {
      const X = x + ox, Y = y + oy;
      if (X >= 0 && X < 28 && Y >= 0 && Y < 28) img[Y * 28 + X] = td[4 * (y * tw + x)] / 255;
    }
    return img;
  }

  function paintGray(cv, data, W, H, lo, hi) {
    cv.width = W; cv.height = H;
    const c = cv.getContext("2d"), im = c.createImageData(W, H);
    for (let i = 0; i < W * H; i++) { const v = Math.max(0, Math.min(255, ((data[i] - lo) / (hi - lo || 1)) * 255)); im.data[4 * i] = im.data[4 * i + 1] = im.data[4 * i + 2] = v; im.data[4 * i + 3] = 255; }
    c.putImageData(im, 0, 0);
  }

  function initPad(root) {
    const base = root.dataset.base || "";
    const compact = root.hasAttribute("data-compact");
    root.innerHTML = `
      <div class="pad-wrap">
        <canvas class="pad" width="280" height="280" aria-label="Drawing pad: draw a digit from 0 to 9"></canvas>
        <div class="btns"><button class="btn small" type="button" data-act="clear">Clear</button><button class="btn small" type="button" data-act="sample">Show me an example</button></div>
        <p class="demo-hint">Draw one digit, large and centred. The network runs in your browser.</p>
      </div>
      <div class="cnn-out">
        <div class="cnn-guess"><strong data-guess>?</strong><span class="muted" data-conf>Loading the trained network…</span></div>
        <div class="bars" data-bars></div>
        <div class="mini-input"><canvas data-input width="28" height="28"></canvas><span>${compact ? "What the network sees (28×28)" : "What the network sees: your drawing cropped, scaled to 20 px and centred in a 28×28 grid, like MNIST."}</span></div>
        ${compact ? "" : `<div class="ctrl"><span class="lab">First-layer feature maps (32 filters). Each lights up where its pattern appears.</span><div class="maps" data-maps></div></div>`}
      </div>`;
    const pad = root.querySelector(".pad"), ctx = pad.getContext("2d");
    const bars = root.querySelector("[data-bars]"), guess = root.querySelector("[data-guess]"), conf = root.querySelector("[data-conf]");
    const inputCv = root.querySelector("[data-input]"), mapsEl = root.querySelector("[data-maps]");
    for (let k = 0; k < 10; k++) bars.insertAdjacentHTML("beforeend", `<div class="bar"><span>${k}</span><div class="bar-track"><div class="bar-fill"></div></div><span>–</span></div>`);
    const mapCvs = mapsEl ? Array.from({ length: 32 }, () => { const c = document.createElement("canvas"); mapsEl.appendChild(c); return c; }) : [];
    function clear() {
      ctx.fillStyle = "#000"; ctx.fillRect(0, 0, 280, 280);
      guess.textContent = "?"; conf.textContent = NET ? "Draw a digit" : conf.textContent;
      bars.querySelectorAll(".bar").forEach((b) => { b.classList.remove("top"); b.querySelector(".bar-fill").style.width = "0"; b.lastElementChild.textContent = "–"; });
    }
    clear();
    let drawing = false, last = null, pending = false;
    const pos = (e) => { const r = pad.getBoundingClientRect(); return [((e.clientX - r.left) / r.width) * 280, ((e.clientY - r.top) / r.height) * 280]; };
    function stroke(p) {
      ctx.strokeStyle = "#fff"; ctx.lineWidth = 20; ctx.lineCap = "round"; ctx.lineJoin = "round";
      ctx.beginPath(); ctx.moveTo(...(last || p)); ctx.lineTo(...p); ctx.stroke(); last = p;
    }
    pad.addEventListener("pointerdown", (e) => { drawing = true; last = null; pad.setPointerCapture(e.pointerId); stroke(pos(e)); schedule(); });
    pad.addEventListener("pointermove", (e) => { if (drawing) { stroke(pos(e)); schedule(); } });
    pad.addEventListener("pointerup", () => { drawing = false; schedule(); });
    function schedule() { if (pending) return; pending = true; requestAnimationFrame(() => { pending = false; run(); }); }
    function run() {
      if (!NET) return;
      const img = toMnist(pad);
      if (!img) return;
      paintGray(inputCv, img, 28, 28, 0, 1);
      const { probs, firstMaps } = forward(NET, img);
      const order = probs.map((p, k) => [p, k]).sort((a, b) => b[0] - a[0]);
      guess.textContent = order[0][1];
      conf.textContent = `${(order[0][0] * 100).toFixed(1)}% confident` + (order[0][0] < 0.6 ? `; second guess ${order[1][1]}` : "");
      bars.querySelectorAll(".bar").forEach((b, k) => {
        b.classList.toggle("top", k === order[0][1]);
        b.querySelector(".bar-fill").style.width = (probs[k] * 100).toFixed(1) + "%";
        b.lastElementChild.textContent = (probs[k] * 100).toFixed(1) + "%";
      });
      mapCvs.forEach((c, k) => { const m = firstMaps.x.subarray(k * 784, (k + 1) * 784); let hi = 0; for (const v of m) if (v > hi) hi = v; paintGray(c, m, 28, 28, 0, hi); });
    }
    root.querySelector('[data-act="clear"]').addEventListener("click", clear);
    root.querySelector('[data-act="sample"]').addEventListener("click", () => {
      clear();
      const n = Math.floor(Math.random() * 10);
      ctx.strokeStyle = "#fff"; ctx.lineWidth = 20; ctx.lineCap = "round"; ctx.lineJoin = "round";
      const paths = {
        0: [[140, 50], [95, 80], [85, 150], [105, 220], [150, 235], [190, 200], [198, 130], [180, 70], [140, 50]],
        1: [[120, 75], [150, 50], [150, 235]],
        2: [[85, 90], [120, 50], [175, 60], [190, 110], [160, 160], [90, 230], [200, 230]],
        3: [[90, 65], [150, 50], [185, 85], [150, 135], [115, 138], [150, 140], [190, 180], [160, 228], [90, 220]],
        4: [[165, 235], [165, 50], [80, 170], [205, 170]],
        5: [[190, 55], [105, 55], [95, 130], [150, 120], [190, 160], [180, 215], [130, 235], [90, 215]],
        6: [[180, 55], [120, 85], [90, 160], [110, 225], [160, 230], [190, 190], [170, 145], [120, 145], [95, 175]],
        7: [[80, 60], [200, 60], [140, 235]],
        8: [[140, 135], [100, 100], [110, 60], [150, 50], [180, 80], [170, 115], [140, 135], [100, 170], [105, 220], [150, 235], [185, 205], [180, 165], [140, 135]],
        9: [[185, 110], [150, 140], [105, 120], [100, 75], [140, 50], [180, 70], [185, 110], [175, 235]],
      }[n];
      ctx.beginPath(); ctx.moveTo(...paths[0]); paths.slice(1).forEach((p) => ctx.lineTo(...p)); ctx.stroke();
      run();
    });
    loadNet(base).then(() => { conf.textContent = "Draw a digit"; }).catch(() => { conf.textContent = "Could not load the network weights."; });
  }

  document.addEventListener("DOMContentLoaded", () => document.querySelectorAll("[data-cnn-pad]").forEach(initPad));
  window.MNISTNet = { loadNet, forward, toMnist };
})();
