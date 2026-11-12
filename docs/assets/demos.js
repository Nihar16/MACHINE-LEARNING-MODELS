/* Interactive demos. Each page marks its demo with <div data-demo="name">. */
(function () {
  "use strict";
  const { Plot, Playground, rng, el, slider, select, checkbox, button, stats, demoShell, tokens, rgba, argmax, hexToRgb } = window.ML;
  const EXTRA = ["#d62728", "#8c564b", "#e377c2", "#17becf"];
  const palette = (t) => [...t.cls, ...EXTRA];
  const fmtPct = (v) => (v * 100).toFixed(0) + "%";

  // ---------- small maths helpers ----------
  function solve(A, b) { // Gaussian elimination with partial pivoting
    const n = b.length, M = A.map((r, i) => [...r, b[i]]);
    for (let c = 0; c < n; c++) {
      let p = c; for (let r = c + 1; r < n; r++) if (Math.abs(M[r][c]) > Math.abs(M[p][c])) p = r;
      [M[c], M[p]] = [M[p], M[c]];
      if (Math.abs(M[c][c]) < 1e-12) continue;
      for (let r = 0; r < n; r++) if (r !== c) { const f = M[r][c] / M[c][c]; for (let k = c; k <= n; k++) M[r][k] -= f * M[c][k]; }
    }
    return M.map((r, i) => (Math.abs(r[i]) < 1e-12 ? 0 : r[n] / r[i]));
  }
  function softmax(z) { const m = Math.max(...z); const e = z.map((v) => Math.exp(v - m)); const s = e.reduce((a, b) => a + b, 0); return e.map((v) => v / s); }
  const sigmoid = (z) => 1 / (1 + Math.exp(-z));

  // marching squares: draw the contour fn(x,y) = level
  function contour(plot, fn, level, color, width = 2, dash = null, res = 110) {
    const rows = Math.round(res / plot.aspect), ctx = plot.ctx;
    const xs = (i) => plot.xmin + (i / res) * (plot.xmax - plot.xmin), ys = (j) => plot.ymax - (j / rows) * (plot.ymax - plot.ymin);
    const v = []; for (let j = 0; j <= rows; j++) { v.push([]); for (let i = 0; i <= res; i++) v[j].push(fn(xs(i), ys(j)) - level); }
    ctx.beginPath(); ctx.setLineDash(dash || []); ctx.strokeStyle = color; ctx.lineWidth = width;
    for (let j = 0; j < rows; j++) for (let i = 0; i < res; i++) {
      const c = [[i, j, v[j][i]], [i + 1, j, v[j][i + 1]], [i + 1, j + 1, v[j + 1][i + 1]], [i, j + 1, v[j + 1][i]]];
      const pts = [];
      for (let e = 0; e < 4; e++) {
        const a = c[e], b = c[(e + 1) % 4];
        if ((a[2] < 0) !== (b[2] < 0)) { const t = a[2] / (a[2] - b[2]); pts.push([a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1])]); }
      }
      if (pts.length >= 2) {
        const tx = (p) => plot.px(xs(p[0])), ty = (p) => plot.py(ys(p[1]));
        ctx.moveTo(tx(pts[0]), ty(pts[0])); ctx.lineTo(tx(pts[1]), ty(pts[1]));
        if (pts.length === 4) { ctx.moveTo(tx(pts[2]), ty(pts[2])); ctx.lineTo(tx(pts[3]), ty(pts[3])); }
      }
    }
    ctx.stroke(); ctx.setLineDash([]);
  }

  const DEMOS = {};

  // =============== Linear regression ===============
  DEMOS["linear-regression"] = (root) => {
    const { stage, side } = demoShell(root, "Drag a point to move it, click empty space to add one, or shift-click a point to remove it. The line always minimises the total area of the squares.");
    const plot = Plot(stage, { aspect: 1.35, xmin: 0, xmax: 10, ymin: 0, ymax: 10 });
    let pts = [], fit = null, drag = null, gd = null, showSq = true, anim = null;
    function sample() {
      const r = rng(Math.floor(Math.random() * 1e6) + 1);
      pts = Array.from({ length: 14 }, () => { const x = 0.6 + r() * 8.8; return { x, y: Math.max(0.3, Math.min(9.7, 1.4 + 0.65 * x + 0.9 * r.normal())) }; });
    }
    function ols() {
      const n = pts.length; if (n < 2) return null;
      const mx = pts.reduce((s, p) => s + p.x, 0) / n, my = pts.reduce((s, p) => s + p.y, 0) / n;
      let sxy = 0, sxx = 0, sst = 0; pts.forEach((p) => { sxy += (p.x - mx) * (p.y - my); sxx += (p.x - mx) ** 2; sst += (p.y - my) ** 2; });
      const w = sxx ? sxy / sxx : 0, b = my - w * mx;
      const sse = pts.reduce((s, p) => s + (p.y - (w * p.x + b)) ** 2, 0);
      return { w, b, mse: sse / n, r2: sst ? 1 - sse / sst : 1 };
    }
    function draw() {
      fit = ols();
      const t = plot.clear(); plot.grid(1);
      const line = gd || fit;
      if (line) {
        pts.forEach((p) => {
          const yh = line.w * p.x + line.b, r = p.y - yh;
          if (showSq) {
            const s = Math.abs(plot.py(p.y) - plot.py(yh));
            plot.ctx.fillStyle = rgba(r > 0 ? t.cls[1] : t.cls[0], 0.13);
            plot.ctx.fillRect(plot.px(p.x), Math.min(plot.py(p.y), plot.py(yh)), r > 0 ? s : -s, s);
          }
          plot.line(p.x, p.y, p.x, yh, t.ink3, 1, [3, 3]);
        });
        plot.line(0, line.b, 10, line.w * 10 + line.b, t.accent, 2.5);
      }
      pts.forEach((p) => plot.point(p.x, p.y, t.cls[0], { r: 6 }));
      plot.text("x", plot.W - 12, plot.H - 8, t.ink3, { align: "right" });
      plot.text("y", 8, 16, t.ink3);
      if (line) {
        refs.eq.textContent = `y = ${line.w.toFixed(2)}x ${line.b >= 0 ? "+" : "−"} ${Math.abs(line.b).toFixed(2)}`;
        const sse = pts.reduce((s, p) => s + (p.y - (line.w * p.x + line.b)) ** 2, 0), my = pts.reduce((s, p) => s + p.y, 0) / pts.length;
        const sst = pts.reduce((s, p) => s + (p.y - my) ** 2, 0);
        refs.mse.textContent = (sse / pts.length).toFixed(2);
        refs.r2.textContent = sst ? (1 - sse / sst).toFixed(3) : "–";
      }
      refs.n.textContent = pts.length;
    }
    plot.onResize = draw;
    const refs = stats(side, [["eq", "Fitted line", true], ["mse", "Mean squared error"], ["r2", "R²"], ["n", "Points"]]);
    checkbox(side, { label: "Show squared errors", checked: true, onChange: (v) => { showSq = v; draw(); } });
    const lr = { v: 0.01 };
    slider(side, { label: "Gradient descent step size", min: 0.002, max: 0.03, step: 0.002, value: 0.01, fmt: (v) => v.toFixed(3), onInput: (v) => (lr.v = v) });
    const btns = el("div", { class: "btns" });
    button(btns, "Watch gradient descent", () => {
      if (anim) cancelAnimationFrame(anim);
      gd = { w: 0, b: 0 }; let it = 0;
      const step = () => {
        for (let k = 0; k < 4; k++) {
          let gw = 0, gb = 0; pts.forEach((p) => { const e = gd.w * p.x + gd.b - p.y; gw += e * p.x; gb += e; });
          gd.w -= lr.v * (2 / pts.length) * gw; gd.b -= lr.v * 2 * (2 / pts.length) * gb; it++;
        }
        refs.eq.textContent = "";
        draw();
        if (!isFinite(gd.w) || Math.abs(gd.w) > 1e3) { gd = null; draw(); refs.eq.textContent = "Diverged: step too large"; return; }
        if (it < 1600) anim = requestAnimationFrame(step); else { gd = null; draw(); }
      };
      anim = requestAnimationFrame(step);
    });
    button(btns, "Add an outlier", () => { pts.push({ x: 8.8, y: 1.2 }); gd = null; draw(); });
    button(btns, "New data", () => { sample(); gd = null; draw(); });
    side.appendChild(btns);
    const near = (pos) => pts.findIndex((p) => Math.hypot(plot.px(p.x) - pos.px, plot.py(p.y) - pos.py) < 12);
    plot.cv.addEventListener("pointerdown", (e) => {
      const pos = plot.eventPos(e), i = near(pos);
      gd = null;
      if (i >= 0 && e.shiftKey) { pts.splice(i, 1); draw(); return; }
      if (i >= 0) { drag = i; plot.cv.setPointerCapture(e.pointerId); return; }
      pts.push({ x: pos.x, y: pos.y }); draw();
    });
    plot.cv.addEventListener("pointermove", (e) => { if (drag === null) return; const pos = plot.eventPos(e); pts[drag] = { x: Math.max(0, Math.min(10, pos.x)), y: Math.max(0, Math.min(10, pos.y)) }; draw(); });
    plot.cv.addEventListener("pointerup", () => (drag = null));
    sample(); draw();
  };

  // =============== Logistic regression ===============
  DEMOS["logistic-regression"] = (root) => {
    const feats = (x, y, poly) => (poly ? [1, x, y, x * x, y * y, x * y] : [1, x, y]);
    Playground(root, {
      hint: "Logistic regression draws one straight boundary and turns distance from it into a probability. Tick “curved features” to let it bend.",
      datasets: ["blobs", "overlap", "three", "circles", "moons"],
      maxClasses: 3,
      params: [
        { key: "lambda", label: "Regularisation strength", min: 0, max: 0.2, step: 0.01, value: 0.01, fmt: (v) => v.toFixed(2) },
        { key: "poly", type: "check", label: "Add curved features (x², y², xy)", value: false },
      ],
      fit(train, p, K) {
        const d = feats(0, 0, p.poly).length;
        const W = Array.from({ length: K }, () => new Array(d).fill(0));
        const X = train.map((q) => feats(q.x, q.y, p.poly));
        for (let it = 0; it < 600; it++) {
          const G = W.map(() => new Array(d).fill(0));
          X.forEach((x, n) => {
            const pr = softmax(W.map((w) => w.reduce((s, wi, i) => s + wi * x[i], 0)));
            pr.forEach((pk, k) => { const e = pk - (train[n].c === k ? 1 : 0); for (let i = 0; i < d; i++) G[k][i] += e * x[i]; });
          });
          W.forEach((w, k) => { for (let i = 0; i < d; i++) w[i] -= 1.5 * (G[k][i] / X.length + (i ? p.lambda * w[i] : 0)); });
        }
        return { W, poly: p.poly };
      },
      predict: (m, x, y) => softmax(m.W.map((w) => w.reduce((s, wi, i) => s + wi * feats(x, y, m.poly)[i], 0))),
      overlay(plot, m, s, t) {
        if (m.W.length === 2) contour(plot, (x, y) => this.predict(m, x, y)[1], 0.5, t.ink, 2);
      },
      sideExtra(side) { const w = el("div", { class: "small muted" }); side.appendChild(w); return { w }; },
      afterDraw(s, r) {
        if (!s.model || s.model.W.length !== 2 || s.model.poly) { r.w.textContent = ""; return; }
        const w = s.model.W[1].map((v, i) => v - s.model.W[0][i]);
        r.w.innerHTML = `Learned: <span class="num">z = ${w[0].toFixed(2)} ${w[1] >= 0 ? "+" : "−"} ${Math.abs(w[1]).toFixed(2)}·x ${w[2] >= 0 ? "+" : "−"} ${Math.abs(w[2]).toFixed(2)}·y</span>, then P(class B) = 1 / (1 + e<sup>−z</sup>)`;
      },
    });
  };

  // =============== KNN ===============
  DEMOS["knn"] = (root) => {
    let hover = null;
    function neighbours(m, x, y) {
      return m.pts.map((p) => ({ p, d: Math.hypot(p.x - x, p.y - y) })).sort((a, b) => a.d - b.d).slice(0, m.k);
    }
    const pg = Playground(root, {
      hint: "Hover over the plot to see the k nearest training points vote. Small k follows every point; large k smooths the boundary.",
      datasets: ["moons", "blobs", "overlap", "circles", "three", "xor"],
      dataset: "moons", maxClasses: 3, res: 70,
      params: [
        { key: "k", label: "k (neighbours)", min: 1, max: 31, step: 2, value: 5 },
        { key: "weights", type: "select", label: "Vote", options: [["uniform", "One vote each"], ["distance", "Closer counts more"]], value: "uniform" },
      ],
      fit: (train, p, K) => ({ pts: train, k: Math.min(p.k, train.length), w: p.weights, K }),
      predict(m, x, y) {
        const v = new Array(m.K).fill(1e-9);
        neighbours(m, x, y).forEach(({ p, d }) => (v[p.c] += m.w === "distance" ? 1 / (d + 1e-3) : 1));
        const s = v.reduce((a, b) => a + b, 0); return v.map((q) => q / s);
      },
      overlayTop(plot, m, s, t) {
        if (!hover) return;
        const nb = neighbours(m, hover.x, hover.y);
        nb.forEach(({ p }) => plot.line(hover.x, hover.y, p.x, p.y, t.ink2, 1.2));
        const r = nb[nb.length - 1].d;
        plot.ctx.beginPath(); plot.ctx.setLineDash([4, 4]);
        plot.ctx.arc(plot.px(hover.x), plot.py(hover.y), r * plot.W / (plot.xmax - plot.xmin), 0, 2 * Math.PI);
        plot.ctx.strokeStyle = t.ink3; plot.ctx.lineWidth = 1; plot.ctx.stroke(); plot.ctx.setLineDash([]);
        const pr = this.predict(m, hover.x, hover.y);
        plot.point(hover.x, hover.y, t.cls[argmax(pr)], { r: 7, ring: t.ink });
      },
      onHover(pos) { hover = pos; },
    });
    return pg;
  };

  // =============== Naive Bayes ===============
  DEMOS["naive-bayes"] = (root) => {
    Playground(root, {
      hint: "Each class is modelled as a bell curve along x times a bell curve along y (that's the “naive” independence). The ellipses show 1 and 2 standard deviations.",
      datasets: ["blobs", "overlap", "three", "moons", "xor"], maxClasses: 3,
      params: [{ key: "smooth", label: "Variance smoothing", min: 0, max: 0.1, step: 0.005, value: 0.0, fmt: (v) => v.toFixed(3) }],
      fit(train, p, K) {
        const cls = [];
        for (let k = 0; k < K; k++) {
          const s = train.filter((q) => q.c === k); const n = Math.max(1, s.length);
          const mx = s.reduce((a, q) => a + q.x, 0) / n, my = s.reduce((a, q) => a + q.y, 0) / n;
          const vx = s.reduce((a, q) => a + (q.x - mx) ** 2, 0) / n + 1e-3 + p.smooth, vy = s.reduce((a, q) => a + (q.y - my) ** 2, 0) / n + 1e-3 + p.smooth;
          cls.push({ prior: s.length / train.length || 1e-6, mx, my, vx, vy, n: s.length });
        }
        return { cls };
      },
      predict(m, x, y) {
        const lp = m.cls.map((c) => c.n ? Math.log(c.prior) - 0.5 * (Math.log(2 * Math.PI * c.vx) + (x - c.mx) ** 2 / c.vx) - 0.5 * (Math.log(2 * Math.PI * c.vy) + (y - c.my) ** 2 / c.vy) : -1e9);
        return softmax(lp);
      },
      overlay(plot, m, s, t) {
        m.cls.forEach((c, k) => {
          if (!c.n) return;
          [1, 2].forEach((z) => {
            plot.ctx.beginPath(); plot.ctx.setLineDash(z === 2 ? [4, 4] : []);
            plot.ctx.ellipse(plot.px(c.mx), plot.py(c.my), z * Math.sqrt(c.vx) * plot.W / (plot.xmax - plot.xmin), z * Math.sqrt(c.vy) * plot.H / (plot.ymax - plot.ymin), 0, 0, 2 * Math.PI);
            plot.ctx.strokeStyle = t.cls[k]; plot.ctx.lineWidth = 1.5; plot.ctx.stroke(); plot.ctx.setLineDash([]);
          });
          plot.ctx.font = "bold 14px sans-serif"; plot.ctx.fillStyle = t.cls[k]; plot.ctx.textAlign = "center"; plot.ctx.fillText("+", plot.px(c.mx), plot.py(c.my) + 5);
        });
      },
    });
  };

  // =============== SVM (simplified SMO) ===============
  DEMOS["svm"] = (root) => {
    const kern = (m, a, b) => m.kernel === "linear" ? a.x * b.x + a.y * b.y : Math.exp(-m.gamma * ((a.x - b.x) ** 2 + (a.y - b.y) ** 2));
    function f(m, x, y) { let s = m.b; const q = { x, y }; m.sv.forEach((v) => (s += v.a * v.t * kern(m, v.p, q))); return s; }
    Playground(root, {
      hint: "The SVM finds the widest street between the classes. Circled points are support vectors: the only points that decide the boundary. Dashed lines are the edges of the street.",
      datasets: ["blobs", "overlap", "moons", "circles", "xor"],
      params: [
        { key: "kernel", type: "select", label: "Kernel", options: [["linear", "Linear (straight street)"], ["rbf", "RBF (curved street)"]], value: "linear" },
        { key: "logC", label: "C (penalty for points inside the street)", min: -2, max: 3, step: 0.25, value: 0, fmt: (v) => (10 ** v).toPrecision(2) },
        { key: "gamma", label: "γ (RBF reach; larger = more local)", min: 0.5, max: 30, step: 0.5, value: 4 },
      ],
      fit(train, p) {
        const m = { kernel: p.kernel, gamma: p.gamma, C: 10 ** p.logC, b: 0 };
        const pts = train.filter((q) => q.c < 2), n = pts.length, t = pts.map((q) => (q.c ? 1 : -1)), a = new Array(n).fill(0);
        const K = pts.map((pi) => pts.map((pj) => kern(m, pi, pj)));
        const fx = (i) => { let s = m.b; for (let j = 0; j < n; j++) if (a[j]) s += a[j] * t[j] * K[j][i]; return s; };
        const r = rng(3); let passes = 0, it = 0;
        while (passes < 8 && it < 400) {
          let changed = 0; it++;
          for (let i = 0; i < n; i++) {
            const Ei = fx(i) - t[i];
            if ((t[i] * Ei < -1e-3 && a[i] < m.C) || (t[i] * Ei > 1e-3 && a[i] > 0)) {
              let j = Math.floor(r() * (n - 1)); if (j >= i) j++;
              const Ej = fx(j) - t[j], ai = a[i], aj = a[j];
              const L = t[i] !== t[j] ? Math.max(0, aj - ai) : Math.max(0, ai + aj - m.C), H = t[i] !== t[j] ? Math.min(m.C, m.C + aj - ai) : Math.min(m.C, ai + aj);
              if (L === H) continue;
              const eta = 2 * K[i][j] - K[i][i] - K[j][j]; if (eta >= 0) continue;
              a[j] = Math.min(H, Math.max(L, aj - (t[j] * (Ei - Ej)) / eta));
              if (Math.abs(a[j] - aj) < 1e-5) continue;
              a[i] = ai + t[i] * t[j] * (aj - a[j]);
              const b1 = m.b - Ei - t[i] * (a[i] - ai) * K[i][i] - t[j] * (a[j] - aj) * K[i][j];
              const b2 = m.b - Ej - t[i] * (a[i] - ai) * K[i][j] - t[j] * (a[j] - aj) * K[j][j];
              m.b = a[i] > 0 && a[i] < m.C ? b1 : a[j] > 0 && a[j] < m.C ? b2 : (b1 + b2) / 2;
              changed++;
            }
          }
          passes = changed ? 0 : passes + 1;
        }
        m.sv = pts.map((q, i) => ({ p: q, a: a[i], t: t[i] })).filter((v) => v.a > 1e-6);
        return m;
      },
      predict(m, x, y) { const p = sigmoid(2 * f(m, x, y)); return [1 - p, p]; },
      overlay(plot, m, s, t) {
        contour(plot, (x, y) => f(m, x, y), 0, t.ink, 2.2);
        contour(plot, (x, y) => f(m, x, y), 1, t.ink2, 1.2, [5, 4]);
        contour(plot, (x, y) => f(m, x, y), -1, t.ink2, 1.2, [5, 4]);
      },
      overlayTop(plot, m, s, t) { m.sv.forEach((v) => { plot.ctx.beginPath(); plot.ctx.arc(plot.px(v.p.x), plot.py(v.p.y), 9, 0, 2 * Math.PI); plot.ctx.strokeStyle = t.ink; plot.ctx.lineWidth = 1.5; plot.ctx.stroke(); }); },
      sideExtra(side) { const w = el("div", { class: "small muted" }); side.appendChild(w); return { w }; },
      afterDraw(s, r) { r.w.textContent = s.model ? `${s.model.sv.length} support vectors out of ${s.pts.filter((p) => !p.test).length} training points.` : ""; },
    });
  };

  // =============== Decision tree / Random forest ===============
  function gini(counts, n) { let g = 1; counts.forEach((c) => (g -= (c / n) ** 2)); return g; }
  function buildTree(pts, K, depth, maxDepth, minLeaf, r, featSub) {
    const counts = new Array(K).fill(0); pts.forEach((p) => counts[p.c]++);
    const node = { counts, n: pts.length };
    if (depth >= maxDepth || pts.length < 2 * minLeaf || counts.some((c) => c === pts.length)) return node;
    let best = null; const parentG = gini(counts, pts.length);
    const feats = featSub ? [r() < 0.5 ? "x" : "y"] : ["x", "y"];
    feats.forEach((f) => {
      const s = [...pts].sort((a, b) => a[f] - b[f]);
      const left = new Array(K).fill(0);
      for (let i = 0; i < s.length - 1; i++) {
        left[s[i].c]++;
        if (s[i][f] === s[i + 1][f] || i + 1 < minLeaf || s.length - i - 1 < minLeaf) continue;
        const nl = i + 1, nr = s.length - nl, right = counts.map((c, k) => c - left[k]);
        const g = (nl * gini(left, nl) + nr * gini(right, nr)) / s.length;
        if (!best || g < best.g) best = { g, f, thr: (s[i][f] + s[i + 1][f]) / 2 };
      }
    });
    if (!best || parentG - best.g < 1e-9) return node;
    node.f = best.f; node.thr = best.thr;
    node.l = buildTree(pts.filter((p) => p[best.f] <= best.thr), K, depth + 1, maxDepth, minLeaf, r, featSub);
    node.r = buildTree(pts.filter((p) => p[best.f] > best.thr), K, depth + 1, maxDepth, minLeaf, r, featSub);
    return node;
  }
  function treeProba(n, x, y) { while (n.f) n = (n.f === "x" ? x : y) <= n.thr ? n.l : n.r; return n.counts.map((c) => c / n.n); }
  function drawSplits(plot, n, box, color, depth = 0) {
    if (!n.f) return;
    if (n.f === "x") plot.line(n.thr, box.y0, n.thr, box.y1, color, Math.max(1, 2.5 - depth * 0.4));
    else plot.line(box.x0, n.thr, box.x1, n.thr, color, Math.max(1, 2.5 - depth * 0.4));
    drawSplits(plot, n.l, n.f === "x" ? { ...box, x1: n.thr } : { ...box, y1: n.thr }, color, depth + 1);
    drawSplits(plot, n.r, n.f === "x" ? { ...box, x0: n.thr } : { ...box, y0: n.thr }, color, depth + 1);
  }
  function treeText(n, names, indent = "") {
    if (!n.f) { const k = argmax(n.counts); return `${indent}→ class ${names[k]}  (${n.counts.join(" / ")})\n`; }
    return `${indent}if ${n.f} ≤ ${n.thr.toFixed(2)}:\n` + treeText(n.l, names, indent + "   ") + `${indent}else:\n` + treeText(n.r, names, indent + "   ");
  }
  const countLeaves = (n) => (n.f ? countLeaves(n.l) + countLeaves(n.r) : 1);

  DEMOS["decision-tree"] = (root) => {
    Playground(root, {
      hint: "The tree splits the plane with straight cuts parallel to the axes, always choosing the cut that makes each side purest. Raise the depth and watch it start memorising noise.",
      datasets: ["blobs", "overlap", "moons", "xor", "circles", "three"], dataset: "overlap", maxClasses: 3, res: 110,
      params: [
        { key: "depth", label: "Max depth", min: 1, max: 12, step: 1, value: 3 },
        { key: "minLeaf", label: "Min points per leaf", min: 1, max: 15, step: 1, value: 1 },
      ],
      fit: (train, p, K) => ({ root: buildTree(train, K, 0, p.depth, p.minLeaf, null, false) }),
      predict: (m, x, y) => treeProba(m.root, x, y),
      overlay(plot, m, s, t) { drawSplits(plot, m.root, { x0: plot.xmin, x1: plot.xmax, y0: plot.ymin, y1: plot.ymax }, t.ink, 0); },
      sideExtra(side) { const w = el("div", { class: "tree-view", "aria-label": "Tree rules" }); side.appendChild(w); return { w }; },
      afterDraw(s, r) { r.w.textContent = s.model ? `${countLeaves(s.model.root)} leaves\n` + treeText(s.model.root, ["A", "B", "C"]) : ""; },
    });
  };

  DEMOS["random-forest"] = (root) => {
    let show = 0;
    const pg = Playground(root, {
      hint: "Each tree trains on a bootstrap sample and picks from a random feature at each split. One tree is jagged; the average of many is smooth. Use the second slider to inspect a single tree.",
      datasets: ["overlap", "moons", "circles", "xor", "three"], dataset: "moons", maxClasses: 3, res: 80,
      params: [
        { key: "trees", label: "Number of trees", min: 1, max: 60, step: 1, value: 25 },
        { key: "depth", label: "Max depth per tree", min: 1, max: 12, step: 1, value: 6 },
      ],
      fit(train, p, K) {
        const r = rng(11), trees = [];
        for (let i = 0; i < p.trees; i++) {
          const boot = Array.from({ length: train.length }, () => train[Math.floor(r() * train.length)]);
          trees.push(buildTree(boot, K, 0, p.depth, 1, r, true));
        }
        return { trees, K };
      },
      predict(m, x, y) {
        if (show > 0 && show <= m.trees.length) return treeProba(m.trees[show - 1], x, y);
        const acc = new Array(m.K).fill(0);
        m.trees.forEach((t) => treeProba(t, x, y).forEach((v, k) => (acc[k] += v / m.trees.length)));
        return acc;
      },
      overlay(plot, m, s, t) { if (show > 0 && show <= m.trees.length) drawSplits(plot, m.trees[show - 1], { x0: plot.xmin, x1: plot.xmax, y0: plot.ymin, y1: plot.ymax }, t.ink, 0); },
      sideExtra(side) {
        const s = slider(side, { label: "Show", min: 0, max: 60, step: 1, value: 0, fmt: (v) => (v === 0 ? "whole forest" : "tree #" + v), onInput: (v) => { show = v; pg.draw(); } });
        return { s };
      },
    });
  };

  // =============== Neural network (MLP) ===============
  DEMOS["neural-network"] = (root) => {
    let epoch = { n: 0, loss: 0 };
    function makeNet(sizes, r) {
      return sizes.slice(1).map((n, l) => {
        const m = sizes[l], sc = Math.sqrt(2 / m);
        return { W: Array.from({ length: n }, () => Array.from({ length: m }, () => r.normal() * sc)), b: new Array(n).fill(0),
          mW: Array.from({ length: n }, () => new Array(m).fill(0)), vW: Array.from({ length: n }, () => new Array(m).fill(0)), mb: new Array(n).fill(0), vb: new Array(n).fill(0) };
      });
    }
    const act = (a, z) => (a === "relu" ? Math.max(0, z) : Math.tanh(z));
    const dact = (a, h) => (a === "relu" ? (h > 0 ? 1 : 0) : 1 - h * h);
    function forward(m, x, y) {
      let h = [x, y]; const hs = [h];
      m.layers.forEach((L, li) => {
        const z = L.W.map((w, j) => w.reduce((s, wi, i) => s + wi * h[i], L.b[j]));
        h = li === m.layers.length - 1 ? softmax(z) : z.map((v) => act(m.act, v));
        hs.push(h);
      });
      return hs;
    }
    const pg = Playground(root, {
      hint: "A small network learns the boundary live. Each hidden neuron draws one soft line; the next layer combines them into curves. Try the spiral with 2 layers.",
      datasets: ["circles", "xor", "moons", "spiral", "three"], dataset: "circles", maxClasses: 3, animate: true, res: 60,
      params: [
        { key: "layers", label: "Hidden layers", min: 1, max: 3, step: 1, value: 1 },
        { key: "neurons", label: "Neurons per layer", min: 1, max: 16, step: 1, value: 4 },
        { key: "act", type: "select", label: "Activation", options: [["tanh", "tanh"], ["relu", "ReLU"]], value: "tanh" },
        { key: "lr", label: "Learning rate", min: 0.005, max: 0.1, step: 0.005, value: 0.03, fmt: (v) => v.toFixed(3) },
      ],
      fit(train, p, K) {
        const sizes = [2, ...new Array(p.layers).fill(p.neurons), K];
        const m = { layers: makeNet(sizes, rng(5)), act: p.act, t: 0 };
        epoch = { n: 0, loss: 0 };
        m.step = () => {
          for (let rep = 0; rep < 10; rep++) {
            const gW = m.layers.map((L) => L.W.map((w) => w.map(() => 0))), gb = m.layers.map((L) => L.b.map(() => 0));
            let loss = 0;
            train.forEach((q) => {
              const hs = forward(m, q.x, q.y);
              let d = hs[hs.length - 1].map((v, k) => v - (k === q.c ? 1 : 0));
              loss -= Math.log(Math.max(1e-9, hs[hs.length - 1][q.c]));
              for (let l = m.layers.length - 1; l >= 0; l--) {
                const L = m.layers[l], hin = hs[l];
                d.forEach((dj, j) => { gb[l][j] += dj; for (let i = 0; i < hin.length; i++) gW[l][j][i] += dj * hin[i]; });
                if (l > 0) d = hin.map((hi, i) => dact(m.act, hi) * d.reduce((s, dj, j) => s + dj * L.W[j][i], 0));
              }
            });
            m.t++;
            const b1 = 0.9, b2 = 0.999, c1 = 1 - b1 ** m.t, c2 = 1 - b2 ** m.t, n = train.length;
            m.layers.forEach((L, l) => {
              L.W.forEach((w, j) => w.forEach((_, i) => {
                const g = gW[l][j][i] / n + 1e-4 * w[i];
                L.mW[j][i] = b1 * L.mW[j][i] + (1 - b1) * g; L.vW[j][i] = b2 * L.vW[j][i] + (1 - b2) * g * g;
                w[i] -= p.lr * (L.mW[j][i] / c1) / (Math.sqrt(L.vW[j][i] / c2) + 1e-8);
              }));
              L.b.forEach((_, j) => {
                const g = gb[l][j] / n; L.mb[j] = b1 * L.mb[j] + (1 - b1) * g; L.vb[j] = b2 * L.vb[j] + (1 - b2) * g * g;
                L.b[j] -= p.lr * (L.mb[j] / c1) / (Math.sqrt(L.vb[j] / c2) + 1e-8);
              });
            });
            epoch = { n: epoch.n + 1, loss: loss / n };
          }
          return epoch.n >= 1500;
        };
        return m;
      },
      predict(m, x, y) { const hs = forward(m, x, y); return hs[hs.length - 1]; },
      sideExtra(side) { return stats(side, [["ep", "Epoch"], ["loss", "Loss"]]); },
      afterDraw(s, r) { r.ep.textContent = epoch.n; r.loss.textContent = epoch.loss.toFixed(3); },
    });
  };

  // =============== Gradient boosting (1-D regression) ===============
  DEMOS["gradient-boosting"] = (root) => {
    const { stage, side } = demoShell(root, "Each new tree is fitted to the leftover errors (residuals, the grey lines) of the trees before it. Add trees one at a time and watch the curve close in. Hollow points are held-out test data.");
    const plot = Plot(stage, { aspect: 1.5, xmin: 0, xmax: 6.3, ymin: -1.8, ymax: 1.8 });
    let pts = [], trees = [], depth = 2, lr = 0.3, lastTree = null;
    const truth = (x) => Math.sin(x) + 0.35 * Math.sin(3 * x);
    function data() { const r = rng(Math.floor(Math.random() * 1e6) + 3); pts = Array.from({ length: 70 }, () => { const x = r() * 6.3; return { x, y: truth(x) + 0.22 * r.normal(), test: r() < 0.3 }; }); }
    function fitStump(xs, rs, d) {
      if (d === 0 || xs.length < 4) return { v: rs.reduce((a, b) => a + b, 0) / Math.max(1, rs.length) };
      const idx = xs.map((_, i) => i).sort((a, b) => xs[a] - xs[b]);
      let best = null, sl = 0, sl2 = 0; const tot = rs.reduce((a, b) => a + b, 0), tot2 = rs.reduce((a, b) => a + b * b, 0);
      for (let k = 0; k < idx.length - 1; k++) {
        const r = rs[idx[k]]; sl += r; sl2 += r * r;
        const nl = k + 1, nr = idx.length - nl; if (nl < 2 || nr < 2) continue;
        const sse = sl2 - sl * sl / nl + (tot2 - sl2) - (tot - sl) ** 2 / nr;
        if (!best || sse < best.sse) best = { sse, thr: (xs[idx[k]] + xs[idx[k + 1]]) / 2 };
      }
      if (!best) return { v: tot / rs.length };
      const L = idx.filter((i) => xs[i] <= best.thr), R = idx.filter((i) => xs[i] > best.thr);
      return { thr: best.thr, l: fitStump(L.map((i) => xs[i]), L.map((i) => rs[i]), d - 1), r: fitStump(R.map((i) => xs[i]), R.map((i) => rs[i]), d - 1) };
    }
    const evalT = (t, x) => (t.thr === undefined ? t.v : evalT(x <= t.thr ? t.l : t.r, x));
    const base = () => { const tr = pts.filter((p) => !p.test); return tr.reduce((a, p) => a + p.y, 0) / tr.length; };
    const pred = (x) => trees.reduce((s, t) => s + lr * evalT(t, x), base());
    function addTree() {
      const tr = pts.filter((p) => !p.test);
      lastTree = fitStump(tr.map((p) => p.x), tr.map((p) => p.y - pred(p.x)), depth);
      trees.push(lastTree);
    }
    const mse = (set) => set.reduce((s, p) => s + (p.y - pred(p.x)) ** 2, 0) / Math.max(1, set.length);
    function curve(fn, color, w, dash) {
      const c = plot.ctx; c.beginPath(); c.setLineDash(dash || []);
      for (let i = 0; i <= 300; i++) { const x = (i / 300) * 6.3, y = fn(x); i ? c.lineTo(plot.px(x), plot.py(y)) : c.moveTo(plot.px(x), plot.py(y)); }
      c.strokeStyle = color; c.lineWidth = w; c.stroke(); c.setLineDash([]);
    }
    function draw() {
      const t = plot.clear(); plot.grid(0.5);
      pts.filter((p) => !p.test).forEach((p) => plot.line(p.x, p.y, p.x, pred(p.x), t.ink3, 1));
      if (lastTree) curve((x) => lr * evalT(lastTree, x), t.cls[1], 1.5, [5, 4]);
      curve(pred, t.accent, 2.5);
      pts.forEach((p) => plot.point(p.x, p.y, t.cls[0], { hollow: p.test, r: 4.5 }));
      refs.n.textContent = trees.length;
      refs.tr.textContent = mse(pts.filter((p) => !p.test)).toFixed(3);
      refs.te.textContent = mse(pts.filter((p) => p.test)).toFixed(3);
    }
    plot.onResize = draw;
    const leg = el("div", { class: "legend" }, `<span><i class="swatch" style="background:var(--accent)"></i>Ensemble prediction</span><span><i class="swatch" style="background:var(--c1)"></i>Newest tree × learning rate</span>`);
    side.appendChild(leg);
    slider(side, { label: "Tree depth", min: 1, max: 4, step: 1, value: 2, onInput: (v) => (depth = v) });
    slider(side, { label: "Learning rate", min: 0.05, max: 1, step: 0.05, value: 0.3, fmt: (v) => v.toFixed(2), onInput: (v) => { lr = v; trees = []; lastTree = null; draw(); } });
    const refs = stats(side, [["n", "Trees"], ["tr", "Train MSE"], ["te", "Test MSE", true]]);
    const b = el("div", { class: "btns" });
    button(b, "Add 1 tree", () => { addTree(); draw(); }, "btn small primary");
    button(b, "Add 10", () => { for (let i = 0; i < 10; i++) addTree(); draw(); });
    button(b, "Add 100", () => { for (let i = 0; i < 100; i++) addTree(); draw(); });
    button(b, "Reset", () => { trees = []; lastTree = null; draw(); });
    button(b, "New data", () => { data(); trees = []; lastTree = null; draw(); });
    side.appendChild(b);
    data(); draw();
  };

  // =============== K-Means ===============
  DEMOS["kmeans"] = (root) => {
    const { stage, side } = demoShell(root, "K-Means repeats two steps: assign every point to its nearest centre, then move each centre to the middle of its points. Step through it, or press Run.");
    const plot = Plot(stage, { aspect: 1.25, xmin: -1.25, xmax: 1.25, ymin: -1, ymax: 1 });
    let pts = [], cents = [], assign = [], k = 3, phase = "assign", iter = 0, trueK = 4, timer = null;
    function data() {
      const r = rng(Math.floor(Math.random() * 1e6) + 9);
      const centres = Array.from({ length: trueK }, () => [r() * 1.8 - 0.9, r() * 1.4 - 0.7]);
      pts = Array.from({ length: 160 }, (_, i) => { const c = centres[i % trueK]; return { x: c[0] + 0.14 * r.normal(), y: c[1] + 0.14 * r.normal() }; });
    }
    function init() {
      const r = rng(Math.floor(Math.random() * 1e6) + 1);
      cents = Array.from({ length: k }, () => { const p = pts[Math.floor(r() * pts.length)]; return { x: p.x, y: p.y, trail: [] }; });
      assign = pts.map(() => -1); phase = "assign"; iter = 0;
    }
    const nearest = (x, y) => { let b = 0, bd = Infinity; cents.forEach((c, i) => { const d = (c.x - x) ** 2 + (c.y - y) ** 2; if (d < bd) { bd = d; b = i; } }); return b; };
    function step() {
      if (phase === "assign") { const old = assign.join(); assign = pts.map((p) => nearest(p.x, p.y)); phase = "update"; iter++; return old === assign.join(); }
      cents.forEach((c, i) => { const s = pts.filter((_, j) => assign[j] === i); if (s.length) { c.trail.push([c.x, c.y]); c.x = s.reduce((a, p) => a + p.x, 0) / s.length; c.y = s.reduce((a, p) => a + p.y, 0) / s.length; } });
      phase = "assign"; return false;
    }
    function draw() {
      const t = plot.clear(), pal = palette(t);
      const bg = hexToRgb(t.paper);
      if (cents.length) plot.field((x, y) => { const c = hexToRgb(pal[nearest(x, y) % pal.length]); const a = 0.12; return [bg[0] * (1 - a) + c[0] * a, bg[1] * (1 - a) + c[1] * a, bg[2] * (1 - a) + c[2] * a, 255]; }, 100);
      plot.grid(0.25);
      pts.forEach((p, i) => plot.point(p.x, p.y, assign[i] >= 0 ? pal[assign[i] % pal.length] : t.ink3, { r: 4 }));
      cents.forEach((c, i) => {
        if (c.trail.length) { const ctx = plot.ctx; ctx.beginPath(); ctx.moveTo(plot.px(c.trail[0][0]), plot.py(c.trail[0][1])); c.trail.forEach(([x, y]) => ctx.lineTo(plot.px(x), plot.py(y))); ctx.lineTo(plot.px(c.x), plot.py(c.y)); ctx.strokeStyle = t.ink; ctx.lineWidth = 1.2; ctx.setLineDash([3, 3]); ctx.stroke(); ctx.setLineDash([]); }
        const X = plot.px(c.x), Y = plot.py(c.y), ctx = plot.ctx;
        ctx.save(); ctx.translate(X, Y); ctx.rotate(Math.PI / 4); ctx.fillStyle = pal[i % pal.length]; ctx.strokeStyle = t.ink; ctx.lineWidth = 2; ctx.fillRect(-7, -7, 14, 14); ctx.strokeRect(-7, -7, 14, 14); ctx.restore();
      });
      const inertia = assign[0] >= 0 ? pts.reduce((s, p, i) => s + (p.x - cents[assign[i]].x) ** 2 + (p.y - cents[assign[i]].y) ** 2, 0) : null;
      refs.it.textContent = iter; refs.in.textContent = inertia === null ? "–" : inertia.toFixed(2);
      refs.next.textContent = phase === "assign" ? "Assign points to nearest centre" : "Move centres to the mean";
    }
    plot.onResize = draw;
    slider(side, { label: "k (number of clusters)", min: 1, max: 8, step: 1, value: 3, onInput: (v) => { k = v; init(); draw(); } });
    slider(side, { label: "True groups in the data", min: 2, max: 6, step: 1, value: 4, onInput: (v) => { trueK = v; data(); init(); draw(); } });
    const refs = stats(side, [["it", "Iteration"], ["in", "Inertia"], ["next", "Next step", true]]);
    const b = el("div", { class: "btns" });
    button(b, "Step", () => { step(); draw(); }, "btn small primary");
    button(b, "Run", () => {
      clearInterval(timer);
      timer = setInterval(() => { const conv = step(); draw(); if (conv || iter > 50) { clearInterval(timer); refs.next.textContent = "Converged: assignments stopped changing"; } }, 350);
    });
    button(b, "New starting centres", () => { clearInterval(timer); init(); draw(); });
    button(b, "New data", () => { clearInterval(timer); data(); init(); draw(); });
    side.appendChild(b);
    data(); init(); draw();
  };

  // =============== PCA ===============
  DEMOS["pca"] = (root) => {
    const { stage, side } = demoShell(root, "Rotate the grey line to find the direction where the points are most spread out. That direction is the first principal component (PC1). Projections onto your line are shown as small marks.");
    const plot = Plot(stage, { aspect: 1.25, xmin: -2.5, xmax: 2.5, ymin: -2, ymax: 2 });
    let pts = [], angle = 0, corr = 0.8, showPC = false;
    function data() {
      const r = rng(Math.floor(Math.random() * 1e6) + 5);
      pts = Array.from({ length: 120 }, () => { const a = r.normal(), b = r.normal(); return { x: a * 0.9, y: (corr * a + Math.sqrt(1 - corr * corr) * b) * 0.9 }; });
    }
    function pca() {
      const n = pts.length, mx = pts.reduce((s, p) => s + p.x, 0) / n, my = pts.reduce((s, p) => s + p.y, 0) / n;
      let sxx = 0, syy = 0, sxy = 0; pts.forEach((p) => { sxx += (p.x - mx) ** 2; syy += (p.y - my) ** 2; sxy += (p.x - mx) * (p.y - my); });
      sxx /= n; syy /= n; sxy /= n;
      const tr = sxx + syy, det = sxx * syy - sxy * sxy, l1 = tr / 2 + Math.sqrt(tr * tr / 4 - det), l2 = tr / 2 - Math.sqrt(tr * tr / 4 - det);
      const a1 = Math.atan2(l1 - sxx, sxy || 1e-9);
      return { mx, my, l1, l2, a1, total: tr, varAt: (a) => sxx * Math.cos(a) ** 2 + syy * Math.sin(a) ** 2 + 2 * sxy * Math.sin(a) * Math.cos(a) };
    }
    function draw() {
      const t = plot.clear(); plot.grid(0.5);
      const P = pca(), ux = Math.cos(angle), uy = Math.sin(angle), L = 3;
      pts.forEach((p) => { const s = (p.x - P.mx) * ux + (p.y - P.my) * uy; plot.line(p.x, p.y, P.mx + s * ux, P.my + s * uy, rgba(t.ink3, 0.35), 1); });
      plot.line(P.mx - L * ux, P.my - L * uy, P.mx + L * ux, P.my + L * uy, t.ink2, 2);
      pts.forEach((p) => plot.point(p.x, p.y, t.cls[0], { r: 4 }));
      pts.forEach((p) => { const s = (p.x - P.mx) * ux + (p.y - P.my) * uy; plot.point(P.mx + s * ux, P.my + s * uy, t.cls[1], { r: 2.5 }); });
      if (showPC) {
        [[P.a1, P.l1, "PC1"], [P.a1 + Math.PI / 2, P.l2, "PC2"]].forEach(([a, l, name]) => {
          const len = 2 * Math.sqrt(l), x2 = P.mx + len * Math.cos(a), y2 = P.my + len * Math.sin(a);
          plot.line(P.mx, P.my, x2, y2, t.accent, 3);
          plot.text(name, plot.px(x2) + 6, plot.py(y2) - 6, t.accent, { size: 13 });
        });
      }
      const v = P.varAt(angle);
      refs.mine.textContent = (v / P.total * 100).toFixed(1) + "%";
      refs.pc1.textContent = (P.l1 / P.total * 100).toFixed(1) + "%";
      refs.pc2.textContent = (P.l2 / P.total * 100).toFixed(1) + "%";
      refs.gap.textContent = v / P.l1 > 0.995 ? "You found PC1" : `${(100 * (1 - v / P.l1)).toFixed(1)}% below the maximum`;
    }
    plot.onResize = draw;
    const ang = slider(side, { label: "Your direction", min: 0, max: 180, step: 1, value: 0, fmt: (v) => v + "°", onInput: (v) => { angle = (v * Math.PI) / 180; draw(); } });
    slider(side, { label: "Correlation between x and y", min: 0, max: 0.98, step: 0.02, value: 0.8, fmt: (v) => v.toFixed(2), onInput: (v) => { corr = v; data(); draw(); } });
    const refs = stats(side, [["mine", "Variance on your line"], ["pc1", "PC1 keeps"], ["pc2", "PC2 keeps"], ["gap", "Status", true]]);
    checkbox(side, { label: "Show principal components", checked: false, onChange: (v) => { showPC = v; draw(); } });
    const b = el("div", { class: "btns" });
    button(b, "Snap to PC1", () => { const P = pca(); let a = P.a1; while (a < 0) a += Math.PI; while (a >= Math.PI) a -= Math.PI; angle = a; ang.set(Math.round((a * 180) / Math.PI)); draw(); }, "btn small primary");
    button(b, "New data", () => { data(); draw(); });
    side.appendChild(b);
    data(); draw();
  };

  // =============== Concepts: overfitting ===============
  DEMOS["overfit"] = (root) => {
    const { stage, side } = demoShell(root, "Fit a polynomial to 14 noisy training points. Low degrees underfit; high degrees chase the noise. Watch the training error keep falling while the test error (hollow points) turns back up.");
    const plot = Plot(stage, { aspect: 1.5, xmin: -1, xmax: 1, ymin: -1.6, ymax: 1.6 });
    const mini = el("canvas", { style: "width:100%;height:120px;display:block;border-radius:8px", "aria-label": "Error versus degree" });
    let train = [], test = [], deg = 3;
    const truth = (x) => 0.9 * Math.sin(Math.PI * x);
    function data() {
      const r = rng(Math.floor(Math.random() * 1e6) + 2);
      train = Array.from({ length: 14 }, () => { const x = r() * 1.9 - 0.95; return { x, y: truth(x) + 0.22 * r.normal() }; });
      test = Array.from({ length: 40 }, () => { const x = r() * 1.9 - 0.95; return { x, y: truth(x) + 0.22 * r.normal() }; });
    }
    function fitPoly(d) {
      const A = Array.from({ length: d + 1 }, () => new Array(d + 1).fill(0)), b = new Array(d + 1).fill(0);
      train.forEach((p) => { const f = Array.from({ length: d + 1 }, (_, i) => p.x ** i); for (let i = 0; i <= d; i++) { b[i] += f[i] * p.y; for (let j = 0; j <= d; j++) A[i][j] += f[i] * f[j]; } });
      for (let i = 0; i <= d; i++) A[i][i] += 1e-9;
      const w = solve(A, b);
      return (x) => w.reduce((s, wi, i) => s + wi * x ** i, 0);
    }
    const mse = (f, s) => s.reduce((a, p) => a + (p.y - f(p.x)) ** 2, 0) / s.length;
    function draw() {
      const t = plot.clear(); plot.grid(0.25);
      const f = fitPoly(deg), c = plot.ctx;
      c.beginPath(); for (let i = 0; i <= 400; i++) { const x = -1 + (2 * i) / 400; const y = Math.max(-5, Math.min(5, f(x))); i ? c.lineTo(plot.px(x), plot.py(y)) : c.moveTo(plot.px(x), plot.py(y)); }
      c.strokeStyle = t.accent; c.lineWidth = 2.5; c.stroke();
      c.beginPath(); for (let i = 0; i <= 200; i++) { const x = -1 + (2 * i) / 200; i ? c.lineTo(plot.px(x), plot.py(truth(x))) : c.moveTo(plot.px(x), plot.py(truth(x))); }
      c.strokeStyle = t.ink3; c.lineWidth = 1; c.setLineDash([4, 4]); c.stroke(); c.setLineDash([]);
      test.forEach((p) => plot.point(p.x, p.y, t.cls[1], { hollow: true, r: 3.5 }));
      train.forEach((p) => plot.point(p.x, p.y, t.cls[0], { r: 5 }));
      const tr = mse(f, train), te = mse(f, test);
      refs.tr.textContent = tr.toFixed(3); refs.te.textContent = te.toFixed(3);
      refs.v.textContent = deg <= 1 ? "Underfitting: too simple" : te > 2 * tr + 0.06 && deg >= 7 ? "Overfitting: memorising noise" : "Good balance";
      // error vs degree chart
      const w = mini.clientWidth || 260, h = 120, dpr = Math.min(window.devicePixelRatio || 1, 2);
      mini.width = w * dpr; mini.height = h * dpr; const m = mini.getContext("2d"); m.setTransform(dpr, 0, 0, dpr, 0, 0);
      m.fillStyle = t.paper; m.fillRect(0, 0, w, h);
      const errs = Array.from({ length: 13 }, (_, i) => { const g = fitPoly(i + 1); return [mse(g, train), Math.min(1, mse(g, test))]; });
      const X = (i) => 26 + (i / 12) * (w - 36), Y = (v) => h - 18 - (Math.min(v, 0.6) / 0.6) * (h - 28);
      m.strokeStyle = t.rule; m.beginPath(); m.moveTo(26, 8); m.lineTo(26, h - 18); m.lineTo(w - 8, h - 18); m.stroke();
      [[0, t.cls[0]], [1, t.cls[1]]].forEach(([s, col]) => { m.beginPath(); errs.forEach((e, i) => (i ? m.lineTo(X(i), Y(e[s])) : m.moveTo(X(i), Y(e[s])))); m.strokeStyle = col; m.lineWidth = 2; m.stroke(); });
      m.strokeStyle = t.ink; m.setLineDash([3, 3]); m.beginPath(); m.moveTo(X(deg - 1), 8); m.lineTo(X(deg - 1), h - 18); m.stroke(); m.setLineDash([]);
      m.fillStyle = t.ink3; m.font = "10px sans-serif"; m.textAlign = "center"; m.fillText("degree 1", X(0), h - 4); m.fillText("13", X(12), h - 4);
      m.save(); m.translate(10, h / 2); m.rotate(-Math.PI / 2); m.fillText("error", 0, 0); m.restore();
    }
    plot.onResize = draw;
    slider(side, { label: "Polynomial degree", min: 1, max: 13, step: 1, value: 3, onInput: (v) => { deg = v; draw(); } });
    const refs = stats(side, [["tr", "Train error"], ["te", "Test error"], ["v", "Diagnosis", true]]);
    side.appendChild(el("div", { class: "legend" }, `<span><i class="swatch" style="background:var(--c0)"></i>Train error</span><span><i class="swatch" style="background:var(--c1)"></i>Test error</span>`));
    side.appendChild(mini);
    const b = el("div", { class: "btns" }); button(b, "New data", () => { data(); draw(); }); side.appendChild(b);
    data(); draw();
  };

  // =============== Concepts: threshold, precision and recall ===============
  DEMOS["threshold"] = (root) => {
    const { stage, side } = demoShell(root, "A classifier gives every example a score. Everything above the threshold is called positive. Move the threshold and the positive share to see why accuracy alone can mislead.");
    const plot = Plot(stage, { aspect: 1.7, xmin: 0, xmax: 1, ymin: 0, ymax: 1 });
    let thr = 0.5, share = 0.1;
    const N = 1000;
    const pdf = (x, m, s) => Math.exp(-0.5 * ((x - m) / s) ** 2) / (s * Math.sqrt(2 * Math.PI));
    const cdf = (x, m, s) => 0.5 * (1 + erf((x - m) / (s * Math.SQRT2)));
    function erf(x) { const t = 1 / (1 + 0.3275911 * Math.abs(x)); const y = 1 - ((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * Math.exp(-x * x); return x >= 0 ? y : -y; }
    const NEG = [0.35, 0.12], POS = [0.68, 0.12];
    function draw() {
      const t = plot.clear(); plot.grid(0.1, false);
      const nPos = Math.round(N * share), nNeg = N - nPos;
      const peak = Math.max(nNeg * pdf(NEG[0], ...NEG), nPos * pdf(POS[0], ...POS));
      [[NEG, nNeg, t.cls[0]], [POS, nPos, t.cls[1]]].forEach(([d, n, col]) => {
        const c = plot.ctx; c.beginPath(); c.moveTo(plot.px(0), plot.py(0));
        for (let i = 0; i <= 200; i++) { const x = i / 200; c.lineTo(plot.px(x), plot.py((0.9 * n * pdf(x, ...d)) / peak)); }
        c.lineTo(plot.px(1), plot.py(0)); c.closePath(); c.fillStyle = rgba(col, 0.25); c.fill(); c.strokeStyle = col; c.lineWidth = 2; c.stroke();
      });
      plot.line(thr, 0, thr, 1, t.ink, 2, [6, 4]);
      plot.text("predicted positive →", plot.px(thr) + 6, 16, t.ink2, { size: 11 });
      plot.text("score", plot.W - 8, plot.H - 6, t.ink3, { align: "right", size: 11 });
      const tp = nPos * (1 - cdf(thr, ...POS)), fn = nPos - tp, fp = nNeg * (1 - cdf(thr, ...NEG)), tn = nNeg - fp;
      const acc = (tp + tn) / N, prec = tp + fp ? tp / (tp + fp) : 0, rec = nPos ? tp / nPos : 0;
      refs.acc.textContent = fmtPct(acc); refs.base.textContent = fmtPct(nNeg / N);
      refs.prec.textContent = fmtPct(prec); refs.rec.textContent = fmtPct(rec);
      cm.innerHTML = `<table><tr><th></th><th class="num">Said no</th><th class="num">Said yes</th></tr>
        <tr><th>Really no</th><td class="num">${Math.round(tn)}</td><td class="num">${Math.round(fp)}</td></tr>
        <tr><th>Really yes</th><td class="num">${Math.round(fn)}</td><td class="num">${Math.round(tp)}</td></tr></table>`;
    }
    plot.onResize = draw;
    side.appendChild(el("div", { class: "legend" }, `<span><i class="swatch" style="background:var(--c0)"></i>Negatives</span><span><i class="swatch" style="background:var(--c1)"></i>Positives</span>`));
    slider(side, { label: "Threshold", min: 0.05, max: 0.95, step: 0.01, value: 0.5, fmt: (v) => v.toFixed(2), onInput: (v) => { thr = v; draw(); } });
    slider(side, { label: "Positives in the data", min: 0.01, max: 0.5, step: 0.01, value: 0.1, fmt: fmtPct, onInput: (v) => { share = v; draw(); } });
    const refs = stats(side, [["acc", "Accuracy"], ["base", "Always “negative”"], ["prec", "Precision"], ["rec", "Recall"]]);
    const cm = el("div", { class: "table-wrap cm", "aria-label": "Confusion matrix out of 1,000 examples" }); side.appendChild(cm);
    draw();
  };

  document.addEventListener("DOMContentLoaded", () => {
    document.querySelectorAll("[data-demo]").forEach((node) => {
      const fn = DEMOS[node.dataset.demo];
      if (!fn) return;
      try { fn(node); } catch (e) { node.textContent = "This demo failed to load: " + e.message; console.error(e); }
    });
  });
  window.DEMOS = DEMOS;
})();
