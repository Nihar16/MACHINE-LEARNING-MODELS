/* Plotting engine shared by every demo: a theme-aware graph-paper canvas, seeded random
   data sets, small UI builders and a reusable 2-D classifier playground. */
(function () {
  "use strict";

  // ---------- colours from CSS tokens ----------
  function tokens() {
    const cs = getComputedStyle(document.documentElement);
    const g = (n) => cs.getPropertyValue(n).trim();
    return { paper: g("--paper"), panel: g("--panel"), ink: g("--ink"), ink2: g("--ink-2"), ink3: g("--ink-3"),
      grid: g("--grid"), rule: g("--rule"), accent: g("--accent"), cls: [g("--c0"), g("--c1"), g("--c2"), g("--c3")],
      good: g("--good"), bad: g("--bad") };
  }
  function hexToRgb(h) {
    h = h.replace("#", "");
    if (h.length === 3) h = h.split("").map((c) => c + c).join("");
    const n = parseInt(h, 16);
    return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
  }
  function rgba(hex, a) { const [r, g, b] = hexToRgb(hex); return `rgba(${r},${g},${b},${a})`; }

  // ---------- seeded random ----------
  function rng(seed) {
    let s = seed >>> 0 || 1;
    const r = () => { s ^= s << 13; s ^= s >>> 17; s ^= s << 5; return ((s >>> 0) % 1e9) / 1e9; };
    r.normal = () => { let u = 0, v = 0; while (!u) u = r(); while (!v) v = r(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v); };
    return r;
  }

  // ---------- data sets in [-1, 1]^2 ----------
  const DATASETS = {
    blobs(r, n = 80) {
      const pts = [];
      const c = [[-0.45, 0.35], [0.45, -0.3]];
      for (let i = 0; i < n; i++) { const k = i % 2; pts.push({ x: c[k][0] + 0.22 * r.normal(), y: c[k][1] + 0.22 * r.normal(), c: k }); }
      return pts;
    },
    overlap(r, n = 90) {
      const pts = [];
      const c = [[-0.2, 0.15], [0.2, -0.12]];
      for (let i = 0; i < n; i++) { const k = i % 2; pts.push({ x: c[k][0] + 0.3 * r.normal(), y: c[k][1] + 0.3 * r.normal(), c: k }); }
      return pts;
    },
    moons(r, n = 100) {
      const pts = [];
      for (let i = 0; i < n; i++) {
        const k = i % 2, t = Math.PI * r();
        const x = k ? 1 - Math.cos(t) : Math.cos(t), y = k ? 0.5 - Math.sin(t) : Math.sin(t);
        pts.push({ x: (x - 0.5) * 0.62 + 0.08 * r.normal(), y: (y - 0.25) * 0.62 + 0.08 * r.normal(), c: k });
      }
      return pts;
    },
    circles(r, n = 110) {
      const pts = [];
      for (let i = 0; i < n; i++) {
        const k = i % 2, t = 2 * Math.PI * r(), rad = (k ? 0.75 : 0.3) + 0.08 * r.normal();
        pts.push({ x: rad * Math.cos(t), y: rad * Math.sin(t), c: k });
      }
      return pts;
    },
    xor(r, n = 120) {
      const pts = [];
      for (let i = 0; i < n; i++) {
        const x = r() * 1.8 - 0.9, y = r() * 1.8 - 0.9;
        if (Math.abs(x) < 0.06 || Math.abs(y) < 0.06) { i--; continue; }
        pts.push({ x, y, c: (x > 0) !== (y > 0) ? 1 : 0 });
      }
      return pts;
    },
    spiral(r, n = 160) {
      const pts = [];
      for (let i = 0; i < n; i++) {
        const k = i % 2, t = (i / n) * 3.2 + 0.25, rad = t / 3.6;
        const a = t * 2.2 + k * Math.PI;
        pts.push({ x: rad * Math.cos(a) + 0.03 * r.normal(), y: rad * Math.sin(a) + 0.03 * r.normal(), c: k });
      }
      return pts;
    },
    three(r, n = 120) {
      const pts = [];
      const c = [[-0.5, -0.35], [0.5, -0.3], [0, 0.5]];
      for (let i = 0; i < n; i++) { const k = i % 3; pts.push({ x: c[k][0] + 0.2 * r.normal(), y: c[k][1] + 0.2 * r.normal(), c: k }); }
      return pts;
    },
  };
  const DATASET_LABELS = { blobs: "Two blobs", overlap: "Overlapping blobs", moons: "Moons", circles: "Circles", xor: "XOR (checkerboard)", spiral: "Spiral", three: "Three classes" };

  // ---------- canvas with a data coordinate system ----------
  function Plot(container, opts = {}) {
    const cv = document.createElement("canvas");
    container.appendChild(cv);
    const ctx = cv.getContext("2d");
    const self = {
      cv, ctx, xmin: opts.xmin ?? -1, xmax: opts.xmax ?? 1, ymin: opts.ymin ?? -1, ymax: opts.ymax ?? 1,
      aspect: opts.aspect ?? 1, W: 0, H: 0, dpr: 1, onResize: null,
    };
    self.resize = () => {
      const w = Math.max(200, container.clientWidth);
      const h = Math.round(w / self.aspect);
      self.dpr = Math.min(window.devicePixelRatio || 1, 2);
      cv.width = Math.round(w * self.dpr); cv.height = Math.round(h * self.dpr);
      cv.style.height = h + "px";
      self.W = w; self.H = h;
      ctx.setTransform(self.dpr, 0, 0, self.dpr, 0, 0);
      if (self.onResize) self.onResize();
    };
    self.px = (x) => ((x - self.xmin) / (self.xmax - self.xmin)) * self.W;
    self.py = (y) => self.H - ((y - self.ymin) / (self.ymax - self.ymin)) * self.H;
    self.dx = (px) => self.xmin + (px / self.W) * (self.xmax - self.xmin);
    self.dy = (py) => self.ymin + ((self.H - py) / self.H) * (self.ymax - self.ymin);
    self.clear = () => {
      const t = tokens();
      ctx.fillStyle = t.paper; ctx.fillRect(0, 0, self.W, self.H);
      return t;
    };
    self.grid = (step = 0.25, axes = true) => {
      const t = tokens();
      ctx.lineWidth = 1; ctx.strokeStyle = t.grid; ctx.beginPath();
      for (let x = Math.ceil(self.xmin / step) * step; x <= self.xmax + 1e-9; x += step) { const p = Math.round(self.px(x)) + 0.5; ctx.moveTo(p, 0); ctx.lineTo(p, self.H); }
      for (let y = Math.ceil(self.ymin / step) * step; y <= self.ymax + 1e-9; y += step) { const p = Math.round(self.py(y)) + 0.5; ctx.moveTo(0, p); ctx.lineTo(self.W, p); }
      ctx.stroke();
      if (axes) {
        ctx.strokeStyle = t.rule; ctx.beginPath();
        if (self.xmin < 0 && self.xmax > 0) { const p = Math.round(self.px(0)) + 0.5; ctx.moveTo(p, 0); ctx.lineTo(p, self.H); }
        if (self.ymin < 0 && self.ymax > 0) { const p = Math.round(self.py(0)) + 0.5; ctx.moveTo(0, p); ctx.lineTo(self.W, p); }
        ctx.stroke();
      }
    };
    self.point = (x, y, color, { r = 5, hollow = false, ring = null } = {}) => {
      const t = tokens();
      ctx.beginPath(); ctx.arc(self.px(x), self.py(y), r, 0, 2 * Math.PI);
      if (hollow) { ctx.fillStyle = t.paper; ctx.fill(); ctx.lineWidth = 2; ctx.strokeStyle = color; ctx.stroke(); }
      else { ctx.fillStyle = color; ctx.fill(); ctx.lineWidth = 1.2; ctx.strokeStyle = t.paper; ctx.stroke(); }
      if (ring) { ctx.beginPath(); ctx.arc(self.px(x), self.py(y), r + 4, 0, 2 * Math.PI); ctx.lineWidth = 1.5; ctx.strokeStyle = ring; ctx.stroke(); }
    };
    self.line = (x1, y1, x2, y2, color, width = 2, dash = null) => {
      ctx.beginPath(); ctx.setLineDash(dash || []); ctx.moveTo(self.px(x1), self.py(y1)); ctx.lineTo(self.px(x2), self.py(y2));
      ctx.lineWidth = width; ctx.strokeStyle = color; ctx.stroke(); ctx.setLineDash([]);
    };
    self.text = (s, x, y, color, { align = "left", size = 12, base = "alphabetic", mono = true } = {}) => {
      ctx.font = `${size}px ${mono ? getComputedStyle(document.documentElement).getPropertyValue("--font-mono") : getComputedStyle(document.documentElement).getPropertyValue("--font-body")}`;
      ctx.fillStyle = color; ctx.textAlign = align; ctx.textBaseline = base; ctx.fillText(s, x, y);
    };
    /* Shade the plane: fn(x, y) -> [r, g, b, a] (0-255). Rendered at low resolution, then scaled. */
    self.field = (fn, res = 80) => {
      const rows = Math.round(res / self.aspect);
      const img = ctx.createImageData(res, rows);
      for (let j = 0; j < rows; j++) for (let i = 0; i < res; i++) {
        const x = self.xmin + ((i + 0.5) / res) * (self.xmax - self.xmin);
        const y = self.ymax - ((j + 0.5) / rows) * (self.ymax - self.ymin);
        const c = fn(x, y), o = 4 * (j * res + i);
        img.data[o] = c[0]; img.data[o + 1] = c[1]; img.data[o + 2] = c[2]; img.data[o + 3] = c[3];
      }
      const off = document.createElement("canvas"); off.width = res; off.height = rows;
      off.getContext("2d").putImageData(img, 0, 0);
      ctx.save(); ctx.imageSmoothingEnabled = true; ctx.drawImage(off, 0, 0, self.W, self.H); ctx.restore();
    };
    self.eventPos = (e) => {
      const r = cv.getBoundingClientRect();
      const p = e.touches ? e.touches[0] : e;
      return { x: self.dx(p.clientX - r.left), y: self.dy(p.clientY - r.top), px: p.clientX - r.left, py: p.clientY - r.top };
    };
    new ResizeObserver(() => self.resize()).observe(container);
    window.addEventListener("themechange", () => self.onResize && self.onResize());
    self.resize();
    return self;
  }

  // ---------- tiny UI builders ----------
  function el(tag, attrs = {}, html = "") {
    const e = document.createElement(tag);
    for (const [k, v] of Object.entries(attrs)) { if (k === "class") e.className = v; else e.setAttribute(k, v); }
    if (html) e.innerHTML = html;
    return e;
  }
  let uid = 0;
  function slider(parent, { label, min, max, step, value, fmt = (v) => v, onInput }) {
    const id = "s" + (++uid);
    const w = el("div", { class: "ctrl" });
    w.innerHTML = `<label for="${id}">${label} <output>${fmt(value)}</output></label><input type="range" id="${id}" min="${min}" max="${max}" step="${step}" value="${value}">`;
    const inp = w.querySelector("input"), out = w.querySelector("output");
    inp.addEventListener("input", () => { out.textContent = fmt(Number(inp.value)); onInput(Number(inp.value)); });
    parent.appendChild(w);
    return { input: inp, set(v) { inp.value = v; out.textContent = fmt(v); } };
  }
  function select(parent, { label, options, value, onChange }) {
    const id = "s" + (++uid);
    const w = el("div", { class: "ctrl" });
    w.innerHTML = `<label for="${id}">${label}</label><select id="${id}">${options.map(([v, t]) => `<option value="${v}"${v === value ? " selected" : ""}>${t}</option>`).join("")}</select>`;
    const s = w.querySelector("select");
    s.addEventListener("change", () => onChange(s.value));
    parent.appendChild(w);
    return s;
  }
  function checkbox(parent, { label, checked, onChange }) {
    const id = "s" + (++uid);
    const w = el("label", { class: "check", for: id });
    w.innerHTML = `<input type="checkbox" id="${id}"${checked ? " checked" : ""}> ${label}`;
    const c = w.querySelector("input");
    c.addEventListener("change", () => onChange(c.checked));
    parent.appendChild(w);
    return c;
  }
  function button(parent, text, onClick, cls = "btn small") {
    const b = el("button", { class: cls, type: "button" }, text);
    b.addEventListener("click", onClick);
    parent.appendChild(b);
    return b;
  }
  function stats(parent, items) {
    const w = el("div", { class: "stats" });
    const refs = {};
    items.forEach(([key, label, wide]) => {
      const s = el("div", { class: "stat" + (wide ? " wide" : "") }, `<span>${label}</span><strong>–</strong>`);
      refs[key] = s.querySelector("strong");
      w.appendChild(s);
    });
    parent.appendChild(w);
    return refs;
  }
  function demoShell(root, hint) {
    root.classList.add("demo");
    const stage = el("div", { class: "demo-stage" });
    const side = el("div", { class: "demo-side" });
    if (hint) side.appendChild(el("p", { class: "demo-hint" }, hint));
    root.append(stage, side);
    return { stage, side };
  }

  /* ---------- 2-D classifier playground ----------
     spec: { hint, datasets, dataset, params: [{key,label,min,max,step,value,fmt}|{key,label,type:'select'|'check',...}],
             fit(train, params) -> model, predict(model, x, y) -> probs[], overlay(plot, model, state), sideExtra(side, state),
             animate: bool (fit returns a stepper with .step() -> bool done) } */
  function Playground(root, spec) {
    const { stage, side } = demoShell(root, spec.hint);
    const plot = Plot(stage, { aspect: 1.25, xmin: -1.25, xmax: 1.25, ymin: -1, ymax: 1 });
    const state = { pts: [], seed: 7, dataset: spec.dataset || "blobs", addClass: 0, params: {}, model: null, showTest: true };
    (spec.params || []).forEach((p) => (state.params[p.key] = p.value));
    const nClasses = () => Math.max(2, ...state.pts.map((p) => p.c + 1));

    function load() {
      const r = rng(state.seed);
      state.pts = DATASETS[state.dataset](r).map((p) => ({ ...p, test: r() < 0.25 }));
      refit();
    }
    let anim = null;
    function refit() {
      if (anim) { cancelAnimationFrame(anim); anim = null; }
      const train = state.pts.filter((p) => !p.test);
      if (train.length < 2 || new Set(train.map((p) => p.c)).size < 2) { state.model = null; draw(); return; }
      state.model = spec.fit(train, state.params, nClasses());
      if (spec.animate && state.model && state.model.step) {
        const tick = () => { const finished = state.model.step(); draw(); if (!finished) anim = requestAnimationFrame(tick); else anim = null; };
        anim = requestAnimationFrame(tick);
      } else draw();
    }
    function accuracy(set) {
      if (!state.model || !set.length) return null;
      let ok = 0;
      set.forEach((p) => { const pr = spec.predict(state.model, p.x, p.y); if (argmax(pr) === p.c) ok++; });
      return ok / set.length;
    }
    function draw() {
      const t = plot.clear();
      const cls = t.cls.map(hexToRgb);
      const bg = hexToRgb(t.paper);
      if (state.model) {
        plot.field((x, y) => {
          const pr = spec.predict(state.model, x, y);
          const k = argmax(pr);
          const conf = nClasses() === 2 ? Math.abs(pr[1] - pr[0]) : (pr[k] - 1 / nClasses()) / (1 - 1 / nClasses());
          const a = 0.1 + 0.32 * Math.min(1, Math.max(0, conf));
          return [bg[0] * (1 - a) + cls[k][0] * a, bg[1] * (1 - a) + cls[k][1] * a, bg[2] * (1 - a) + cls[k][2] * a, 255];
        }, spec.res || 90);
      }
      plot.grid(0.25);
      if (spec.overlay && state.model) spec.overlay(plot, state.model, state, t);
      state.pts.forEach((p) => { if (!p.test || state.showTest) plot.point(p.x, p.y, t.cls[p.c], { hollow: p.test, r: p.test ? 4.5 : 5 }); });
      if (spec.overlayTop && state.model) spec.overlayTop(plot, state.model, state, t);
      const tr = accuracy(state.pts.filter((p) => !p.test)), te = accuracy(state.pts.filter((p) => p.test));
      refs.train.textContent = tr === null ? "–" : (tr * 100).toFixed(0) + "%";
      refs.test.textContent = te === null ? "–" : (te * 100).toFixed(0) + "%";
      if (spec.afterDraw) spec.afterDraw(state, extraRefs);
    }
    plot.onResize = draw;

    // controls
    select(side, { label: "Data set", options: (spec.datasets || ["blobs", "moons", "circles", "xor"]).map((d) => [d, DATASET_LABELS[d]]), value: state.dataset,
      onChange: (v) => { state.dataset = v; load(); } });
    (spec.params || []).forEach((p) => {
      if (p.type === "select") select(side, { label: p.label, options: p.options, value: p.value, onChange: (v) => { state.params[p.key] = v; refit(); } });
      else if (p.type === "check") checkbox(side, { label: p.label, checked: p.value, onChange: (v) => { state.params[p.key] = v; refit(); } });
      else slider(side, { ...p, onInput: (v) => { state.params[p.key] = v; refit(); } });
    });
    const seg = el("div", { class: "ctrl" }, `<span class="lab">Click the plot to add</span>`);
    const segb = el("div", { class: "seg" });
    const classBtns = [0, 1, 2].slice(0, spec.maxClasses || 2).map((k) => {
      const b = el("button", { type: "button", "aria-pressed": k === 0 ? "true" : "false" }, `<i class="swatch" style="background:var(--c${k})"></i>Class ${"ABC"[k]}`);
      b.addEventListener("click", () => { state.addClass = k; classBtns.forEach((x, j) => x.setAttribute("aria-pressed", j === k ? "true" : "false")); });
      segb.appendChild(b);
      return b;
    });
    seg.appendChild(segb);
    side.appendChild(seg);
    const refs = stats(side, [["train", "Train accuracy"], ["test", "Test accuracy"]]);
    const extraRefs = spec.sideExtra ? spec.sideExtra(side, state) : {};
    const btns = el("div", { class: "btns" });
    button(btns, "New sample", () => { state.seed = Math.floor(Math.random() * 1e6); load(); });
    button(btns, "Clear points", () => { state.pts = []; state.model = null; draw(); });
    side.appendChild(btns);
    checkbox(side, { label: "Show test points (hollow)", checked: true, onChange: (v) => { state.showTest = v; draw(); } });

    plot.cv.addEventListener("pointerdown", (e) => {
      const p = plot.eventPos(e);
      state.pts.push({ x: p.x, y: p.y, c: state.addClass, test: false });
      refit();
    });
    plot.cv.addEventListener("pointermove", (e) => { if (spec.onHover) { spec.onHover(plot.eventPos(e), state); draw(); } });
    plot.cv.addEventListener("pointerleave", () => { if (spec.onHover) { spec.onHover(null, state); draw(); } });
    load();
    return { state, refit, draw, plot };
  }

  function argmax(a) { let k = 0; for (let i = 1; i < a.length; i++) if (a[i] > a[k]) k = i; return k; }

  window.ML = { tokens, rgba, hexToRgb, rng, DATASETS, Plot, el, slider, select, checkbox, button, stats, demoShell, Playground, argmax };
})();
