/* Site behaviour: theme toggle, learning progress, quizzes, model chooser, k-fold diagram. */
(function () {
  "use strict";
  const store = {
    get(k, d) { try { const v = localStorage.getItem(k); return v === null ? d : JSON.parse(v); } catch (e) { return d; } },
    set(k, v) { try { localStorage.setItem(k, JSON.stringify(v)); } catch (e) { /* storage blocked */ } },
  };
  window.SiteStore = store;

  // ---- theme: explicit choice wins, otherwise follow the OS ----
  const saved = store.get("theme", null);
  if (saved) document.documentElement.setAttribute("data-theme", saved);
  function currentTheme() {
    const t = document.documentElement.getAttribute("data-theme");
    if (t) return t;
    return window.matchMedia && matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
  }
  function paintThemeBtn(btn) {
    const dark = currentTheme() === "dark";
    btn.innerHTML = dark
      ? '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="4"/><path d="M12 2v2M12 20v2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M2 12h2M20 12h2M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4"/></svg>'
      : '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 12.8A9 9 0 1 1 11.2 3a7 7 0 0 0 9.8 9.8z"/></svg>';
    btn.setAttribute("aria-label", dark ? "Switch to light theme" : "Switch to dark theme");
  }

  // ---- progress ----
  const done = () => new Set(store.get("learned", []));
  function markLearned(slug) {
    const s = done(); s.add(slug); store.set("learned", [...s]); paintProgress();
  }
  function paintProgress() {
    const s = done();
    document.querySelectorAll("[data-learned-slug]").forEach((el) => { el.hidden = !s.has(el.dataset.learnedSlug); });
    const c = document.getElementById("progress-count");
    if (c) c.textContent = String(s.size);
  }
  window.SiteProgress = { markLearned, done };

  // ---- quiz ----
  function initQuiz(root) {
    const qs = [...root.querySelectorAll(".q")];
    const score = root.querySelector(".quiz-score");
    let answered = 0, right = 0;
    qs.forEach((q) => {
      const answer = Number(q.dataset.answer);
      const btns = [...q.querySelectorAll(".q-opts button")];
      const explain = q.querySelector(".q-explain");
      btns.forEach((b, i) => b.addEventListener("click", () => {
        btns.forEach((x, j) => { x.disabled = true; if (j === answer) x.classList.add("right"); });
        if (i !== answer) b.classList.add("wrong"); else right++;
        answered++;
        explain.hidden = false;
        if (score) {
          score.textContent = `${right} of ${qs.length} correct` + (answered === qs.length ? (right === qs.length ? ". Marked as learned." : ". Reload the page to try again.") : "");
        }
        if (answered === qs.length && right === qs.length && root.dataset.slug) markLearned(root.dataset.slug);
      }));
    });
  }

  // ---- model chooser ----
  const TREE = {
    start: { q: "Do your examples have a label (the answer you want to predict)?", opts: [["Yes", "task"], ["No", "unsup"]] },
    unsup: { q: "What do you want to do with the data?", opts: [["Find groups", "r:kmeans"], ["Reduce the number of features", "r:pca"]] },
    task: { q: "What kind of answer?", opts: [["A number (price, temperature)", "num"], ["A category (spam / not spam)", "cat"], ["A category, and my inputs are images", "r:cnn"]] },
    num: { q: "Is the relationship roughly a straight line?", opts: [["Yes, or I need to explain it", "r:linear-regression"], ["No, it's complicated", "r:gradient-boosting"]] },
    cat: { q: "Do you need to explain each prediction?", opts: [["Yes", "explain"], ["No, accuracy matters most", "tab"]] },
    explain: { q: "Which explanation suits your audience?", opts: [["Feature weights", "r:logistic-regression"], ["If/then rules", "r:decision-tree"]] },
    tab: { q: "What does your data look like?", opts: [["A table of mixed columns", "r:gradient-boosting"], ["Text (word counts)", "r:naive-bayes"], ["Few rows, many numeric features", "r:svm"], ["Lots of rows with complex patterns", "r:neural-network"]] },
  };
  const WHY = {
    "kmeans": "K-Means groups similar rows without labels. Try several values of k.",
    "pca": "PCA compresses many correlated features into a few components that keep most of the variance.",
    "cnn": "Convolutional networks are built for images: they learn edges, strokes and shapes on their own.",
    "linear-regression": "Linear regression is fast, a strong baseline, and every coefficient has a meaning.",
    "gradient-boosting": "Gradient boosting (XGBoost / LightGBM) usually wins on tabular data with non-linear patterns.",
    "logistic-regression": "Logistic regression gives probabilities and a weight per feature you can read.",
    "decision-tree": "A shallow decision tree is a flowchart anyone can follow.",
    "naive-bayes": "Naive Bayes is fast and strong on word counts, even with little training data.",
    "svm": "SVMs find the widest margin between classes and work well when features outnumber rows.",
    "neural-network": "Neural networks learn complex patterns when you have enough data.",
  };
  function initChooser(root) {
    const qEl = root.querySelector(".chooser-q"), optsEl = root.querySelector(".chooser-opts");
    const trailEl = root.querySelector(".chooser-trail"), resEl = root.querySelector(".chooser-result");
    const names = JSON.parse(root.dataset.names || "{}");
    let trail = [];
    function show(node) {
      resEl.hidden = true; resEl.innerHTML = "";
      if (node.startsWith("r:")) {
        const slug = node.slice(2);
        qEl.textContent = "Start with:";
        optsEl.innerHTML = "";
        resEl.hidden = false;
        resEl.innerHTML = `<h3><a href="${root.dataset.base || ""}models/${slug}.html">${names[slug] || slug}</a></h3><p class="muted">${WHY[slug] || ""}</p>`;
        const again = document.createElement("button");
        again.className = "btn small"; again.textContent = "Start over";
        again.addEventListener("click", () => { trail = []; show("start"); });
        resEl.appendChild(again);
      } else {
        const n = TREE[node];
        qEl.textContent = n.q;
        optsEl.innerHTML = "";
        n.opts.forEach(([label, next]) => {
          const b = document.createElement("button");
          b.className = "btn"; b.textContent = label;
          b.addEventListener("click", () => { trail.push(label); show(next); });
          optsEl.appendChild(b);
        });
      }
      trailEl.textContent = trail.length ? trail.join("  ›  ") : "Answer a few questions.";
    }
    show("start");
  }

  // ---- k-fold diagram ----
  function initFolds(root) {
    const rows = [...root.querySelectorAll(".fold-row")];
    let i = 0;
    function paint() {
      rows.forEach((r, ri) => r.querySelectorAll(".fold-cell").forEach((c, ci) => {
        c.classList.toggle("val", ci === ri);
        c.textContent = ci === ri ? "check" : "train";
      }));
      rows.forEach((r, ri) => r.style.opacity = ri <= i ? "1" : "0.35");
    }
    paint();
    const btn = root.querySelector("button");
    if (btn) btn.addEventListener("click", () => { i = (i + 1) % rows.length; paint(); });
  }

  document.addEventListener("DOMContentLoaded", () => {
    const btn = document.getElementById("theme-btn");
    if (btn) {
      paintThemeBtn(btn);
      btn.addEventListener("click", () => {
        const next = currentTheme() === "dark" ? "light" : "dark";
        document.documentElement.setAttribute("data-theme", next);
        store.set("theme", next);
        paintThemeBtn(btn);
        window.dispatchEvent(new Event("themechange"));
      });
    }
    if (window.matchMedia) matchMedia("(prefers-color-scheme: dark)").addEventListener?.("change", () => window.dispatchEvent(new Event("themechange")));
    const sel = document.getElementById("rail-select");
    if (sel) sel.addEventListener("change", () => { location.href = sel.value; });
    document.querySelectorAll(".quiz").forEach(initQuiz);
    document.querySelectorAll(".chooser").forEach(initChooser);
    document.querySelectorAll(".folds").forEach(initFolds);
    const mark = document.getElementById("mark-learned");
    if (mark) mark.addEventListener("click", () => { markLearned(mark.dataset.slug); mark.textContent = "Marked as learned"; mark.disabled = true; });
    paintProgress();
  });
})();
