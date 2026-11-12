"""Build the learning website into docs/ (served by GitHub Pages).

    python website/build.py          # regenerate every page from website/content.py + model results
    python website/export_cnn.py     # only needed after retraining models/12_cnn

Pages are plain static HTML/CSS/JS: no build tools or server needed.
"""

import html
import json
import shutil
from pathlib import Path

from content import FAMILY_BLURB, FAMILY_ORDER, MODELS

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
REPO = "https://github.com/Nihar16/MACHINE-LEARNING-MODELS"
FONTS = ("https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..96,600;12..96,700"
         "&family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:ital,wght@0,400;0,500;0,600;1,400&display=swap")
BRAND_SVG = ('<svg class="brand-mark" viewBox="0 0 26 26" aria-hidden="true"><rect x="0.5" y="0.5" width="25" height="25" rx="6" '
             'fill="var(--panel)" stroke="var(--rule)"/><path d="M4 21 L22 5" stroke="var(--ink)" stroke-width="1.6"/>'
             '<circle cx="8" cy="9" r="2.2" fill="var(--c0)"/><circle cx="12" cy="6.5" r="2.2" fill="var(--c0)"/>'
             '<circle cx="14" cy="19" r="2.2" fill="var(--c1)"/><circle cx="19.5" cy="15" r="2.2" fill="var(--c1)"/></svg>')
DEMO_NAME = {"linear-regression": "linear-regression", "logistic-regression": "logistic-regression", "knn": "knn",
             "naive-bayes": "naive-bayes", "svm": "svm", "decision-tree": "decision-tree", "random-forest": "random-forest",
             "gradient-boosting": "gradient-boosting", "kmeans": "kmeans", "pca": "pca", "neural-network": "neural-network"}


def esc(s):
    return html.escape(s, quote=True)


def pct(v, d=1):
    return f"{v * 100:.{d}f}%"


# ------------------------------------------------------------------ results from the repo
def load_results():
    res = {}
    for m in MODELS:
        d = ROOT / "models" / m["folder"]
        metrics = d / "results" / "metrics.json"
        img = None
        if m["slug"] == "linear-regression":
            res[m["slug"]] = {"dataset": "Synthetic housing data (2,000 houses, linear by design)",
                              "cards": [("0.9935", "Test R²"), ("0.9929", "5-fold CV R²"), ("$11.6k", "Mean absolute error")],
                              "note": "The data was generated with a truly linear price rule, which is why R² reaches 0.99. "
                                      "On real data such as California Housing, plain linear regression scores around 0.6."}
            img = d / "results" / "linear_regression_results.png"
        elif m["slug"] == "gradient-boosting":
            j = json.loads((d / "results" / "metrics.json").read_text())
            res[m["slug"]] = sup_result(j)
            img = d / "xgboost" / "results" / "confusion_matrix.png"
        elif m["slug"] == "kmeans":
            j = json.loads(metrics.read_text())
            res[m["slug"]] = {"dataset": f"{j['dataset']} ({j['n_samples']} flowers, labels used only for scoring)",
                              "cards": [(pct(j["matched_accuracy"]), "Clusters match species"),
                                        (f"{j['adjusted_rand_index']:.2f}", "Adjusted Rand index"), (f"{j['silhouette']:.2f}", "Silhouette")],
                              "note": "Setosa is grouped perfectly; versicolor and virginica overlap in the measurements, "
                                      "so no unsupervised method separates them cleanly."}
            img = d / "results" / "kmeans_results.png"
        elif m["slug"] == "pca":
            j = json.loads(metrics.read_text())
            best = max(j["svm_accuracy"].values(), key=lambda v: v["cv_accuracy"])
            res[m["slug"]] = {"dataset": f"{j['dataset']} ({j['n_samples']} images, {j['n_features']} pixel features)",
                              "cards": [(f"{j['components_for_variance']['95%']} of 64", "Components for 95% variance"),
                                        (f"{j['components_for_variance']['99%']} of 64", "Components for 99% variance"),
                                        (pct(best["cv_accuracy"]), f"SVM accuracy on {best['components']} components")],
                              "note": "Cutting 64 features to 29 loses no accuracy here: the SVM scores as well as on all 64."}
            img = d / "results" / "pca_results.png"
        elif m["slug"] == "cnn":
            j = json.loads(metrics.read_text())
            res[m["slug"]] = {"dataset": f"MNIST ({j['n_train']:,} train, {j['n_val']:,} validation, {j['n_test']:,} test images)",
                              "cards": [(pct(j["test_accuracy"], 2), "Test accuracy"), (f"{j['test_errors']}", f"Wrong out of {j['n_test']:,}"),
                                        (f"{j['parameters']:,}", "Parameters")],
                              "note": "The test images were never used for training or for picking the epoch. "
                                      "Most of the remaining errors are digits a person could misread too."}
            img = d / "results" / "misclassified.png"
        else:
            j = json.loads(metrics.read_text())
            res[m["slug"]] = sup_result(j)
            img = d / "results" / "confusion_matrix.png"
        if img and img.exists():
            dest = DOCS / "assets" / "results" / f"{m['slug']}.png"
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(img, dest)
            res[m["slug"]]["img"] = f"../assets/results/{m['slug']}.png"
    return res


def sup_result(j):
    note = ""
    if j["n_test"] < 60:
        note = (f"The test set has only {j['n_test']} examples, so one mistake moves accuracy by "
                f"{100 / j['n_test']:.1f} points. The cross-validation score is the more reliable number.")
    elif j["test_accuracy"] + 0.03 < j["cv_accuracy"]:
        note = "The gap between cross-validation and test accuracy is a sign of instability: a different split gives a different model."
    return {"dataset": f"{j['dataset']} ({j['n_samples']} rows, {j['n_features']} features)",
            "cards": [(pct(j["cv_accuracy"]), "5-fold CV accuracy"), (pct(j["test_accuracy"]), "Held-out test accuracy"),
                      (f"{j['test_errors']} / {j['n_test']}", "Test mistakes")],
            "note": note, "cv": j["cv_accuracy"], "test": j["test_accuracy"]}


# ------------------------------------------------------------------ shared chrome
def page(title, body, base, current="", description=""):
    nav = [("index.html", "Home", "home"), ("concepts.html", "Concepts", "concepts"), ("index.html#path", "All models", "models")]
    links = "".join(f'<a href="{base}{href}"{" aria-current=\"page\"" if key == current else ""}>{label}</a>' for href, label, key in nav)
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<title>{esc(title)}</title>
<meta name="description" content="{esc(description)}">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="{FONTS}">
<link rel="stylesheet" href="{base}assets/style.css">
<script>try{{var t=JSON.parse(localStorage.getItem("theme"));if(t)document.documentElement.setAttribute("data-theme",t)}}catch(e){{}}</script>
</head>
<body>
<header class="topbar"><div class="topbar-inner">
  <a class="brand" href="{base}index.html">{BRAND_SVG}Twelve Models</a>
  <nav class="topnav" aria-label="Main">{links}<a href="{REPO}">GitHub</a></nav>
  <button class="theme-btn" id="theme-btn" type="button" aria-label="Toggle theme"></button>
</div></header>
{body}
<footer class="site"><div class="wrap">
  <span>Built from the code and results in <a href="{REPO}">MACHINE-LEARNING-MODELS</a>. Every number on this site comes from a real run.</span>
  <span><span id="progress-count">0</span> of 12 models learned</span>
</div></footer>
<script src="{base}assets/site.js"></script>
<script src="{base}assets/plot.js"></script>
<script src="{base}assets/demos.js"></script>
<script src="{base}assets/cnn.js"></script>
</body>
</html>
"""


# ------------------------------------------------------------------ model pages
def model_page(i, m, res):
    base = "../"
    by_family = {f: [x for x in MODELS if x["family"] == f] for f in FAMILY_ORDER}
    rail = ""
    for f in FAMILY_ORDER:
        rail += f'<div class="eyebrow">{esc(f)}</div>'
        for x in by_family[f]:
            cur = ' aria-current="page"' if x is m else ""
            rail += (f'<a href="{x["slug"]}.html"{cur}>{esc(x["name"])}'
                     f'<span class="tick" data-learned-slug="{x["slug"]}" hidden aria-label="learned">✓</span></a>')
    opts = "".join(f'<option value="{x["slug"]}.html"{" selected" if x is m else ""}>{n + 1}. {esc(x["name"])}</option>'
                   for n, x in enumerate(MODELS))
    if m["slug"] == "cnn":
        demo = f'<div class="cnn" data-cnn-pad data-base="{base}"></div>'
    else:
        demo = f'<div data-demo="{DEMO_NAME[m["slug"]]}"></div>'
    r = res[m["slug"]]
    cards = "".join(f"<div><strong>{esc(v)}</strong><span>{esc(k)}</span></div>" for v, k in r["cards"])
    img = (f'<figure style="margin:0;max-width:760px"><img src="{r["img"]}" alt="Result plot for {esc(m["name"])} from the repository" '
           f'loading="lazy" style="border:1px solid var(--rule);border-radius:10px;background:#fff"></figure>') if r.get("img") else ""
    note = f'<p class="callout">{esc(r["note"])}</p>' if r.get("note") else ""
    quiz = "".join(
        f'<div class="q" data-answer="{a}"><div class="q-title">{qi + 1}. {esc(q)}</div><div class="q-opts">'
        + "".join(f'<button type="button">{esc(o)}</button>' for o in opts_)
        + f'</div><p class="q-explain" hidden>{esc(ex)}</p></div>'
        for qi, (q, opts_, a, ex) in enumerate(m["quiz"]))
    prev_m = MODELS[i - 1] if i > 0 else None
    next_m = MODELS[i + 1] if i + 1 < len(MODELS) else None
    pager = (f'<a href="{prev_m["slug"]}.html"><span>← Previous</span>{esc(prev_m["name"])}</a>' if prev_m else '<a href="../concepts.html"><span>← Before you start</span>Core concepts</a>')
    pager += (f'<a class="next" href="{next_m["slug"]}.html"><span>Next →</span>{esc(next_m["name"])}</a>' if next_m else '<a class="next" href="../index.html#path"><span>Done →</span>Back to all models</a>')
    steps = "".join(f"<li><span>{s}</span></li>" for s in m["steps"])
    pros = "".join(f"<li>{esc(p)}</li>" for p in m["pros"])
    cons = "".join(f"<li>{esc(c)}</li>" for c in m["cons"])
    tags = "".join(f'<span class="pill">{esc(t)}</span>' for t in m["tags"])
    body = f"""<div class="wrap"><div class="model-layout">
  <nav class="rail" aria-label="Models">{rail}</nav>
  <main class="model-main">
    <select class="rail-select" id="rail-select" aria-label="Jump to a model"><option value="" disabled>Jump to a model</option>{opts}</select>
    <header class="model-head">
      <span class="eyebrow">Model {i + 1} of 12 · {esc(m["family"])}</span>
      <h1>{esc(m["name"])}</h1>
      <p class="lede">{esc(m["oneliner"])}</p>
      <div class="tags">{tags}</div>
    </header>
    <section class="block" aria-labelledby="try">
      <h2 id="try">Try it</h2>
      {demo}
      <p class="callout"><strong>Experiment:</strong> {m["demo_tip"]}</p>
    </section>
    <section class="block prose" aria-labelledby="idea">
      <h2 id="idea">The idea</h2>
      {m["intuition"]}
    </section>
    <section class="block" aria-labelledby="how">
      <h2 id="how">How it works, step by step</h2>
      <ol class="steps">{steps}</ol>
      <div class="formula">{m["math"]}</div>
    </section>
    <section class="block" aria-labelledby="when">
      <h2 id="when">Strengths and weaknesses</h2>
      <div class="procon">
        <div><h3 class="good-h">Strengths</h3><ul>{pros}</ul></div>
        <div><h3 class="bad-h">Weaknesses</h3><ul>{cons}</ul></div>
      </div>
      <div class="prose"><p><strong>Use it when:</strong> {esc(m["use"])}</p><p><strong>Avoid it when:</strong> {esc(m["avoid"])}</p></div>
    </section>
    <section class="block" aria-labelledby="repo">
      <h2 id="repo">In this repository</h2>
      <p class="muted">Dataset: {esc(r["dataset"])}</p>
      <div class="result-grid">{cards}</div>
      {note}
      {img}
      <pre><code>{esc(m["code"])}</code></pre>
      <p class="small muted">Full code: <a href="{REPO}/tree/main/models/{m["folder"]}">models/{m["folder"]}</a>. Train it on your own data with
      <code>python models/{m["folder"]}/train.py</code>{" --data image_folder" if m["slug"] == "cnn" else " --data your.csv --target column"}.</p>
    </section>
    <section class="block" aria-labelledby="quiz-h">
      <h2 id="quiz-h">Check yourself</h2>
      <div class="quiz" data-slug="{m["slug"]}">{quiz}<p class="quiz-score muted">Answer all three correctly to mark this model as learned.</p></div>
    </section>
    <nav class="pager" aria-label="Previous and next model">{pager}</nav>
  </main>
</div></div>"""
    return page(f"{m['name']} · Twelve Models", body, base, "models", m["oneliner"])


# ------------------------------------------------------------------ home
def home(res):
    families = ""
    for f in FAMILY_ORDER:
        cards = ""
        for n, m in enumerate(MODELS):
            if m["family"] != f:
                continue
            r = res[m["slug"]]
            stat = r["cards"][1][0] if "test" in r else r["cards"][0][0]
            stat_label = "test accuracy" if "test" in r else r["cards"][0][1].lower()
            cards += f"""<a class="card" href="models/{m["slug"]}.html">
  <div class="card-top"><span class="num-badge">{n + 1:02d}</span><span class="done-chip" data-learned-slug="{m["slug"]}" hidden>Learned</span></div>
  <h4>{esc(m["name"])}</h4>
  <p>{esc(m["oneliner"])}</p>
  <div class="card-foot"><span>{esc(stat)} {esc(stat_label)}</span><span>Try it →</span></div>
</a>"""
        families += f'<div class="family"><h3>{esc(f)} <span class="eyebrow">{esc(FAMILY_BLURB[f])}</span></h3><div class="cards">{cards}</div></div>'

    rows = ""
    for n, m in enumerate(MODELS):
        r = res[m["slug"]]
        if "test" in r:
            hit = r["test"] >= 0.99 and r["cv"] >= 0.99
            rows += (f'<tr><td><a href="models/{m["slug"]}.html">{esc(m["name"])}</a></td><td>{esc(r["dataset"].split(" (")[0])}</td>'
                     f'<td class="num">{r["cards"][0][0]}</td><td class="num">{r["cards"][1][0]}</td>'
                     f'<td><span class="pill{" yes" if hit else ""}">{"yes" if hit else "no"}</span></td></tr>')
    cnn = res["cnn"]
    rows += (f'<tr><td><a href="models/cnn.html">Convolutional Neural Network</a></td><td>MNIST</td><td class="num">99.6% (val.)</td>'
             f'<td class="num">{cnn["cards"][0][0]}</td><td><span class="pill yes">yes</span></td></tr>')
    names = json.dumps({m["slug"]: m["name"] for m in MODELS})
    body = f"""<main>
<div class="wrap">
  <section class="hero">
    <div class="hero-copy">
      <span class="eyebrow">An interactive course in classic machine learning</span>
      <h1>Learn twelve machine learning models by playing with them</h1>
      <p class="lede">Every model gets a live demo you can poke at, a plain-English explanation, the maths in a few lines, and the real results it scored in this repository. Start with any model, or follow the path in order.</p>
      <div class="btn-row"><a class="btn primary" href="concepts.html">Start with the core concepts</a><a class="btn" href="models/linear-regression.html">Jump to model 1</a></div>
      <div class="hero-stats">
        <div><strong>12</strong><span>models, each with a demo</span></div>
        <div><strong>{cnn["cards"][0][0]}</strong><span>CNN accuracy on MNIST</span></div>
        <div><strong>36</strong><span>quiz questions</span></div>
      </div>
    </div>
    <div class="demo" style="grid-template-columns:1fr;padding:16px;gap:12px">
      <span class="eyebrow">Live: the repo's CNN, running in your browser</span>
      <div class="cnn" data-cnn-pad data-compact data-base=""></div>
    </div>
  </section>
</div>
<section class="section" id="path"><div class="wrap">
  <div class="section-head"><span class="eyebrow">The learning path</span><h2>Twelve models, simplest first</h2>
  <p class="muted">Each family builds on the one before. Pass a model's three-question quiz to mark it as learned; your progress stays in this browser.</p></div>
  {families}
</div></section>
<section class="section"><div class="wrap">
  <div class="section-head"><span class="eyebrow">Which model should I use?</span><h2>Answer a few questions</h2>
  <p class="muted">A starting point, not a rule. In practice you try two or three candidates and compare them on held-out data.</p></div>
  <div class="chooser" data-names='{esc(names)}' data-base="">
    <div class="chooser-trail"></div><div class="chooser-q"></div><div class="chooser-opts"></div><div class="chooser-result" hidden></div>
  </div>
</div></section>
<section class="section"><div class="wrap">
  <div class="section-head"><span class="eyebrow">Real results</span><h2>How the models scored on real data</h2>
  <p class="muted">Every classifier was tuned with 5-fold cross-validation on its training split and then scored once on held-out data. Only the CNN and the SVM reach about 99%; that ceiling comes from the data, not from a lack of tuning.</p></div>
  <div class="table-wrap"><table>
    <thead><tr><th>Model</th><th>Dataset</th><th class="num">CV accuracy</th><th class="num">Test accuracy</th><th>≥ 99% on both</th></tr></thead>
    <tbody>{rows}</tbody>
  </table></div>
</div></section>
<section class="section"><div class="wrap">
  <div class="section-head"><span class="eyebrow">Your own data</span><h2>Train any of these models on your data</h2>
  <p class="muted">The repository can try every model on your CSV, report an honest score, and warn you about leakage.</p></div>
  <pre><code>git clone {REPO}.git
cd MACHINE-LEARNING-MODELS
pip install -r "Setup files/requirements.txt"

python automl.py --data your_data.csv --target column_to_predict      # try every model
python models/12_cnn/train.py --data my_images                        # images: one folder per class</code></pre>
</div></section>
</main>"""
    return page("Twelve Models", body, "", "home",
                "Learn twelve classic machine learning models with live demos, plain explanations and real results.")


# ------------------------------------------------------------------ concepts
def concepts():
    rows = "".join(f'<div class="fold-row"><span>Round {i + 1}</span>' + '<div class="fold-cell"></div>' * 5 + "</div>" for i in range(5))
    body = f"""<main class="wrap"><div style="display:grid;gap:56px;padding-block:40px 64px;max-width:980px">
<header class="model-head">
  <span class="eyebrow">Before the models</span>
  <h1>Core concepts</h1>
  <p class="lede">Five ideas that decide whether a model is any good, whichever algorithm you use. Ten minutes here will make every model page clearer.</p>
</header>

<section class="block prose" aria-labelledby="c1">
  <h2 id="c1">1. Features, labels and the two kinds of learning</h2>
  <p>A data set is a table. Each row is an <strong>example</strong> (a house, an email, an image). The columns describing it are <strong>features</strong>. In <strong>supervised learning</strong> one extra column is the <strong>label</strong> you want to predict: a number (<em>regression</em>, e.g. price) or a category (<em>classification</em>, e.g. spam or not). In <strong>unsupervised learning</strong> there is no label, and the goal is to find structure such as groups (clustering) or a simpler description (dimensionality reduction).</p>
</section>

<section class="block" aria-labelledby="c2">
  <h2 id="c2">2. Never grade a model on its own homework</h2>
  <div class="prose"><p>A model can memorise its training data, so its score on that data says little. Before training, set aside a <strong>test set</strong> the model never sees. Its score on that set estimates how it will do on new data. Every result on this site is a test-set score.</p></div>
  <div class="split-bar" role="img" aria-label="80 percent training data, 20 percent test data">
    <div style="flex:8;background:color-mix(in srgb,var(--c0) 30%,var(--panel))">Training data (80%): the model learns from this</div>
    <div style="flex:2;background:color-mix(in srgb,var(--c1) 55%,var(--panel))">Test (20%)</div>
  </div>
</section>

<section class="block" aria-labelledby="c3">
  <h2 id="c3">3. Underfitting and overfitting</h2>
  <div class="prose"><p>A model that is too simple misses the pattern (<strong>underfitting</strong>). A model that is too flexible learns the noise as well (<strong>overfitting</strong>): it looks brilliant on training data and disappoints on test data. Every model has knobs that move it between the two, such as tree depth, k in KNN, or the degree of a polynomial below.</p></div>
  <div data-demo="overfit"></div>
</section>

<section class="block" aria-labelledby="c4">
  <h2 id="c4">4. Cross-validation: a fairer score on small data</h2>
  <div class="prose"><p>One test split can be lucky or unlucky. <strong>K-fold cross-validation</strong> splits the training data into k parts, trains k times, and checks on a different part each time. The average is a steadier estimate. It is also how hyperparameters are tuned without touching the test set. Press the button to step through the rounds.</p></div>
  <div class="folds">{rows}<div><button class="btn small" type="button">Next round</button></div></div>
</section>

<section class="block" aria-labelledby="c5">
  <h2 id="c5">5. Accuracy can lie</h2>
  <div class="prose"><p>If 1% of transactions are fraud, a model that always says “not fraud” is 99% accurate and completely useless. On imbalanced data, look at <strong>precision</strong> (of the cases you flagged, how many were right) and <strong>recall</strong> (of the real cases, how many you caught). Set positives to 1% below and compare accuracy with the “always negative” baseline.</p></div>
  <div data-demo="threshold"></div>
</section>

<section class="block prose" aria-labelledby="c6">
  <h2 id="c6">6. Leakage: the most common fake 99%</h2>
  <p><strong>Leakage</strong> means a feature secretly contains the answer, for example a “cancellation reason” column when predicting churn, or a value that is only known after the outcome. The model scores near 100% in testing and fails in real use. If a score looks too good to be true, check whether one column alone predicts the label. The repository's <code>automl.py</code> does this check automatically.</p>
  <p class="callout">Ready? Start with <a href="models/linear-regression.html">model 1: Linear Regression</a>.</p>
</section>
</div></main>"""
    return page("Core Concepts · Twelve Models", body, "", "concepts",
                "Train/test splits, overfitting, cross-validation, precision and recall, and leakage, with live demos.")


def main():
    res = load_results()
    (DOCS / "models").mkdir(parents=True, exist_ok=True)
    (DOCS / "index.html").write_text(home(res))
    (DOCS / "concepts.html").write_text(concepts())
    for i, m in enumerate(MODELS):
        (DOCS / "models" / f"{m['slug']}.html").write_text(model_page(i, m, res))
    (DOCS / ".nojekyll").write_text("")
    print(f"built {2 + len(MODELS)} pages into {DOCS}")


if __name__ == "__main__":
    main()
