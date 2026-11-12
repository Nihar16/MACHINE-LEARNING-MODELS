# Learning website

The interactive site in `docs/` ("Twelve Models"): one page per model with a live demo, a plain-English
explanation, the maths, real results from this repository, code and a short quiz, plus a core-concepts page.
The CNN page runs the repo's trained MNIST network in the browser.

It is plain HTML, CSS and JavaScript, with no build tools, so GitHub Pages can serve it directly.

## Publish it with GitHub Pages

1. On GitHub, open **Settings → Pages**.
2. Under **Build and deployment**, choose **Deploy from a branch**, branch **main**, folder **/docs**, and save.
3. After a minute the site is live at `https://<your-user>.github.io/MACHINE-LEARNING-MODELS/`.

## Edit it

| What | Where |
|---|---|
| Page text, steps, pros and cons, quizzes | `website/content.py` |
| Page layout and the results section | `website/build.py` |
| Interactive demos | `docs/assets/demos.js` (shared plotting code in `docs/assets/plot.js`) |
| Look and feel | `docs/assets/style.css` |
| In-browser CNN | `docs/assets/cnn.js` |

After editing `content.py` or `build.py`, regenerate the pages (this also copies the latest result plots and
numbers from `models/*/results/`):

```bash
python website/build.py
```

After retraining the CNN (`models/12_cnn/model.py`), export its weights for the browser:

```bash
python website/export_cnn.py
```

Preview locally:

```bash
python -m http.server 8000 --directory docs
```

Then open http://localhost:8000.
