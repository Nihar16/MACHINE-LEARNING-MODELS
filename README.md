![MACHINE-LEARNING-MODELS](https://socialify.git.ci/Nihar16/MACHINE-LEARNING-MODELS/image?custom_description=This+repository+is+a+hands-on+Python+collection+of+classic+machine+learning+models+with+clean+code%2C+visual+intuition%2C+and+reproducible+workflows.&description=1&font=Source+Code+Pro&name=1&owner=1&pattern=Circuit+Board&theme=Dark)

A hands-on collection of classic machine learning models implemented, explained, and benchmarked in Python. Each model has its own folder with a notebook, clean code, results, and a write-up of when to use it and when not to.

> **Goal:** build a practical, intuition-first understanding of the most important ML algorithms, from simple linear models to neural networks, and be able to choose the right one for a given problem.

---

## Table of Contents

1. [Project Goals](#project-goals)
2. [ML Landscape](#ml-landscape)
2. [Models Covered](#models-covered)
3. [Model Comparison at a Glance](#model-comparison-at-a-glance)
   - [Visual Intuition](#visual-intuition)
   - [Which Model Should I Use?](#which-model-should-i-use)
4. [Repository Structure](#repository-structure)
5. [Getting Started](#getting-started)
6. [Workflow Used for Every Model](#workflow-used-for-every-model)
7. [Evaluation Metrics](#evaluation-metrics)
8. [Use It on Your Own Data](#use-it-on-your-own-data)
8. [Results](#results)
8. [Datasets](#datasets)
9. [Roadmap](#roadmap)
10. [Contributing](#contributing)
11. [License](#license)

---

## Project Goals

- Implement each model with **scikit-learn** and, where useful, **from scratch with NumPy** to understand the math.
- Document **how each model works**, its **assumptions**, **pros**, and **cons**.
- Compare models on the same datasets using consistent metrics.
- Keep everything reproducible (fixed seeds, pinned requirements).

---

## ML Landscape

```mermaid
flowchart LR
    ML["Machine Learning"] --> SUP["Supervised"]
    ML --> UNS["Unsupervised"]
    ML --> DL["Deep Learning"]
    SUP --> REG["Regression"]
    SUP --> CLS["Classification"]
    SUP --> ENS["Tree and Ensemble"]
    REG --> LR["Linear Regression"]
    CLS --> LOG["Logistic Regression"]
    CLS --> KNN["KNN"]
    CLS --> NB["Naive Bayes"]
    CLS --> SVM["SVM"]
    ENS --> DT["Decision Tree"]
    ENS --> RF["Random Forest"]
    ENS --> GB["Gradient Boosting"]
    UNS --> KM["K-Means"]
    UNS --> PCA["PCA"]
    DL --> MLP["Neural Network"]
    DL --> CNN["CNN"]
```

---

## Models Covered

### Supervised Learning: Regression

#### 1. Linear Regression
**What it is:** Fits a straight line (or hyperplane) that minimizes the squared error between predictions and actual values: `y = w·x + b`.

**How it works:** Finds weights using Ordinary Least Squares (closed form) or Gradient Descent.

**Typical use cases:** House price prediction, sales forecasting, trend analysis.

| Pros | Cons |
|------|------|
| Simple, fast, highly interpretable | Assumes a linear relationship |
| Works well on small datasets | Sensitive to outliers |
| Coefficients show feature impact | Suffers from multicollinearity |
| Great baseline model | Poor with complex, non-linear patterns |

**Variants planned:** Ridge (L2), Lasso (L1), Polynomial Regression.

---

### Supervised Learning: Classification

#### 2. Logistic Regression
**What it is:** A linear model for classification that outputs a probability using the sigmoid function: `P(y=1) = 1 / (1 + e^-(w·x + b))`.

**How it works:** Optimizes log-loss (cross-entropy) via gradient descent.

**Typical use cases:** Spam detection, churn prediction, medical diagnosis, fraud flags.

| Pros | Cons |
|------|------|
| Outputs calibrated probabilities | Linear decision boundary only |
| Fast, easy to train and interpret | Struggles with complex feature interactions |
| Regularization prevents overfitting | Needs feature scaling for best results |
| Strong baseline for binary problems | Sensitive to multicollinearity |

#### 3. K-Nearest Neighbors (KNN)
**What it is:** Classifies a point by majority vote of its *k* closest training points (or averages them for regression).

**How it works:** No training phase; stores the data and computes distances (Euclidean, Manhattan) at prediction time.

**Typical use cases:** Recommendation systems, simple pattern recognition, anomaly detection.

| Pros | Cons |
|------|------|
| Very simple, no training step | Slow predictions on large datasets |
| Non-linear decision boundaries | Needs feature scaling |
| Few assumptions about data | Hurt by high dimensionality (curse of dimensionality) |
| Works for classification and regression | Sensitive to noise and irrelevant features |

#### 4. Naive Bayes
**What it is:** A probabilistic classifier based on Bayes' theorem with the "naive" assumption that features are independent given the class.

**How it works:** Computes `P(class | features) ∝ P(class) · ∏ P(feature | class)`.

**Typical use cases:** Spam filtering, text classification, sentiment analysis.

| Pros | Cons |
|------|------|
| Extremely fast, scales to huge data | Independence assumption rarely true |
| Works well with small training sets | Poor probability estimates |
| Excellent for text/high-dimensional data | Struggles with correlated features |
| Handles multiple classes naturally | Zero-frequency issue (needs smoothing) |

#### 5. Support Vector Machine (SVM)
**What it is:** Finds the hyperplane that separates classes with the **maximum margin**. Kernels (RBF, polynomial) allow non-linear boundaries.

**How it works:** Uses support vectors (the points closest to the boundary) to define the decision surface; the kernel trick maps data into higher dimensions.

**Typical use cases:** Image classification, bioinformatics, text categorization, small/medium datasets.

| Pros | Cons |
|------|------|
| Effective in high-dimensional spaces | Slow on very large datasets |
| Memory efficient (uses support vectors) | Needs careful tuning (C, gamma, kernel) |
| Flexible via kernels | Requires feature scaling |
| Robust with clear margin of separation | No direct probability outputs |

---

### Tree-Based and Ensemble Models

#### 6. Decision Tree
**What it is:** A flowchart-like model that splits data on feature thresholds to make predictions.

**How it works:** Recursively chooses splits that maximize information gain or minimize Gini impurity.

**Typical use cases:** Credit approval, rule-based decisions, explainable predictions.

| Pros | Cons |
|------|------|
| Easy to visualize and explain | Prone to overfitting |
| No feature scaling needed | Unstable (small data changes alter the tree) |
| Handles numeric and categorical data | Biased toward dominant classes |
| Captures non-linear relationships | Axis-aligned splits only |

#### 7. Random Forest
**What it is:** An ensemble of many decision trees trained on random subsets of data and features; predictions are averaged or voted.

**How it works:** Bagging (bootstrap aggregation) plus random feature selection reduces variance.

**Typical use cases:** Tabular data, feature importance analysis, intrusion/fraud detection.

| Pros | Cons |
|------|------|
| High accuracy, robust to overfitting | Less interpretable than a single tree |
| Handles missing values and outliers well | Larger memory and slower inference |
| Provides feature importance | Can underperform on very sparse data |
| Minimal tuning needed | Not great at extrapolation |

#### 8. Gradient Boosting (XGBoost / LightGBM)
**What it is:** Builds trees sequentially, where each new tree corrects the errors of the previous ones.

**How it works:** Fits trees to the negative gradient of the loss function, with a learning rate controlling each step.

**Typical use cases:** Kaggle-style tabular competitions, ranking, risk scoring.

| Pros | Cons |
|------|------|
| Often state-of-the-art on tabular data | Many hyperparameters to tune |
| Handles mixed feature types and missing values | Can overfit if not regularized |
| Built-in regularization (XGBoost) | Sequential training is slower than RF |
| Flexible loss functions | Harder to interpret |

---

### Unsupervised Learning

#### 9. K-Means Clustering
**What it is:** Partitions data into *k* clusters by assigning points to the nearest centroid and updating centroids iteratively.

**How it works:** Minimizes within-cluster sum of squares (inertia). Choose *k* with the elbow method or silhouette score.

**Typical use cases:** Customer segmentation, image compression, grouping similar behaviors.

| Pros | Cons |
|------|------|
| Simple, fast, scalable | Must specify *k* in advance |
| Easy to interpret clusters | Assumes spherical, similarly sized clusters |
| Works well on clean numeric data | Sensitive to initialization and outliers |
| Good starting point for exploration | Needs feature scaling |

#### 10. Principal Component Analysis (PCA)
**What it is:** A dimensionality reduction technique that projects data onto the directions of maximum variance.

**How it works:** Computes eigenvectors of the covariance matrix (or uses SVD) and keeps the top components.

**Typical use cases:** Visualization, noise reduction, speeding up other models, removing correlated features.

| Pros | Cons |
|------|------|
| Reduces dimensionality and noise | Components are hard to interpret |
| Removes feature correlation | Only captures linear structure |
| Speeds up downstream training | Requires scaling; sensitive to outliers |
| Helps visualize high-dimensional data | Some information is always lost |

---

### Deep Learning

#### 11. Neural Network (Multi-Layer Perceptron)
**What it is:** Layers of connected neurons that learn non-linear transformations of input data.

**How it works:** Forward pass computes predictions; backpropagation with gradient descent updates weights. Built with TensorFlow/Keras or PyTorch.

**Typical use cases:** Complex tabular problems, pattern recognition, a foundation for CNNs/RNNs.

| Pros | Cons |
|------|------|
| Learns highly complex non-linear patterns | Needs lots of data |
| Flexible architectures | Computationally expensive |
| Scales with data and compute | Black box, hard to interpret |
| Basis for modern deep learning | Many hyperparameters; easy to overfit |

#### 12. Convolutional Neural Network (CNN)
**What it is:** A neural network built for images. Convolution layers slide small filters over the image to detect edges and strokes, then combine them into shapes and objects.

**How it works:** Stacks of convolution, batch normalization, pooling and dropout layers, then a small dense head. Data augmentation (random shifts, rotations and zooms) teaches it that a moved or tilted object is still the same object. Built with PyTorch.

**Typical use cases:** Image classification, medical imaging, defect detection, handwriting and document recognition.

| Pros | Cons |
|------|------|
| State of the art for images | Needs more data and compute than classic models |
| Learns features automatically | Hard to interpret |
| Robust to small shifts and distortions | Many design choices to tune |
| Transfer learning from pretrained networks works well on small datasets | Training is slow without a GPU |

---

## Model Comparison at a Glance

| Model | Type | Interpretability | Training Speed | Needs Scaling | Handles Non-linearity |
|-------|------|:---:|:---:|:---:|:---:|
| Linear Regression | Regression | High | Very fast | Helpful | No |
| Logistic Regression | Classification | High | Very fast | Yes | No |
| KNN | Both | Medium | None (slow predict) | Yes | Yes |
| Naive Bayes | Classification | Medium | Very fast | No | Limited |
| SVM | Both | Low | Slow on large data | Yes | Yes (kernels) |
| Decision Tree | Both | High | Fast | No | Yes |
| Random Forest | Both | Medium | Medium | No | Yes |
| Gradient Boosting | Both | Low | Medium | No | Yes |
| K-Means | Clustering | Medium | Fast | Yes | No |
| PCA | Dim. reduction | Low | Fast | Yes | No |
| Neural Network | Both | Very low | Slow | Yes | Yes |
| CNN | Images | Very low | Slow (fast on GPU) | Yes (normalize) | Yes |

---

## Visual Intuition

### Bagging vs Boosting (Random Forest vs Gradient Boosting)

```mermaid
flowchart TB
    subgraph Bagging["Random Forest: trees trained in parallel, then vote"]
        D1["Training data"] --> S1["Sample 1"] --> T1["Tree 1"]
        D1 --> S2["Sample 2"] --> T2["Tree 2"]
        D1 --> S3["Sample 3"] --> T3["Tree 3"]
        T1 --> V["Average or majority vote"]
        T2 --> V
        T3 --> V
    end
    subgraph Boosting["Gradient Boosting: trees trained in sequence"]
        D2["Training data"] --> B1["Tree 1"]
        B1 -->|"errors"| B2["Tree 2"]
        B2 -->|"remaining errors"| B3["Tree 3"]
        B3 --> SUM["Weighted sum of all trees"]
    end
```

### K-Means Loop

```mermaid
flowchart LR
    A["Pick k random centroids"] --> B["Assign each point to nearest centroid"]
    B --> C["Move centroids to cluster means"]
    C --> D{"Centroids changed?"}
    D -->|"Yes"| B
    D -->|"No"| E["Done: final clusters"]
```

### Neural Network (MLP)

```mermaid
flowchart LR
    subgraph In["Input layer"]
        x1(("x1"))
        x2(("x2"))
        x3(("x3"))
    end
    subgraph H1["Hidden layer 1"]
        h1(("h1"))
        h2(("h2"))
        h3(("h3"))
        h4(("h4"))
    end
    subgraph H2["Hidden layer 2"]
        g1(("g1"))
        g2(("g2"))
        g3(("g3"))
    end
    subgraph Out["Output layer"]
        y(("y"))
    end
    x1 & x2 & x3 --> h1 & h2 & h3 & h4
    h1 & h2 & h3 & h4 --> g1 & g2 & g3
    g1 & g2 & g3 --> y
```

Training loop: forward pass, compute loss, backpropagate gradients, update weights, repeat.

### Decision Tree Example

```mermaid
flowchart TD
    A{"Income above 50k?"} -->|"Yes"| B{"Credit score above 700?"}
    A -->|"No"| C{"Existing debt above 20k?"}
    B -->|"Yes"| D["Approve"]
    B -->|"No"| E["Review"]
    C -->|"Yes"| F["Reject"]
    C -->|"No"| E
```

---

## Which Model Should I Use?

```mermaid
flowchart TD
    Start["What is your goal?"] --> Q1{"Do you have labels?"}
    Q1 -->|"No"| Q2{"Group data or reduce features?"}
    Q2 -->|"Group"| KM["K-Means"]
    Q2 -->|"Reduce"| PCA["PCA"]
    Q1 -->|"Yes"| Q3{"Predict a number or a category?"}
    Q3 -->|"Number"| R1{"Linear relationship?"}
    R1 -->|"Yes"| LR["Linear or Ridge or Lasso"]
    R1 -->|"No"| R2["Random Forest or Gradient Boosting"]
    Q3 -->|"Category"| C1{"Need interpretability?"}
    C1 -->|"Yes"| C2["Logistic Regression or Decision Tree"]
    C1 -->|"No"| C3{"Tabular data?"}
    C3 -->|"Yes"| C4["Random Forest or Gradient Boosting"]
    C3 -->|"Text"| C5["Naive Bayes or Logistic Regression"]
    C3 -->|"Images"| C6["CNN"]
```

---

## Repository Structure

```
MACHINE-LEARNING-MODELS/
├── data/                      # Datasets (small ones) or download scripts
├── models/
│   ├── 01_linear_regression/
│   │   ├── notebook.ipynb
│   │   ├── model.py
│   │   └── README.md          # Model-specific notes and results
│   ├── 02_logistic_regression/
│   ├── 03_knn/
│   ├── ...
├── utils/                     # Shared preprocessing and plotting helpers
├── results/                   # Saved metrics, plots, comparison tables
├── requirements.txt
├── LICENSE
└── README.md
```

---

## Getting Started

```bash
git clone https://github.com/Nihar16/MACHINE-LEARNING-MODELS.git
cd MACHINE-LEARNING-MODELS
```

Then run the setup script for your OS. It checks Python, creates a virtual environment, installs dependencies, and registers a Jupyter kernel.

| OS | Command |
|----|---------|
| macOS / Linux | `chmod +x setup.sh && ./setup.sh` |
| Windows (PowerShell) | `.\setup.ps1` |
| Windows (double-click) | `setup.bat` |

Add `--launch` (macOS/Linux) or `-Launch` (PowerShell) to open Jupyter when setup finishes.

<details>
<summary>Manual setup</summary>

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
jupyter notebook
```

</details>

**Core libraries:** NumPy, pandas, scikit-learn, matplotlib, seaborn, XGBoost, TensorFlow/PyTorch, Jupyter.

---

## Workflow Used for Every Model

```mermaid
flowchart LR
    A["Understand problem"] --> B["EDA"] --> C["Preprocess"] --> D["Baseline"]
    D --> E["Train with CV"] --> F["Tune hyperparameters"] --> G["Evaluate"] --> H["Write-up"]
    G -.->|"needs improvement"| C
```

1. **Problem and data understanding:** goal, features, target.
2. **Exploratory data analysis:** distributions, correlations, missing values.
3. **Preprocessing:** cleaning, encoding, scaling, train/test split.
4. **Baseline model:** a simple reference to beat.
5. **Training:** with cross-validation.
6. **Hyperparameter tuning:** GridSearchCV / RandomizedSearchCV.
7. **Evaluation:** metrics, confusion matrix, error analysis.
8. **Write-up:** pros, cons, lessons learned.

---

## Evaluation Metrics

| Task | Metrics |
|------|---------|
| Regression | MAE, MSE, RMSE, R² |
| Classification | Accuracy, Precision, Recall, F1, ROC-AUC, Confusion Matrix |
| Clustering | Silhouette Score, Inertia, Davies-Bouldin Index |
| Dimensionality Reduction | Explained Variance Ratio |

> Accuracy alone is misleading on imbalanced data; precision, recall, and F1 are prioritized there.

---

## Use It on Your Own Data

Every model can be trained on any table you have: CSV, Excel, Parquet or TSV, with numbers, text categories, true/false values, dates and missing values. You only name the column to predict.

```bash
# Try every model and keep the best one
python automl.py --data your_data.csv --target column_to_predict

# Or train one specific model
python models/07_random_forest/train.py --data your_data.csv --target column_to_predict

# Predict on new rows with the saved model
python predict.py --model ml_output/best/model.joblib --data new_rows.csv --out predictions.csv
```

| Model | Command | Tasks |
|---|---|---|
| Linear Regression (OLS / Ridge / Lasso) | `python models/01_linear_regression/train.py` | regression |
| Logistic Regression | `python models/02_logistic_regression/train.py` | classification |
| K-Nearest Neighbors | `python models/03_knn/train.py` | classification, regression |
| Gaussian Naive Bayes | `python models/04_naive_bayes/train.py` | classification |
| Support Vector Machine | `python models/05_svm/train.py` | classification, regression |
| Decision Tree | `python models/06_decision_tree/train.py` | classification, regression |
| Random Forest | `python models/07_random_forest/train.py` | classification, regression |
| Gradient Boosting (XGBoost / LightGBM) | `python models/08_gradient_boosting/train.py` | classification, regression |
| K-Means | `python models/09_kmeans/train.py` | clustering (no target needed) |
| PCA + linear model | `python models/10_pca/train.py` | classification, regression |
| Neural Network (MLP) | `python models/11_neural_network/train.py` | classification, regression |
| CNN | `python models/12_cnn/train.py --data image_folder` | image classification (one sub-folder per class) |

**What happens automatically**

- **Task detection:** a text or small-integer target means classification, otherwise regression. Override it with `--task`.
- **Cleaning:** ID columns, constant columns and free-text columns are dropped. Dates become year, month, day and weekday. Missing values are filled in, categories are one-hot encoded, and numbers are scaled. All of this is learned on training rows only.
- **Tuning:** randomized hyperparameter search with cross-validation on the training split (`--n-iter`, `--folds`).
- **Honest scoring:** the final score comes from a 20% held-out test split. The winning model is chosen by its cross-validation score, never by its test score.
- **Safety checks:** each run compares against a do-nothing baseline and warns about class imbalance and duplicated rows. It also flags **leakage**, meaning a single column that predicts the target almost perfectly, which is the most common reason for a fake 99%.

**About 99% accuracy:** each run reports whether `--target-score` (default 0.99) was reached. Whether 99% is possible depends on how much your features say about the target, not on the model. Clean, well-separated problems reach it, but noisy real-world ones (churn, prices, medical outcomes) often top out far lower, and no amount of tuning changes that. If a run reports 99% together with a leakage warning, drop that column with `--drop column_name` and rerun.

From Python or Colab:

```python
import pandas as pd
from utils.generalized import run_automl, train_model

df = pd.read_csv("your_data.csv")
report = run_automl(df, target="column_to_predict", out_dir="ml_output/best")
model = report["model_object"]  # a fitted scikit-learn pipeline: model.predict(new_df)
```

Run the tests with `python -m pytest tests -q`.

---

## Results

Every model is tuned with 5-fold cross-validation on the training set and then scored once on a held-out 20% test set. Full table: [`results/comparison.md`](results/comparison.md).

### Classification

| Model | Dataset | CV accuracy | Test accuracy | ≥99% on both |
|---|---|---|---|---|
| [Logistic Regression](models/02_logistic_regression/) | Breast Cancer Wisconsin | 98.2% ± 0.5% | 98.2% (2/114 wrong) | no |
| [KNN](models/03_knn/) | Digits | 98.7% ± 0.8% | 98.3% (6/360 wrong) | no |
| [Gaussian Naive Bayes](models/04_naive_bayes/) | Wine | 97.2% ± 2.6% | 100.0% (0/36 wrong) | no |
| [SVM](models/05_svm/) | Digits | 99.1% ± 0.3% | 98.6% (5/360 wrong) | no |
| [Decision Tree](models/06_decision_tree/) | Breast Cancer Wisconsin | 93.8% ± 1.9% | 87.7% (14/114 wrong) | no |
| [Random Forest](models/07_random_forest/) | Wine | 97.9% ± 2.8% | 100.0% (0/36 wrong) | no |
| [XGBoost](models/08_gradient_boosting/) | Breast Cancer Wisconsin | 97.8% ± 1.6% | 95.6% (5/114 wrong) | no |
| [Neural Network (MLP)](models/11_neural_network/) | Digits | 98.1% ± 0.9% | 98.1% (7/360 wrong) | no |
| [**CNN**](models/12_cnn/) | **MNIST** (70,000 images) | 99.6% (validation) | **99.67%** (33/10,000 wrong) | **yes** |

### Regression

| Model | Dataset | Result |
|---|---|---|
| [Linear Regression](models/01_linear_regression/) | Synthetic linear housing data | test R² 0.9935, CV R² 0.9929 |

### Unsupervised

| Model | Dataset | Result |
|---|---|---|
| [K-Means](models/09_kmeans/) | Iris | matched accuracy 89.3%, ARI 0.73, silhouette 0.55 |
| [PCA](models/10_pca/) | Digits | 41 of 64 components keep 99% of variance; SVM on PCA 29 comp (95% var): 99.2% CV |

> **About 99% accuracy:** the **CNN clears it on real data, with 99.67% on the official 10,000-image MNIST test set.** Among the classic models, only the SVM (and an SVM on PCA-reduced features) reaches about 99%, and only in cross-validation. The other classifiers score 93.8% (Decision Tree) to 98.7% (KNN) in CV. A 100% test score on Wine comes from 36 samples, so the CV score is the honest estimate there. Linear Regression reaches R² 0.99 because its dataset is synthetic and linear by design.

Re-run everything:

```bash
for m in models/*/model.py; do python "$m"; done
python utils/compare.py
```

---

## Datasets

| Dataset | Used For |
|---------|----------|
| California Housing | Regression models |
| Iris | Classification basics, KNN, SVM, clustering |
| Breast Cancer Wisconsin | Binary classification, model comparison |
| Titanic | Tree-based models, feature engineering |
| MNIST | Neural networks, PCA visualization |
| Mall Customers | K-Means segmentation |
| NSL-KDD / CICIDS2017 *(optional)* | Applied security: intrusion/anomaly detection |

---

## Roadmap

- [x] Linear and Logistic Regression
- [x] KNN and Naive Bayes
- [x] Decision Tree and Random Forest
- [x] SVM
- [x] Gradient Boosting (XGBoost / LightGBM)
- [x] K-Means and PCA
- [x] Neural Network (MLP)
- [x] CNN for image classification
- [x] Final model comparison report
- [ ] Applied security project (network intrusion detection)

---

## Contributing

Suggestions and improvements are welcome. Open an issue to discuss a change, or submit a pull request.

---

## License

Distributed under the **MIT License**. See [`LICENSE`](LICENSE) for details.

---

**Author:** [Nihar Mane](https://github.com/Nihar16)
