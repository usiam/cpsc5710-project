# Explaining Diagnostic Uncertainty: Integrating Conformal Prediction with Visual XAI

**Course:** CPSC 5710: Trustworthy Artificial Intelligence  
**Authors:** Joshua Lu, Tahamid Siam, Grant Shanklin  
**Institution:** Yale University  
**Date:** December 10, 2025

## 📋 Project Overview

This project proposes a system to make deep learning models for medical diagnosis more trustworthy by explaining their uncertainty. We combine **Conformal Prediction (CP)**, a framework that provides statistically guaranteed prediction sets, with **Explainable AI (XAI)** to visually diagnose the source of a model's uncertainty.

Using the **CUB-200-2011** dataset as a high-fidelity proxy for fine-grained medical imaging (such as dermatoscopic skin lesion images), we aim to distinguish between:

- **Aleatoric Uncertainty:** Inherent feature ambiguity (e.g., visual mimicry between species)
- **Epistemic Uncertainty:** Model ignorance due to rare or poor-quality data (e.g., background noise).

Our system allows clinicians to inspect visual evidence supporting each potential outcome in a prediction set, making the model's reasoning transparent.

## 📂 Repository Structure

```text
├── data/                  # Dataset storage (CUB-200-2011)
├── notebooks/
│   ├── model_finetune.ipynb   # Fine-tuning ResNet-50 backbone
│   └── trustworthyML.ipynb    # Conformal calibration, SER calculation, and XAI viz
├── results/               # Saved figures and evaluation metrics
├── scripts/               # Utility scripts for data processing
├── src/                   # Source code for custom metrics (SER, etc.)
├── environment.yml        # Conda environment configuration
└── README.md              # Project documentation
```

🛠️ Installation & Setup

- Clone the repository:

```
git clone [git@github.com:usiam/cpsc5710-project.git](git@github.com:usiam/cpsc5710-project.git)
cd cpsc5710-project
```

- Create the environment: This project uses conda for dependency management.

```
conda env create -f environment.yml
conda activate project
```

- Data Preparation: Download the CUB-200-2011 dataset and place it in the data/ directory. The codebase expects the standard directory structure including images/ and attributes/.

## 🚀 Usage Pipeline

The workflow is divided into two primary stages corresponding to the provided Jupyter notebooks:

### 1. Model Training (`notebooks/model_finetune.ipynb`)

This notebook handles the fine-tuning of the classifier:

- **Backbone:** ResNet-50 pre-trained on ImageNet.
- **Modifications:** The final layer is replaced with a $2048 \rightarrow 200$ classifier head.
- **Training:** Full fine-tuning using AdamW optimizer with decay and a gradually reduced learning rate.

### 2. Trustworthy Analysis (`notebooks/trustworthyML.ipynb`)

This notebook implements the core contributions of the paper:

- **Split Conformal Calibration:** Uses `TorchCP` to compute non-conformity scores on a held-out calibration set and generate prediction sets $\Gamma_{\alpha}(x)$ with a 95% target coverage guarantee.
- **Visual Explanation:** Generates multi-resolution visual explanations for every diagnosis in the prediction set using **Grad-CAM** (region focus) and **Guided Backpropagation** (fine features).
- **Saliency Energy Ratio (SER):** Calculates the proportion of model attention falling within the ground-truth bounding box to distinguish between trusted predictions and attentional failure.

## 📊 Methodology & Metrics

### Saliency Energy Ratio (SER)

To verify where the model attends, we introduced the SER metric. For a normalized attribution map $M$ and ground-truth bounding box $B$:
$$SER(M, B) = \sum_{i,j} M_{ij} \cdot B_{ij}$$
High SER values validate that uncertainty arises from relevant features (aleatoric), whereas low SER flags reliance on background noise (epistemic failure).

### Conformal Prediction

We apply split conformal prediction to ensure that for any new input $x$, the prediction set $\Gamma_{\alpha}(x)$ contains the true label with probability $1-\alpha$ (set to 95%).

## 📈 Key Results

- **Robustness:** The Conformal Predictor successfully adapted to noisy, uncropped data by increasing the average set size from 2.19 to 5.06, whereas a Naive Top-K baseline failed.
- **Diagnostic Insight:** We found a strong correlation between high uncertainty (large set sizes) and low SER ($<0.35$). This indicates that extreme uncertainty is often driven by "attentional failure," where the model attends to background scenery rather than the subject.

## 📚 References

- [1] Wah et al. Caltech-UCSD Birds-200-2011 (2011).
- [2] Angelopoulos & Bates. A Gentle Introduction to Conformal Prediction (2022).
- [3] Selvaraju et al. Grad-CAM: Visual Explanations... (2019).

---

_For questions regarding reproducibility, please contact the authors._
