# CAMS-Net: Contrast-Aware Multi-Scale Network with Morphology-Guided Training for Early Oral Malignancy Triage

<p align="center">
  <img width="220" height="220" alt="CAMS-Net Logo" src="https://github.com/user-attachments/assets/84d5afb1-b07c-4237-a0d0-b1d0b1f2132e" />
</p>

<p align="center">
  <a href="https://www.python.org/"><img src="https://img.shields.io/badge/Python-3.9%2B-blue.svg" alt="Python 3.9+"></a>
  <a href="https://pytorch.org/"><img src="https://img.shields.io/badge/PyTorch-2.0-ee4c2c.svg" alt="PyTorch 2.0"></a>
  <a href="https://doi.org/10.17632/6v83j8y3xj.1"><img src="https://img.shields.io/badge/Dataset-Mendeley%20Oral%20Images%20(N%3D323)-green.svg" alt="Dataset"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License: MIT"></a>
</p>

Official implementation, evaluation pipeline, Grad-CAM interpretability tools, and publication assets for **CAMS-Net**, a transfer-learning framework engineered for low-resource, smartphone-based binary triage (**Benign** vs. **Malignant**) of oral potentially malignant disorders (OPMDs) and oral squamous cell carcinoma (OSCC).

---

## Key Contributions

1. **Dual-Path Preprocessing & Contrast-Aware Gate (CAG):**
   - Processes both **LAB-CLAHE enhanced** ($L^*$ luminance equalization, $\rho_{\text{clip}}=3.0$, $8\times 8$ grid) and **raw** RGB inputs through independent convolutional stems.
   - - Computes a learned channel-wise gate $\mathbf{g} \in [0,1]^{64}$ from the per-channel absolute difference $|\mathbf{F}_{\text{enh}} - \mathbf{F}_{\text{raw}}|$ to adaptively weight contrast-enhanced features against raw mucosal features, preventing noise amplification on well-lit captures.
2. **Selective Layer Recalibration (SLR):**
   - Lightweight Squeeze-and-Excitation recalibration modules inserted after ResNet50 Stages 2, 3, and 4 to suppress over-enhanced channels prior to spatial downsampling.
3. **Morphology-Aware Auxiliary Head (MAAH):**
   - Training-time multi-task regression branch ($\lambda_{\text{aux}} = 0.1$) supervising Layer4 representations against four quantitative tissue texture descriptors: **LBP uniformity**, **Gabor energy**, **Sobel edge magnitude**, and **2D FFT high-frequency energy**. Completely detached at inference for **zero latency overhead**.
4. **Triage-Oriented Operating-Point Selection:**
   - Sensitivity-constrained threshold optimization ($\tau^* = 0.40$ for $S_{\min} = 0.90$) achieving **91.05% sensitivity** and **78.64% specificity** without model retraining.

---

## Qualitative Grad-CAM Lesion Localization ($4 \times 5$ Cohort Grid)

Comparison across **20 representative patient cases** (balanced across Benign and Malignant classes from the Mendeley Oral Images Dataset) illustrating CAMS-Net's spatial localization and contour delineation ($\text{CAM} \ge 0.55$):

| Raw Clinical Inputs ($2000 \times 1600$ px) | CAMS-Net Triage Predictions & Grad-CAM Contours ($2000 \times 1600$ px) |
| :---: | :---: |
| <img width="100%" alt="Raw Oral Cavity Input Grid (4x5)" src="https://github.com/user-attachments/assets/d15980bc-3b61-4b4b-8ee0-b731b45ee5f5" /> | <img width="100%" alt="Predicted Triage & Grad-CAM Grid (4x5)" src="https://github.com/user-attachments/assets/febddf05-5dbf-4683-a139-37ac3a7d6bd2" /> |
| *Spatially aligned 224 &times; 224 center-cropped raw RGB photographs showing intraoral lighting and mucosal variation.* | *Binary triage headers (`BENIGN` / `MALIGNANT`), Layer4 Grad-CAM overlays, and red–yellow contours delineating predicted malignant lesions.* |
---

## Performance Summary (Patient-Level 5-Fold Cross-Validation, $N=323$)

Evaluated on **323 biopsy-verified images from 323 unique patients** (strictly patient-disjoint splits, no pre-augmented duplicates) from the Mendeley Oral Images Dataset:

| Metric | Default Threshold ($\tau = 0.50$) | Triage Operating Point ($\tau^* = 0.40$) |
| :--- | :---: | :---: |
| **Accuracy** | $87.62\% \pm 2.21\%$ | — |
| **Sensitivity (Recall)** | $89.87\% \pm 2.95\%$ | **91.05%** |
| **Specificity** | $85.34\% \pm 4.98\%$ | **78.64%** |
| **AUC-ROC** | $0.923 \pm 0.015$ | $0.923$ |
| **Weighted F1-Score** | $0.878$ | — |
| **Brier Score / ECE** | $0.108$ / $0.061$ | Well-calibrated |
| **Clinician Concordance** | Cohen's $\kappa_C = 0.82$ ($p < 0.001$) | Non-inferior ($p = 0.52$) |

### Edge & Compute Profile

| Precision / Hardware | Model Size | Latency (Batch=1) | Sensitivity |
| :--- | :---: | :---: | :---: |
| **FP32 (NVIDIA T4 GPU)** | $99.0\text{ MB}$ | $17.0\text{ ms}$ | $89.87\%$ |
| **FP16 (NVIDIA T4 GPU)** | $49.5\text{ MB}$ | $10.4\text{ ms}$ | $89.87\%$ |
| **INT8 Dynamic (NVIDIA T4 GPU)** | **$25.4\text{ MB}$** | **$6.8\text{ ms}$** | **$88.60\%$** |
| **FP32 (Xeon Gold 6248 1-Core CPU)** | $99.0\text{ MB}$ | $120.0\text{ ms}$ | $89.87\%$ |
| **INT8 Dynamic (Xeon Gold 1-Core CPU)** | **$25.4\text{ MB}$** | **$78.0\text{ ms}$** | **$88.60\%$** |

---

## Repository Structure

```text
CAMS-Net/
├── configs/
│   └── cams_net_default.yaml              # Training, CLAHE, and hyperparameter configuration
├── splits/
│   └── fold_indices.pkl                   # Deterministic patient-disjoint 5-fold splits (seed=42)
├── src/
│   ├── model.py                           # Dual-stem, CAG, SLR-ResNet50, and MAAH definitions
│   ├── preprocessing.py                   # CIELAB-CLAHE + raw dual-path transforms & augmentations
│   ├── morphology.py                      # LBP, Gabor, Sobel, 2D FFT, and Haar wavelet extractors
│   ├── train.py                           # 5-fold cross-validation training loop with AdamW & LR finder
│   ├── evaluate.py                        # Out-of-fold metrics, calibration (ECE/Brier), & threshold sweep
│   └── generate_grid_gradcam.py           # Paired 4x5 raw vs. Grad-CAM lesion localization grid generator
├── result images/
│   ├── architecture.png                   # End-to-end CAMS-Net architecture diagram
│   ├── Fig1_Preprocessing_Comparison.png  # LAB-CLAHE vs. baseline preprocessing comparison
│   ├── Fig2_Noise_And_Edges.png           # Spatial filtering and Canny edge maps
│   ├── Fig3_Frequency_Wavelet.png         # 2D FFT spectrum and Haar wavelet sub-bands
│   ├── Fig4_Texture_Features.png          # LBP, Gabor, Sobel, and Laplacian texture visualizations
│   ├── oral_cancer_final_journal.png      # Confusion matrix, ROC curve, and fold stability plots
│   ├── journal_pipeline_and_features.png  # Step-by-step pipeline and Grad-CAM overlay illustration
│   ├── oral_cancer_grid_4x5_raw.png       # 4x5 raw clinical cohort grid (2000 x 1600 px)
│   └── oral_cancer_grid_4x5_predicted.png # 4x5 Grad-CAM + lesion contour grid (2000 x 1600 px)
├── main.tex                               # IEEE JBHI LaTeX manuscript source
├── requirements.txt                       # Python dependencies
└── README.md                              # Project documentation
