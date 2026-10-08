# 🏥 LIGHTCURE

## Postoperative Dynamic Monitoring System for HCC Ablation Therapy

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://lightcure.streamlit.app)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

---

## 📖 Overview

LIGHTCURE is a deep learning-based dynamic monitoring system for hepatocellular carcinoma (HCC) ablation therapy. Unlike static preoperative models, LIGHTCURE performs **sequential risk prediction** at three postoperative time points (T0, T1, T3), incorporating both imaging features and IHC markers, and supports **missing values at inference** through a missing-adaptive encoder.

### Key Features

- ✅ **Multi-Timepoint Prediction**: Predicts future LTP risk at three time points:
  - **T0 (preoperative)**: predicts LTP within 0–12 months
  - **T1 (3 months post-op)**: predicts LTP within 3–12 months
  - **T3 (6 months post-op)**: predicts LTP within 6–24 months
- ✅ **Missing-Adaptive Input**: Postoperative and IHC variables are fully optional; missing values are handled by learned missing embeddings
- ✅ **IHC Heat Phenotype**: Uses HSP70, HIF-1α, BCL-2 to derive a heat-tolerance phenotype for auxiliary interpretation
- ✅ **Risk Stratification**: Three-tier risk grouping (Low < 0.2, Intermediate 0.2–0.5, High ≥ 0.5)
- ✅ **Privacy Protection**: All computations run locally; no patient data is stored

### Research Foundation

- Multi-center datasets with 5,228 patients
- **11 preoperative variables** (imaging + clinical)
- **13 postoperative variables** (6 from 3-month + 7 from 6-month follow-up)
- **8 IHC markers** (HSP70, HIF-1α, BCL-2, MVI, E-cadherin, CK19, VEGF, MMP-9)
- Deep learning architecture with missing-adaptive encoder (LIGHTCURE-Dynamic)
- Focal Loss + Consistency Loss for class-imbalanced risk prediction

---

## 🚀 Live Demo

Access the deployed version: [https://lightcure.streamlit.app](https://lightcure.streamlit.app)

> **Note**: The first visit may take 10–30 seconds to wake up (free tier auto-sleep).

---

## 📁 Project Structure
