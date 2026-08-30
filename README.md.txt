# 🏥 LIGHTCURE

## Personalized Decision Support System for HCC Ablation Therapy

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://lightcure.streamlit.app)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

---

## 📖 Overview

LIGHTCURE is a deep learning-based clinical decision support system for hepatocellular carcinoma (HCC) ablation therapy. It analyzes 20 core clinical variables to predict the 12-month local tumor progression (LTP) probability for both Radiofrequency Ablation (RFA) and Irreversible Electroporation (IRE), providing personalized treatment recommendations.

### Key Features

- ✅ **Dual-Model Prediction**: Predicts LTP probabilities for RFA and IRE separately
- ✅ **ΔP Benefit Score**: Calculates individual treatment benefit difference (RFA - IRE)
- ✅ **Treatment Recommendation**: Provides IRE, RFA, or Either recommendation based on ΔP
- ✅ **Risk Stratification**: Three-tier risk grouping (Low/Intermediate/High)
- ✅ **Batch Prediction**: Supports CSV file batch processing
- ✅ **Privacy Protection**: All computations run locally; no patient data is stored

### Research Foundation

- 13 multi-center datasets with 5,228 patients
- 20 core predictive variables (12 pre-treatment + 8 post-treatment dynamic features)
- Deep learning architecture (LIGHTCURE-Predictor)
- Biological mechanism validation (LIGHTCURE-Mechanism)
- Counterfactual inference framework (LIGHTCURE-Optimizer)

---

## 🚀 Live Demo

Access the deployed version: [https://lightcure.streamlit.app](https://lightcure.streamlit.app)

> **Note**: The first visit may take 10-30 seconds to wake up (free tier auto-sleep).

---

## 📁 Project Structure
