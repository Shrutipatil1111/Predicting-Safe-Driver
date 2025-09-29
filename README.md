# 🚗 Predicting Safe Drivers – Advanced Driver Risk Segmentation & Predictive Modeling  

## 📝 Executive Summary – Driving Actuarial Precision 🎯  
This project delivers a **machine learning solution** for predicting safe drivers, a critical need in the **insurance and automotive sectors**. Accurate risk segmentation enables:  
- Optimized premium pricing  
- Minimized loss ratios  
- Enhanced profitability  

We benchmarked multiple algorithms and confirmed that **XGBoost** delivers the superior predictive performance required for **strategic risk underwriting**, achieving a **Cross-Validation Score of ≈87.5%** on the test set.  

---

## 🎯 Strategic Objectives  
- **Risk Mitigation**: Identify high-risk drivers (unsafe) to reduce claim payouts.  
- **Precision Pricing**: Support granular, risk-based pricing for competitive premiums.  
- **Modeling Excellence**: Benchmark diverse ML models to select a robust, scalable, and explainable predictor.  

---

## 🔑 Key Findings & Model Performance  

- **Dataset Size**: 30,240 driver records, 17 features  
- **Class Imbalance**: ~71% Safe vs. 29% Unsafe → addressed with Random Under-Sampling  
- **Evaluation Metrics**: Cross-Validation Score, ROC AUC, MCC (Matthews Correlation Coefficient)  

### 📊 Model Benchmark Summary  

| **Model**                | **CV Score (Test)** | **Key Value Proposition** |
|---------------------------|---------------------|----------------------------|
| **XGBoost Classifier**    | ≈87.5%              | Best-in-class performance; highly scalable for deployment |
| **Gradient Boosting (GB)**| ≈86.5%              | Strong alternative; effective trade-off between complexity and performance |
| **SVC**                   | ≈85.5%              | Reliable non-linear classifier; computationally heavier |

---

## 🔍 Key Feature Drivers  
- **Credit History** (critical; Box-Cox transformed for stability)  
- **Engine HP**  
- **Years of Experience**  
- **Annual Claims**  
- **Miles Driven Annually**  

These features emerged as primary differentiators of safe vs. unsafe drivers.  

---

## ⚙️ Technical Architecture  

### 🔧 Methodology & Feature Engineering  
1. **Data Quality**: Imputed missing values in `Miles_driven_annually`  
2. **Transformation**: Applied **Box-Cox normalization** to `credit_history`  
3. **Feature Selection**: Used `SelectKBest` (ANOVA F-test)  
4. **Dimensionality Reduction**: PCA tested for variance compression  
5. **Imbalance Handling**: Random Under-Sampling to balance safe vs. unsafe classes  

### 🛠️ Technology Stack  

| **Category**   | **Libraries/Tools** |
|----------------|----------------------|
| Language       | Python |
| Core ML        | Scikit-learn, NumPy, Pandas |
| Ensemble/Boost | XGBoost, Gradient Boosting, StackingClassifier |
| Evaluation     | accuracy_score, roc_auc_score, confusion_matrix, Matthews Correlation Coefficient (MCC) |

---

## 📦 Installation & Usage  


#### Clone the repository
```bash
git clone https://github.com/yourusername/predicting-safe-drivers.git
cd predicting-safe-drivers
```
#### Install dependencies
```bash
pip install -r requirements.txt
```

## 📜 License  
This repository is structured for enterprise-grade delivery and professional handoff. 

© Shruti Vilas Patil – All rights reserved.  
Licensed under the MIT License.

# Run the notebook
jupyter notebook "Predicting Safe Drivers ML (1).ipynb"
