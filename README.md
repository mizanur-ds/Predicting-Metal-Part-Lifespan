# 💡 Predictive Modeling of Metal Part Lifespan

This project predicts the lifespan of metal parts using supervised machine learning. It involves regression (to predict lifespan as a continuous variable) and classification (by clustering lifespan into discrete categories).

---

## 📁 Table of Contents

- [Project Objective](#project-objective)
- [Data Description](#data-description)
- [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
- [Methodology](#methodology)
  - [Regression Modeling](#regression-modeling)
  - [Classification Modeling](#classification-modeling)
- [Results](#results)
- [Model Evaluation](#model-evaluation)
- [Conclusion](#conclusion)
- [References](#references)

---

## 🎯 Project Objective

The objective is to develop machine learning models that can predict the lifespan of metal parts from manufacturing features. This insight can help manufacturers make better decisions regarding quality control and resource allocation.

---

## 📊 Data Description

The dataset contains several numerical and categorical features related to metal manufacturing processes. The main target variable is the **Lifespan (in hours)** of each metal part.

---

## 📊 Exploratory Data Analysis (EDA)

- **Target Distribution**: The `Lifespan` variable is right-skewed.
- **Outliers**: Detected in features like `CoolingRate` and `ProcessTemp`.
- **Correlation Analysis**: Strong correlation between `Lifespan` and `Hardness`, `TreatmentTime`, etc.
- **Categorical Analysis**: Boxplots showed noticeable class-based differences in `MaterialType`.
- **Missing Values**: Handled using mean/mode imputation.
- **Visualization**: Histograms, scatter plots, box plots, and correlation matrix were used.

---

## ⚙️ Methodology

### 🔁 Regression Modeling

#### 1. Polynomial Ridge Regression
- Captures non-linear patterns using polynomial features
- L2 regularization to reduce overfitting
- Best performance at polynomial degree = 2
- Hyperparameters tuned via Grid Search

#### 2. Random Forest Regression
- Handles non-linearity and feature interactions
- Less sensitive to feature scaling
- Hyperparameter tuning using Randomized Search

**Preprocessing**:
- Train/Validation/Test split (80/10/10)
- Standard Scaling for numeric data
- One-Hot Encoding for categorical variables

---

### 📦 Classification Modeling

#### Feature Engineering
- **K-Means Clustering** used to transform lifespan into 6 discrete classes
- New feature: `Target_Hour`
- Clustering guided by Elbow Method

#### Models Used:
1. **Logistic Regression**
   - Used as a baseline
   - Regularization parameter `C` optimized

2. **Artificial Neural Network (ANN)**
   - Multi-layer feedforward network
   - Dropout regularization added
   - Learning rate tuned for best performance

---

## 📈 Results

### Regression

| Model                 | MSE        | R² Score |
|----------------------|------------|----------|
| Polynomial Ridge      | **24,066.54** | **0.81**  |
| Random Forest         | ~30,500    | 0.74     |

---

### Classification

#### Logistic Regression

| Metric      | Before Tuning | After Tuning |
|-------------|---------------|--------------|
| Accuracy    | 0.18          | 0.24         |
| Precision   | 0.14          | 0.15         |
| Recall      | 0.28          | 0.19         |
| F1 Score    | 0.16          | 0.16         |

#### ANN with Dropout

| Metric      | Without Dropout | With Dropout |
|-------------|------------------|--------------|
| Accuracy    | 0.32             | **0.56**     |
| Precision   | 0.30             | 0.56         |
| Recall      | 0.29             | 0.57         |
| F1 Score    | 0.28             | **0.56**     |

---

## 🧪 Model Evaluation

- **Polynomial Ridge Regression** is best for lifespan prediction due to its high R² and low MSE.
- ANN outperforms Logistic Regression in classification but still struggles with class imbalance.
- Preprocessing, regularization, and dropout helped mitigate overfitting.

---

## ✅ Conclusion

Polynomial Ridge Regression (degree 2) is the best model for deployment due to its strong ability to generalize and handle non-linearity. Classification models were less effective and sensitive to imbalance and data complexity.

---

## 📚 References

- [Miller, A. (2022)](https://www.sciencedirect.com/science/article/pii/S092523122101907X)
- [Ostertagová, E. (2012)](https://www.sciencedirect.com/science/article/pii/S1877705812046085)
- [Sperandei, S. (2014)](https://www.researchgate.net/publication/260810482_Understanding_logistic_regression_analysis)
- [Tu, J. (1996)](https://www.sciencedirect.com/science/article/pii/S0895435696000029)
- [Ao, Y. (2019)](https://www.sciencedirect.com/science/article/pii/S0920410518310635)
