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

The dataset contains several numerical and categorical features related to metal manufacturing processes. The main target variable is the **Lifespan** of each metal part.

---

## 📊 Exploratory Data Analysis (EDA)

- **Target Distribution**: The `Lifespan` variable is right-skewed.
- **Outliers**: Detected in features like `CoolingRate` and `ProcessTemp`.
- **Correlation Analysis**: Strong correlation between `Lifespan` and `Hardness`, `TreatmentTime`, etc.
- **Categorical Analysis**: Boxplots showed noticeable class-based differences in `MaterialType`.
- **Missing Values**: Handled using mean imputation.
- **Visualization**: Histograms, scatter plots, box plots, and correlation matrix were used.
<table>
  <tr>
    <td align="center">
      <img src="https://github.com/user-attachments/assets/88bcbb3a-bebb-4985-ae6a-5dd017988da1" width="100%">
      <br><strong>Figure 1: Correlation Heatmap</strong>
    </td>
    <td align="center">
      <img src="https://github.com/user-attachments/assets/469e2bfe-e714-4064-bd30-8075e1c8200e" width="100%">
      <br><strong>Figure 2: Box plots of the Lifespan feature with all categorical features</strong>
    </td>
  </tr>
  <tr>
    <td align="center">
      <img src="https://github.com/user-attachments/assets/933dcc3e-a7c2-4856-8cd0-b9d27ee5eb03" width="100%">
      <br><strong>Figure 3: Scatter plots of Lifespan with all numerical features 1</strong>
    </td>
    <td align="center">
      <img src="https://github.com/user-attachments/assets/e244a034-fdcd-4ea3-b3b5-6b61b976c755" width="100%">
      <br><strong>Figure 4: Scatter plots of Lifespan with all numerical features 2</strong>
    </td>
  </tr>
</table>





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
<p align="center">
  <img src="https://github.com/user-attachments/assets/5d4e79d3-c205-426a-b6d8-fdafa4905341" width="45%" style="height: 300px; object-fit: contain;" />
  <img src="https://github.com/user-attachments/assets/de746553-c368-4519-a7b2-925f1396c919" width="45%" />
</p>

<p align="center">
  <strong>Figure 5: Number of Clusters vs Inertia</strong>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
  <strong>Figure 6: Inertia Values for Different Clusters</strong>
</p>




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

<p align="center">
  <img src="https://github.com/user-attachments/assets/9aafd6c2-de44-416d-91bd-98b268d32ffa" width="50%" />
</p>

<p align="center"><strong>Figure: Logistic Regression Confusion Matrix (After Tuning)</strong></p>


#### ANN with Dropout

| Metric      | Without Dropout | With Dropout |
|-------------|------------------|--------------|
| Accuracy    | 0.32             | **0.56**     |
| Precision   | 0.30             | 0.56         |
| Recall      | 0.29             | 0.57         |
| F1 Score    | 0.28             | **0.56**     |


<p align="center">
  <img src="https://github.com/user-attachments/assets/6fa4076d-d31f-4167-a132-4b9a70b32786" width="50%" />
</p>

<p align="center"><strong>Figure: ANN Confusion Matrix (With Regularization))</strong></p>



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
