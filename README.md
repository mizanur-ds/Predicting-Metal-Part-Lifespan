# Predicting-Metal-Part-Lifespan
**Predict metal part lifespan based on manufacturing and material features. Regression and classification models are applied and compared to identify the most accurate prediction method.**

---

## Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Installation and Setup](#setup)
- [Exploratory Data Analysis](#exploratory-data-analysis)
- [Data Preprocessing](#data-dreprocessing)
- [Feature Crafting to Make Category of Target Variable](#feature-crafting)
- [Model Development](#model-development)
- [Evaluation Metrics](#evaluation-metrics)
- [Results](#result)
---
## Overview
This work aims to predict the lifespan of metal parts using manufacturing data and material features. The dataset includes information on manufacturing processes and material characteristics, along with some known metal part lifespans that will serve as training data. To achieve the best predictions, various models from both regression and classification categories, as well as model combinations, will be explored. The model that demonstrates the lowest error, determined through statistical evaluation, will be selected as the optimal solution.

## Dataset
The Dataset contains 1000 rows, each representing a metal part with various manufacturing and 16 features/columns, with “Lifespan” as the target feature. Columns are a mix of numerical and categorical features.

---
## Installation and Setup
```bash
pip install -r requirements.txt
python metal_projection.py
```
---
##  Exploratory Data Analysis (EDA)
<img src="https://github.com/user-attachments/assets/65a6e432-0996-48c0-8fd5-2c7f0d96601a" alt="Image Description" width="1000" height="350"/>

<img src="https://github.com/user-attachments/assets/5512d486-2975-44b3-90da-a687b86dec19" alt="Image Description" width="450" height="300"/> <img src="https://github.com/user-attachments/assets/c692f630-5bf5-4cba-9896-59e552b9b064" alt="Sample Image" width="450" height="300"/>

The target feature 'Lifespans' ranges from 418 to 2135, averaging around 1300. Numerical features like 'Heat Treat Time', 'Nickel%', and 'Iron%' show considerable variability, reflecting diverse manufacturing processes. A correlation heatmap highlights a positive correlation between cooling rate and small defects, while 'Nickel%' and 'Iron%' are negatively correlated. No strong linear relationship between lifespan and numerical features was found. Box plots reveal variability in lifespan across categorical features, with Continuous casting methods linked to longer lifespans. Scatter plots suggest complex, non-linear relationships with lifespan. Consequently, all numerical features and key categorical variables like 'partType' and 'castType' will be included in the machine learning model.

---
## Data Preprocessing
The following steps are performed during data preprocessing:

1.Feature Scaling: Standard scaling is applied to numerical features to ensure effective regularization.

2.Train-Validation-Test Split: The data is split into 80% training, 10% validation, and 10% testing.

3.Feature Encoding: Categorical features are encoded using One-Hot-Encoding.

---
## Feature Crafting to Make Category of Target Variable

An unsupervised clustering technique, **K-Means**, was used to categorize lifespan hours into multiple classes based on their lifespan and related features. The **Elbow Method** was applied to analyze the sum of squared distances (inertia) for k values ranging from 1 to 10. This analysis revealed an optimal k value of 6 clusters, as indicated by the elbow plot, which showed a significant reduction in inertia up to this point. This approach resulted in six distinct clusters based on lifespan, reflecting subtle variations in manufacturing quality and process parameters.

<img src="https://github.com/user-attachments/assets/3df5df1e-422e-40d0-8ef9-1aeb379fa378" alt="Image Description" width="500" height="300"/> <img src="https://github.com/user-attachments/assets/f1a2dc44-84cf-4381-9883-91071a551f4c" alt="Image Description" width="500" height="300"/>

**Fig-1:** Number of clusters vs Inertia  
**Fig-2:** Inertia Values for Different Clusters

Based on the results, the values of the Lifespan feature were distributed into six categories under a new feature named **‘Target Hour’**. Additionally, only six clusters maintained the threshold of 1500 hours. In cases with fewer than six clusters, one category ranged between 1400 to 1600 hours, violating the threshold rule.

Below is the distribution of Lifespan across the six clusters:

| **Cluster** | **Lifespan Range (Hours)** |
|-------------|----------------------------|
| 0           | 1300.66 – 1527.35          |
| 1           | 850.00 – 1078.50           |
| 2           | 1082.10 – 1299.90          |
| 3           | 1774.38 – 2134.53          |
| 4           | 417.99 – 845.40            |
| 5           | 1529.60 – 1761.96          |
---

## Model Development

### Model Choices

For predicting metal part lifespan and performing classification tasks, several models have been selected based on the data characteristics and problem requirements. The chosen models include Polynomial Ridge Regression, Random Forest Regression, Logistic Regression, and Artificial Neural Network (ANN).

#### Polynomial Ridge Regression
- **Why Chosen**: This model was selected to capture non-linear relationships between features and the continuous target variable, 'Lifespan.' By adding polynomial features, it allows the model to fit complex patterns that linear regression may miss. Ridge regularization helps prevent overfitting, which is important in datasets with many features and potential multicollinearity.
- **Benefits**: Effective for datasets with non-linear relationships and for preventing overfitting through regularization.

#### Random Forest Regression
- **Why Chosen**: Random Forest Regression is an ensemble learning technique capable of handling complex, non-linear relationships in the data. By using multiple decision trees, it aggregates predictions to reduce variance and improve accuracy without needing to explicitly specify the relationships.
- **Benefits**: Robust, automatic in uncovering feature interactions, and resistant to overfitting with proper tuning.

#### Logistic Regression
- **Why Chosen**: Selected for the classification task, where 'Lifespan' is divided into multiple classes. Logistic Regression is a simple, efficient model for multi-class classification tasks, especially when there are linear separations between the classes.
- **Benefits**: Interpretable, computationally efficient, and useful as a baseline model for comparison with more complex models.

#### Artificial Neural Network (ANN)
- **Why Chosen**: ANNs are highly effective at capturing complex, non-linear patterns in data. With high-dimensional features and potential non-linear relationships in manufacturing, ANNs can model intricate interactions between features and the target variable.
- **Benefits**: Capable of modeling highly complex relationships, particularly useful in high-dimensional, intricate datasets, with regularization techniques like dropout to reduce overfitting.

## Evaluation Metrics 

#### Regression Metrics
For regression models, the following metrics were used:

- **Mean Squared Error (MSE):** Measures the average squared difference between predicted and actual values. Lower MSE indicates better model performance.
- **R-squared (R²) Score:** Indicates the proportion of variance in the target variable explained by the model. Values closer to 1 indicate better performance.

#### Classification Metrics
For classification models, these metrics were used:

- **Accuracy:** Measures the proportion of correct predictions. Useful but may be insufficient for imbalanced datasets.
- **Precision (Macro):** Measures the percentage of true positive predictions across all classes equally.
- **Recall (Macro):** Measures the percentage of actual positive instances correctly predicted across all classes equally.
- **F1 Score (Macro):** The harmonic mean of precision and recall, balancing both metrics for better evaluation of imbalanced data.

---

### Performance Results

#### Regression Models (Polynomial Ridge Regression vs. Random Forest Regression)

| **Model**                  | **MSE**    | **R² Score** |
|----------------------------|-------------|---------------|
| Polynomial Ridge Regression | 24066.54    | 0.81          |
| Random Forest Regression    | 40010.09    | 0.69          |

**Interpretation:**  
- Polynomial Ridge Regression outperforms Random Forest Regression with a significantly lower MSE and higher R² score. This suggests it better captures the relationships between features and the target variable.  
- Random Forest Regression, while still effective, may struggle to model complex non-linear relationships as efficiently as Polynomial Ridge Regression.

---

#### Classification Models (Logistic Regression vs. ANN)

| **Metric**       | **Logistic Regression (After Tuning)** | **ANN (With Regularization)** |
|------------------|----------------------------------------|-------------------------------|
| Accuracy           | 0.24                                   | 0.56                          |
| Precision (Macro)  | 0.15                                   | 0.56                          |
| Recall (Macro)     | 0.19                                   | 0.57                          |
| F1 Score (Macro)   | 0.16                                   | 0.56                          |

##### Logistiction Regression Model Confusion Matrix (After Tuning)
<img src="https://github.com/user-attachments/assets/c5c306a4-3c23-4327-84e4-160d6f579529" alt="Image Description" width="500" height="300"/>

##### ANN Model Confusion Matrix (After Tuning)
<img src="https://github.com/user-attachments/assets/a02945b8-b46d-43b4-8579-a5075bab8739" alt="Image Description" width="500" height="300"/>

**Interpretation:**  
- The **ANN with Dropout Regularization** significantly outperforms Logistic Regression across all metrics. This improvement highlights ANN's ability to capture complex patterns in the data, especially with regularization to reduce overfitting.  
- Logistic Regression, while simple and interpretable, struggles to model the dataset’s complexity, particularly in handling minority classes.

---

## Result

Polynomial Ridge Regression outperformed Random Forest Regression with a lower **MSE** of **24,066.54** and a higher **R²** score of **0.81**, effectively capturing non-linear relationships in the Lifespan data.  

For classification tasks, **Logistic Regression** struggled with **24%** accuracy, while the **Artificial Neural Network (ANN)** improved accuracy to **56%**, though it still faced overfitting and class imbalance issues.  

As highlighted in the **Data Exploration Section**, since the target feature is a continuous numerical value, linear regression—particularly **Ridge Regression** with polynomial features—proved to be the most effective in capturing non-linear relationships and mitigating overfitting.  

Ultimately, the **Ridge Regression** model using a **2-degree polynomial** delivered the best overall performance.

---

