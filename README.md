# Predicting-Metal-Part-Lifespan
Predict metal part lifespan based on manufacturing and material features. Regression and classification models are applied and compared to identify the most accurate prediction method.

## Table of Contents
- [Overview](#overview)
- [Data](#data)
- [Setup](#setup)
- [Exploratory Data Analysis](#exploratory-data-analysis)
- [Model Training](#model-training)
- [Results](#results)
- [References](#references)
- [Contact](#contact)

## Overview
This work aims to predict the lifespan of metal parts using manufacturing data and material features. The dataset includes information on manufacturing processes and material characteristics, along with some known metal part lifespans that will serve as training data. To achieve the best predictions, various models from both regression and classification categories, as well as model combinations, will be explored. The model that demonstrates the lowest error, determined through statistical evaluation, will be selected as the optimal solution.

## Dataset
-Link:

The Dataset contains 1000 rows, each representing a metal part with various manufacturing and 16 features/columns, with “Lifespan” as the target feature. Columns are a mix of numerical and categorical features.
## Installation and Setup
```bash
pip install -r requirements.txt
python metal_projection.py
```
##  Exploratory Data Analysis (EDA)
<img src="https://github.com/user-attachments/assets/65a6e432-0996-48c0-8fd5-2c7f0d96601a" alt="Image Description" width="800" height="350"/>

<img src="https://github.com/user-attachments/assets/5512d486-2975-44b3-90da-a687b86dec19" alt="Image Description" width="500" height="300"/> <img src="https://github.com/user-attachments/assets/c692f630-5bf5-4cba-9896-59e552b9b064" alt="Sample Image" width="500" height="300"/>

The target feature 'Lifespans' ranges from 418 to 2135, averaging around 1300. Numerical features like 'Heat Treat Time', 'Nickel%', and 'Iron%' show considerable variability, reflecting diverse manufacturing processes. A correlation heatmap highlights a positive correlation between cooling rate and small defects, while 'Nickel%' and 'Iron%' are negatively correlated. No strong linear relationship between lifespan and numerical features was found. Box plots reveal variability in lifespan across categorical features, with Continuous casting methods linked to longer lifespans. Scatter plots suggest complex, non-linear relationships with lifespan. Consequently, all numerical features and key categorical variables like 'partType' and 'castType' will be included in the machine learning model.

## Data Preprocessing
The following steps are performed during data preprocessing:

1.Feature Scaling: Standard scaling is applied to numerical features to ensure effective regularization.

2.Train-Validation-Test Split: The data is split into 80% training, 10% validation, and 10% testing.

3.Feature Encoding: Categorical features are encoded using One-Hot-Encoding.

## Feature Crafting to Make Category of Target Variable

An unsupervised clustering technique, **K-Means**, was used to categorize lifespan hours into multiple classes based on their lifespan and related features. The **Elbow Method** was applied to analyze the sum of squared distances (inertia) for k values ranging from 1 to 10. This analysis revealed an optimal k value of 6 clusters, as indicated by the elbow plot, which showed a significant reduction in inertia up to this point. This approach resulted in six distinct clusters based on lifespan, reflecting subtle variations in manufacturing quality and process parameters.

<img src="https://github.com/user-attachments/assets/3df5df1e-422e-40d0-8ef9-1aeb379fa378" alt="Image Description" width="500" height="300"/>
![image](https://github.com/user-attachments/assets/f1a2dc44-84cf-4381-9883-91071a551f4c)

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

