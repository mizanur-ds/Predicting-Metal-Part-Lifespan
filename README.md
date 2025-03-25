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
<img src="https://github.com/user-attachments/assets/65a6e432-0996-48c0-8fd5-2c7f0d96601a" alt="Image Description" width="600" height="300"/>

<img src="https://github.com/user-attachments/assets/5512d486-2975-44b3-90da-a687b86dec19" alt="Image Description" width="400" height="300"/> <img src="https://github.com/user-attachments/assets/c692f630-5bf5-4cba-9896-59e552b9b064" alt="Sample Image" width="400" height="300"/>

The target feature 'Lifespans' ranges from 418 to 2135, averaging around 1300. Numerical features like 'Heat Treat Time', 'Nickel%', and 'Iron%' show considerable variability, reflecting diverse manufacturing processes. A correlation heatmap highlights a positive correlation between cooling rate and small defects, while 'Nickel%' and 'Iron%' are negatively correlated. No strong linear relationship between lifespan and numerical features was found. Box plots reveal variability in lifespan across categorical features, with Continuous casting methods linked to longer lifespans. Scatter plots suggest complex, non-linear relationships with lifespan. Consequently, all numerical features and key categorical variables like 'partType' and 'castType' will be included in the machine learning model.

