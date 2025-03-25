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
