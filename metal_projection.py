#!/usr/bin/env python
# coding: utf-8

# ## Importing useful packages

# In[2]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn.linear_model
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, PolynomialFeatures, LabelEncoder, OneHotEncoder
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from keras.losses import CategoricalCrossentropy
from keras.metrics import CategoricalAccuracy
import warnings
warnings.filterwarnings('ignore')
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, precision_score, recall_score, f1_score, classification_report, balanced_accuracy_score # Various classification metrics we may find useful
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.metrics import SparseCategoricalAccuracy
import seaborn as sns; sns.set()  # for plot styling
from sklearn.metrics import  accuracy_score, precision_score, recall_score, f1_score, balanced_accuracy_score # required for evaluating classification models
from sklearn.preprocessing import StandardScaler # We will be using the inbuilt preprocessing functions sklearn provides
from sklearn.model_selection import train_test_split # A library that can automatically perform data splitting for us
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler, MinMaxScaler, PolynomialFeatures, LabelEncoder, OneHotEncoder
from tensorflow.keras.activations import sigmoid, linear, relu, softmax
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.losses import MeanSquaredError, BinaryCrossentropy, CategoricalCrossentropy, SparseCategoricalCrossentropy  # We will be using TFs MSE loss function for regression and BinaryCross Entropy for classification.
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.regularizers import L1, L2
from tensorflow.keras.metrics import BinaryAccuracy, Accuracy
from tensorflow.keras.callbacks import EarlyStopping


# # Data Exploration

# ## Data Loading and Manipulation

# In[4]:


# Reading the data
data = pd.read_csv("COMP1801_Coursework_Dataset.csv")
df = pd.DataFrame(data)
df.head()


# ## Data Preprocessing and EDA

# In[5]:


# finding various attributes of the dataset
print('Shape of the data (rows and columns):')
print(df.shape)
print()
print('List of the column names:')
print(df.columns)
print()
print('The data type of all the columns (all just floats here):')
print(df.dtypes)


# In[6]:


# Making list of numerical and categorical features
numeric_cols = df.select_dtypes(include=[np.number]).columns
categorical_cols = df.select_dtypes(include=['object']).columns
num_without_target = numeric_cols.drop('Lifespan')
target = ['Lifespan']
all_features = df.columns


# In[7]:


print(numeric_cols)
print(categorical_cols)
print(num_without_target)
print(all_features)
print(target)


# In[8]:


df.describe()


# In[9]:


# Checking the null valuses
print(df.isnull().sum())


# #### Creating scatter plots using the Lifespan feature alongside all numerical features individually

# In[10]:


fig, axes = plt.subplots(2, 3, figsize=(10, 8))

# Flatten the axes array for easier indexing
axes = axes.flatten()

# Loop through the features and create scatter plots
for idx, feature in enumerate(num_without_target[:6]):
    if idx < len(axes):  # Ensure we don't exceed the number of subplots
        axes[idx].scatter(df[feature], df['Lifespan'])
        axes[idx].set_title(f'{feature} vs Lifespan')
        axes[idx].set_xlabel(feature)
        axes[idx].set_ylabel('Lifespan')

plt.tight_layout()  # Adjust layout to prevent overlap
plt.show()


# In[11]:


fig, axes = plt.subplots(2, 3, figsize=(10, 8))

# Flatten the axes array for easier indexing
axes = axes.flatten()

# Loop through the features and create scatter plots
for idx, feature in enumerate(num_without_target[6:]):
    if idx < len(axes):  # Ensure we don't exceed the number of subplots
        axes[idx].scatter(df[feature], df['Lifespan'])
        axes[idx].set_title(f'{feature} vs Lifespan')
        axes[idx].set_xlabel(feature)
        axes[idx].set_ylabel('Lifespan')

plt.tight_layout()  # Adjust layout to prevent overlap
plt.show()


# #### Creating box plot using the Lifespan feature alongside all categorical features

# In[12]:


categorical_features = ['partType', 'microstructure', 'seedLocation', 'castType']
fig, axes = plt.subplots(2, 2, figsize=(10, 8))

# Flatten the axes array for easier indexing
axes = axes.flatten()

# Loop through the features and create box plots using Seaborn
for idx, feature in enumerate(categorical_features):
    if idx < len(axes):  # Ensure we don't exceed the number of subplots
        sns.boxplot(x=feature, y='Lifespan', data=df, ax=axes[idx])
        axes[idx].set_title(f'Relationship between {feature} and Lifespan')
        axes[idx].set_xlabel(feature)
        axes[idx].set_ylabel('Lifespan')


plt.tight_layout()  # Adjust layout to prevent overlap
plt.show()


# #### Checking the Correlation among numerical features by Heatmap

# In[13]:


correlation_matrix = df[numeric_cols].corr()

# Figure size
plt.figure(figsize=(12,8))

# Creating heatmap
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
#

# Show the plot
plt.show()


# According to the heatmap it is clear that there have two strong linear relationship. Small_defects and colling_rate has a positive relation where nickel% and Iron% has negative relation.

# In[14]:


# Categorical features will be apply in model
categorical_final = ['partType', 'castType']


# # Regression Implementation

# ### Data Preprocessing

# ####  Split data in Train, Valid and Test set

# In[15]:


X = df.drop(columns=['Lifespan','microstructure', 'seedLocation'])
y = df['Lifespan']


# In[16]:


X_train_valid, X_test, y_train_valid, y_test = train_test_split(X, y, test_size=0.10, shuffle=True, random_state=0)
X_train, X_valid, y_train, y_valid = train_test_split(X_train_valid, y_train_valid, test_size=0.11, shuffle=True, random_state=0)


# #### Standardizing the numerical data  and  encoding the categorical data by OneHotEncoder

# In[17]:


scaler = StandardScaler()
onehot = OneHotEncoder(handle_unknown='ignore')

#Apply encoder and sclaing on data
numerical_transformer = Pipeline(steps=[
    ('scale', scaler)
])

categorical_transformer = Pipeline(steps=[
    ('categorical', onehot)
])

preprocessor =ColumnTransformer(transformers=[
    ('num', numerical_transformer, num_without_target),
    ('cat', categorical_transformer, categorical_final)
])

scal_encod_pipeline = Pipeline(steps=[
    ('preprocessor',preprocessor)
])


# In[18]:


X_train=scal_encod_pipeline.fit_transform(X_train)
X_valid=scal_encod_pipeline.fit_transform(X_valid)
X_test=scal_encod_pipeline.fit_transform(X_test)


# In[19]:


print('The shape of `X_train`:', X_train.shape)
print('The shape of `X_valid`:', X_valid.shape)
print('The shape of `X_test`:', X_test.shape)


# ## Linear Regression Model

# In[20]:


################
# Training Data
################
obj = sklearn.linear_model.LinearRegression()
obj.fit(X_train, y_train)

y_pred_train = obj.predict(X_train)
MSE_train = sklearn.metrics.mean_squared_error(y_train, y_pred_train)
R2_train = sklearn.metrics.r2_score(y_train, y_pred_train)

#The mean squared error loss
print('Mean squared loss of train:', MSE_train)
#The R2 score of train
print('R2 score of train:',R2_train)

################
# Validation Data
################
obj = sklearn.linear_model.LinearRegression()
obj.fit(X_valid, y_valid)

y_pred_valid = obj.predict(X_valid)
MSE_valid = sklearn.metrics.mean_squared_error(y_valid, y_pred_valid)
R2_valid = sklearn.metrics.r2_score(y_valid, y_pred_valid)

#The mean squared error loss
print('Mean squared loss of valid:', MSE_valid)
#The R2 score of train
print('R2 score of valid:',R2_valid)


# The linear model indicates a slight linear relationship between target features and other variables. It may be more effective to explore polynomial regression, as there could be a non-linear relationship.

# ## Polynomial

# In[21]:


# Separate features and target variable
X = df.drop(columns=['Lifespan','microstructure', 'seedLocation'])
y = df['Lifespan']

# Split the data into training, validation, and test sets
X_train_valid, X_test, y_train_valid, y_test = train_test_split(X, y, test_size=0.10, shuffle=True, random_state=0)
X_train, X_valid, y_train, y_valid = train_test_split(X_train_valid, y_train_valid, test_size=0.12, shuffle=True, random_state=0)

# Initialize lists to store MSE and R² values for each degree
mse_train_list = []
mse_valid_list = []
R2_train_list = []
R2_valid_list = []

# Define numerical and categorical feature columns (specify these lists)
num_without_target = X.select_dtypes(include=['float64', 'int64']).columns.tolist()  # List of numerical columns
categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()  # List of categorical columns

# Loop through polynomial degrees from 1 to 4
for degree in range(1, 5):
    print(f"Degree: {degree}")

    # Define the transformers
    poly = PolynomialFeatures(degree=degree)
    scaler = StandardScaler()
    onehot = OneHotEncoder(handle_unknown='ignore')

    # Apply PolynomialFeatures and Scaling only to numerical features
    numerical_transformer = Pipeline(steps=[
        ('poly', poly),
        ('scale', scaler)
    ])

    # Apply OneHotEncoder to categorical features
    categorical_transformer = Pipeline(steps=[
        ('onehot', onehot)
    ])

    # Combine numerical and categorical transformations using ColumnTransformer
    preprocessor = ColumnTransformer(transformers=[
        ('num', numerical_transformer, num_without_target),  # Apply poly and scale to numerical
        ('cat', categorical_transformer, categorical_cols)  # One-hot encode categorical features
    ])

    # Create a pipeline that applies preprocessing and then the model
    model_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', LinearRegression())  # Linear regression model
    ])

    # Fit the model on the training set
    model_pipeline.fit(X_train, y_train)

    # Make predictions and evaluate on training and validation sets
    y_pred_train = model_pipeline.predict(X_train)
    y_pred_valid = model_pipeline.predict(X_valid)

    mse_train = mean_squared_error(y_train, y_pred_train)
    mse_valid = mean_squared_error(y_valid, y_pred_valid)
    R2_train = r2_score(y_train, y_pred_train)
    R2_valid = r2_score(y_valid, y_pred_valid)

    mse_train_list.append(mse_train)
    mse_valid_list.append(mse_valid)
    R2_train_list.append(R2_train)
    R2_valid_list.append(R2_valid)

    # Print the results
    print("Training Set MSE:", mse_train)
    print("Training Set R2 Score:", R2_train)
    print("Validation Set MSE:", mse_valid)
    print("Validation Set R2 Score:", R2_valid)

# Plot MSE values for training and validation sets
degrees = [1, 2, 3, 4]
plt.plot(degrees, mse_train_list, label='MSE on training dataset')
plt.plot(degrees, mse_valid_list, label='MSE on validation dataset')

# Labels and title for the plot
plt.xlabel('Degree of polynomial features')
plt.ylabel('Mean squared error')
plt.title('Degree of polynomial features and mean squared error')

# Add legend to the plot
plt.legend()

# Show the plot
plt.show()

# Find the best degree based on validation MSE
best_degree = np.nanargmin(mse_valid_list) + 1  # +1 since degrees are 1-indexed
print('The best degree of polynomials:', best_degree)

# Final evaluation on the test set with the best degree
# Re-run the pipeline with the best degree on the full training set
best_poly = PolynomialFeatures(degree=best_degree)
numerical_transformer = Pipeline(steps=[
    ('poly', best_poly),
    ('scale', scaler)
])

preprocessor = ColumnTransformer(transformers=[
    ('num', numerical_transformer, num_without_target),
    ('cat', categorical_transformer, categorical_final)
])

# Final model pipeline
final_model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', Ridge())
])

# Fit on entire training/validation data
final_model_pipeline.fit(X_train_valid, y_train_valid)

# Test set predictions and evaluation
y_pred_test = final_model_pipeline.predict(X_test)
mse_test = mean_squared_error(y_test, y_pred_test)
R2_test = r2_score(y_test, y_pred_test)

print("Test Set MSE:", mse_test)
print("Test Set R2 Score:", R2_test)


# In[22]:


# Define the transformers
degree = 2
poly = PolynomialFeatures(degree=degree)
scaler = StandardScaler()
onehot = OneHotEncoder(handle_unknown='ignore')

# Apply PolynomialFeatures and Scaling only to numerical features
numerical_transformer = Pipeline(steps=[
    ('poly', poly),
    ('scale', scaler)
])

# Apply OneHotEncoder to categorical features
categorical_transformer = Pipeline(steps=[
    ('onehot', onehot)
])

# Combine numerical and categorical transformations using ColumnTransformer
preprocessor = ColumnTransformer(transformers=[
    ('num', numerical_transformer, num_without_target),  # Apply poly and scale to numerical
    ('cat', categorical_transformer, categorical_final)  # One-hot encode categorical features
])

# Create a pipeline that applies preprocessing and then the model
model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', Ridge())  # Ridge regression model
])

# Fit the model
model_pipeline.fit(X_train, y_train)

# Make predictions on training, validation, and test sets
y_pred_train = model_pipeline.predict(X_train)
y_pred_valid = model_pipeline.predict(X_valid)
y_pred_test = model_pipeline.predict(X_test)

# Calculate MSE and R² scores for each dataset
mse_train = mean_squared_error(y_train, y_pred_train)
R2_train = r2_score(y_train, y_pred_train)
mse_valid = mean_squared_error(y_valid, y_pred_valid)
R2_valid = r2_score(y_valid, y_pred_valid)
mse_test = mean_squared_error(y_test, y_pred_test)
R2_test = r2_score(y_test, y_pred_test)

# Append MSE and R² results to lists
mse_train_list.append(mse_train)
R2_train_list.append(R2_train)
mse_valid_list.append(mse_valid)
R2_valid_list.append(R2_valid)

# Print results
print("Training Set MSE:", mse_train)
print("Training Set R2 Score:", R2_train)
print("Validation Set MSE:", mse_valid)
print("Validation Set R2 Score:", R2_valid)
print("Test Set MSE:", mse_test)
print("\nTest Set R2 Score:", R2_test)


# ## Ridge regression

# ##### Finding the best value of 𝛼 by iterating 𝛼

# In[23]:


# Generate alpha values from 10^(-10) to 10^(10)
alpha_indices = np.arange(20)
alphas = 10.0 ** (alpha_indices - 10)

# Degree for polynomial transformation
degree = 2

# Arrays to store the training and validation MSE
mse_train_array = np.full(len(alphas), np.nan)
mse_valid_array = np.full(len(alphas), np.nan)

for alpha_index, alpha in enumerate(alphas):
    # Define the transformers
    poly = PolynomialFeatures(degree=degree)
    scaler = StandardScaler()
    onehot = OneHotEncoder(handle_unknown='ignore')

    # Apply PolynomialFeatures and Scaling only to numerical features
    numerical_transformer = Pipeline(steps=[
        ('poly', poly),
        ('scale', scaler)
    ])

    # Apply OneHotEncoder to categorical features
    categorical_transformer = Pipeline(steps=[
        ('onehot', onehot)
    ])

    # Combine numerical and categorical transformations
    preprocessor = ColumnTransformer(transformers=[
        ('num', numerical_transformer, num_without_target),
        ('cat', categorical_transformer, categorical_final)
    ])

    # Create a pipeline with preprocessing and Ridge regression with current alpha
    model_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', Ridge(alpha=alpha))
    ])

    # Fit the model
    model_pipeline.fit(X_train, y_train)

    # Predict and calculate MSE for training and validation sets
    y_pred_train = model_pipeline.predict(X_train)
    y_pred_valid = model_pipeline.predict(X_valid)

    mse_train = mean_squared_error(y_train, y_pred_train)
    mse_valid = mean_squared_error(y_valid, y_pred_valid)

    # Store the MSE values for this alpha
    mse_train_array[alpha_index] = mse_train
    mse_valid_array[alpha_index] = mse_valid
    print(f'alpha: {alpha}, Validation mean squared error: {mse_valid}.')

# Plot MSE against alphas for both training and validation sets
plt.plot(alphas, mse_train_array, label='MSE on training dataset')
plt.plot(alphas, mse_valid_array, label='MSE on validation dataset')
plt.xlabel(r'Regularization weights $\alpha$')
plt.ylabel('Mean squared error')
plt.title(r'Regularization weights $\alpha$ and mean squared error')
plt.xscale('log')
plt.legend()
plt.show()

# Select the best alpha based on minimum validation MSE
best_alpha_index = np.argmin(mse_valid_array)
best_alpha = alphas[best_alpha_index]
print('The best alpha:', best_alpha)


# In[24]:


# Define transformers
degree = 2  # Degree of polynomial features
poly = PolynomialFeatures(degree=degree)
scaler = StandardScaler()
onehot = OneHotEncoder(handle_unknown='ignore')

# Pipeline for numerical features: polynomial transformation and scaling
numerical_transformer = Pipeline(steps=[
    ('poly', poly),
    ('scale', scaler)
])

# Pipeline for categorical features: one-hot encoding
categorical_transformer = Pipeline(steps=[
    ('onehot', onehot)
])

# Combine numerical and categorical transformations
preprocessor = ColumnTransformer(transformers=[
    ('num', numerical_transformer, num_without_target),
    ('cat', categorical_transformer, categorical_final)
])

# Define a pipeline to apply preprocessing and then the Ridge regression model
model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', Ridge(alpha=best_alpha))  # Use the best alpha found earlier
])

# Fit the model to the training data
model_pipeline.fit(X_train, y_train)

# Make predictions on the test set
y_test_pred = model_pipeline.predict(X_test)

# Calculate and print MSE and R² score on the test set
mse_test = mean_squared_error(y_test, y_test_pred)
r2_test = r2_score(y_test, y_test_pred)

print(f'Test set Mean Squared Error (MSE): {mse_test}')
print(f'Test set R² score: {r2_test}')


# In[25]:


# Make predictions on the test set
y_test_pred = model_pipeline.predict(X_test)

# Calculate and print MSE and R² score on the test set
mse_test = mean_squared_error(y_test, y_test_pred)
r2_test = r2_score(y_test, y_test_pred)

print(f'Test set Mean Squared Error (MSE): {mse_test}')
print(f'Test set R² score: {r2_test}')


# ### Lasso Regression

# In[ ]:


# Generate alpha values from 10^(-10) to 10^(10)
alpha_indices = np.arange(20)
alphas = 10.0 ** (alpha_indices - 10)

# Degree for polynomial transformation
degree = 2

# Arrays to store the training and validation MSE
mse_train_array = np.full(len(alphas), np.nan)
mse_valid_array = np.full(len(alphas), np.nan)

for alpha_index, alpha in enumerate(alphas):
    # Define the transformers
    poly = PolynomialFeatures(degree=degree)
    scaler = StandardScaler()
    onehot = OneHotEncoder(handle_unknown='ignore')

    # Apply PolynomialFeatures and Scaling only to numerical features
    numerical_transformer = Pipeline(steps=[
        ('poly', poly),
        ('scale', scaler)
    ])

    # Apply OneHotEncoder to categorical features
    categorical_transformer = Pipeline(steps=[
        ('onehot', onehot)
    ])

    # Combine numerical and categorical transformations
    preprocessor = ColumnTransformer(transformers=[
        ('num', numerical_transformer, num_without_target),
        ('cat', categorical_transformer, categorical_final)
    ])

    # Create a pipeline with preprocessing and Lasso regression with current alpha
    model_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', Lasso(alpha=alpha, max_iter=10000))
    ])

    # Fit the model
    model_pipeline.fit(X_train, y_train)

    # Predict and calculate MSE for training and validation sets
    y_pred_train = model_pipeline.predict(X_train)
    y_pred_valid = model_pipeline.predict(X_valid)

    mse_train = mean_squared_error(y_train, y_pred_train)
    mse_valid = mean_squared_error(y_valid, y_pred_valid)

    # Store the MSE values for this alpha
    mse_train_array[alpha_index] = mse_train
    mse_valid_array[alpha_index] = mse_valid
    print(f'alpha: {alpha}, Validation mean squared error: {mse_valid}.')

# Plot MSE against alphas for both training and validation sets
plt.plot(alphas, mse_train_array, label='MSE on training dataset')
plt.plot(alphas, mse_valid_array, label='MSE on validation dataset')
plt.xlabel(r'Regularization weights $\alpha$')
plt.ylabel('Mean squared error')
plt.title(r'Regularization weights $\alpha$ and mean squared error')
plt.xscale('log')
plt.legend()
plt.show()

# Select the best alpha based on minimum validation MSE
best_alpha_index = np.argmin(mse_valid_array)
best_alpha = alphas[best_alpha_index]
print('The best alpha:', best_alpha)


# In[ ]:


# Define transformers
degree = 2  # Degree of polynomial features
poly = PolynomialFeatures(degree=degree)
scaler = StandardScaler()
onehot = OneHotEncoder(handle_unknown='ignore')

# Pipeline for numerical features: polynomial transformation and scaling
numerical_transformer = Pipeline(steps=[
    ('poly', poly),
    ('scale', scaler)
])

# Pipeline for categorical features: one-hot encoding
categorical_transformer = Pipeline(steps=[
    ('onehot', onehot)
])

# Combine numerical and categorical transformations
preprocessor = ColumnTransformer(transformers=[
    ('num', numerical_transformer, num_without_target),
    ('cat', categorical_transformer, categorical_final)
])

# Define a pipeline to apply preprocessing and then the Lasso regression model
model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', Lasso(alpha=best_alpha, max_iter=10000))  # Use the best alpha found earlier
])

# Fit the model to the training data
model_pipeline.fit(X_train, y_train)

# Make predictions on the test set
y_test_pred = model_pipeline.predict(X_test)

# Calculate and print MSE and R² score on the test set
mse_test = mean_squared_error(y_test, y_test_pred)
r2_test = r2_score(y_test, y_test_pred)

print(f'Test set Mean Squared Error (MSE): {mse_test}')
print(f'Test set R² score: {r2_test}')


# ### Random Forest

# In[88]:


# Separate features and target variable
X = df.drop(columns=['Lifespan','microstructure', 'seedLocation'])
y = df['Lifespan']

# Split the data into training, validation, and test sets
X_train_valid, X_test, y_train_valid, y_test = train_test_split(X, y, test_size=0.10, shuffle=True, random_state=0)
X_train, X_valid, y_train, y_valid = train_test_split(X_train_valid, y_train_valid, test_size=0.12, shuffle=True, random_state=0)

# Define numerical and categorical feature columns
num_without_target = X.select_dtypes(include=['float64', 'int64']).columns.tolist()
categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()

# Define the transformers
scaler = StandardScaler()
onehot = OneHotEncoder(handle_unknown='ignore')

# Apply Scaling to numerical features
numerical_transformer = Pipeline(steps=[
    ('scale', scaler)
])

# Apply OneHotEncoder to categorical features
categorical_transformer = Pipeline(steps=[
    ('onehot', onehot)
])

# Combine numerical and categorical transformations using ColumnTransformer
preprocessor = ColumnTransformer(transformers=[
    ('num', numerical_transformer, num_without_target),
    ('cat', categorical_transformer, categorical_final)
])

scal_encod_pipeline = Pipeline(steps=[
    ('preprocessor',preprocessor)
])

X_train=scal_encod_pipeline.fit_transform(X_train)
X_valid=scal_encod_pipeline.fit_transform(X_valid)
X_test=scal_encod_pipeline.fit_transform(X_test)


# In[89]:


# Define updated hyperparameter ranges
n_estimators_range = [50, 100, 200, 300]
max_depth_range = [None, 10, 20, 30]
max_features_range = [None, 'sqrt', 'log2']  # Replace 'auto' with None for compatibility

# Initialize lists to store results
mse_train_list = []
mse_valid_list = []
R2_train_list = []
R2_valid_list = []

# Loop through all combinations of hyperparameters
for n_estimators in n_estimators_range:
    for max_depth in max_depth_range:
        for max_features in max_features_range:
            print(f"n_estimators: {n_estimators}, max_depth: {max_depth}, max_features: {max_features}")

            # Create a pipeline that applies preprocessing and then the model
            model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, max_features=max_features, random_state=0)

            # Fit the model on the training set
            model.fit(X_train, y_train)

            # Make predictions and evaluate on training and validation sets
            y_pred_train = model.predict(X_train)
            y_pred_valid = model.predict(X_valid)

            mse_train = mean_squared_error(y_train, y_pred_train)
            mse_valid = mean_squared_error(y_valid, y_pred_valid)
            R2_train = r2_score(y_train, y_pred_train)
            R2_valid = r2_score(y_valid, y_pred_valid)

            mse_train_list.append((mse_train, n_estimators, max_depth, max_features))
            mse_valid_list.append((mse_valid, n_estimators, max_depth, max_features))
            R2_train_list.append(R2_train)
            R2_valid_list.append(R2_valid)

            # Print the results
            print("Training Set MSE:", mse_train)
            print("Training Set R2 Score:", R2_train)
            print("Validation Set MSE:", mse_valid)
            print("Validation Set R2 Score:", R2_valid)

# Identifying the best parameters (if needed)
best_params = min(mse_valid_list, key=lambda x: x[0])
best_n_estimators, best_max_depth, best_max_features = best_params[1], best_params[2], best_params[3]
print(f'\nThe best parameters: n_estimators={best_n_estimators}, max_depth={best_max_depth}, max_features={best_max_features}')


# In[90]:


# Final model training and test evaluation
# Retrain the model with the best parameters on the full training/validation set
final_model = RandomForestRegressor(n_estimators=best_n_estimators, max_depth=best_max_depth, max_features=best_max_features, random_state=0)
final_model.fit(X_train, y_train)

# Test set predictions and evaluation
y_pred_test = final_model.predict(X_test)
mse_test = mean_squared_error(y_test, y_pred_test)
R2_test = r2_score(y_test, y_pred_test)

print("Test Set MSE:", mse_test)
print("Test Set R2 Score:", R2_test)


# # Multi-Class Classification

# ###### Using clustering method to make categories on Lifesapn feature

# In[101]:


df_classification = df.copy()

from sklearn.cluster import KMeans


# Assuming 'lifespan' and other relevant features are in the dataframe
X = df_classification[['Lifespan','Nickel%','coolingRate', 'smallDefects','HeatTreatTime','Iron%']]

# Determine the optimal number of clusters using the Elbow Method
inertia = []
k_range = range(1,11)
for k in k_range:
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(X)
    inertia.append(kmeans.inertia_)

plt.plot(range(1, 11), inertia)
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
plt.title('Elbow Method')
plt.show()

# Choose an optimal k based on the elbow plot
optimal_k = 6
kmeans = KMeans(n_clusters=optimal_k)
df_classification['cluster'] = kmeans.fit_predict(X)


# In[102]:


inertia_summary = pd.DataFrame({'Number of Clusters': k_range, 'Inertia': inertia})

colors = plt.cm.rainbow(np.linspace(0, 1, len(inertia_summary)))

plt.figure(figsize=(12, 6))
bars = plt.bar(inertia_summary['Number of Clusters'], inertia_summary['Inertia'], color=colors)
plt.xticks(inertia_summary['Number of Clusters'])
plt.title('Inertia Values for Different Numbers of Clusters', fontsize=16)
plt.xlabel('Number of Clusters (k)', fontsize=14)
plt.ylabel('Inertia', fontsize=14)
plt.grid(axis='y')

for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval, round(yval, 2), ha='center', va='bottom', fontsize=10)

plt.show()


# In[103]:


# Assuming 'cluster' is the column with cluster labels
cluster_ranges = df_classification.groupby('cluster')['Lifespan'].agg(['min', 'max'])
print("Cluster ranges based on 'Lifespan':\n", cluster_ranges)


# In[104]:


# Reassign the cluster labels
df_classification['Target_hour'] = df_classification['cluster']


# In[105]:


# Group by 'Target_hour' and calculate min and max for 'Lifespan'
cluster_ranges = df_classification.groupby('Target_hour')['Lifespan'].agg(['min', 'max'])
print(cluster_ranges)


# In[106]:


print(df_classification['Target_hour'].value_counts())
print(df_classification['cluster'].value_counts())


# In[107]:


df_classification = df_classification.drop(columns=['cluster','Lifespan'])


# In[108]:


df_classification.head()


# ### Logistic Regression

# In[109]:


# Separate the features (X) and the target variable (y)
X = df_classification.drop(columns=['Target_hour','microstructure', 'seedLocation'])  # Drop 'Target_hour' to create the feature set
y = df_classification['Target_hour']  # Set 'Target_hour' as the target variable

# Split the data into training, validation, and test sets
X_train_valid, X_test, y_train_valid, y_test = train_test_split(X, y, test_size=0.10,  random_state=0)
X_train, X_valid, y_train, y_valid = train_test_split(X_train_valid, y_train_valid, test_size=0.11,  random_state=0)

# Identify numerical and categorical columns
num_without_target = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()

# Define transformers
scaler = StandardScaler()
onehot = OneHotEncoder(handle_unknown='ignore')

# Create separate transformers for numerical and categorical data
numerical_transformer = Pipeline(steps=[('scale', scaler)])
categorical_transformer = Pipeline(steps=[('onehot', onehot)])

# Combine transformations into a ColumnTransformer
preprocessor = ColumnTransformer(transformers=[
    ('num', numerical_transformer, num_without_target),
    ('cat', categorical_transformer, categorical_final)
])

# Apply transformations on X_train, X_valid, and X_test
X_train = preprocessor.fit_transform(X_train)
X_valid = preprocessor.transform(X_valid)
X_test = preprocessor.transform(X_test)

# Initialize and train the model
model = LogisticRegression()
model.fit(X_train, y_train)

# Predict on validation data
y_pred_valid = model.predict(X_valid)

# Evaluate the model performance on validation data
print('Validation Accuracy: {:.4f}'.format(accuracy_score(y_valid, y_pred_valid)))

# Confusion matrix
confusion_mat = confusion_matrix(y_valid, y_pred_valid, normalize='all')
print(f'Confusion matrix:\n', confusion_mat)

# Visualize the confusion matrix
ConfusionMatrixDisplay(confusion_mat).plot(cmap=plt.cm.Blues)
plt.grid(False)
plt.show()

# Print classification report for detailed metrics
print(classification_report(y_valid, y_pred_valid))


# ### Hyperparameter selection

# In[110]:


# Set up C values for regularization
C_indices = np.arange(20)
Cs = 10.0 ** (C_indices - 10)
f1_train_array = np.full([len(Cs)], np.nan)
f1_valid_array = np.full([len(Cs)], np.nan)

# Data Splitting
X = df_classification.drop(columns=['Target_hour','microstructure', 'seedLocation'])
y = df_classification['Target_hour']
X_train_valid, X_test, y_train_valid, y_test = train_test_split(X, y, test_size=0.10, random_state=0)
X_train, X_valid, y_train, y_valid = train_test_split(X_train_valid, y_train_valid, test_size=0.12, random_state=0)

# Iterate over C values
for C_index, C in zip(C_indices, Cs):
    model_pipeline = Pipeline(steps=[
        ('preprocessor', ColumnTransformer(
            transformers=[
                ('num', Pipeline([('scaler', StandardScaler())]), num_without_target),
                ('cat', Pipeline([('onehot', OneHotEncoder(handle_unknown='ignore'))]), categorical_final)
            ])),
        ('model', LogisticRegression(C=C, multi_class='multinomial', solver='lbfgs', max_iter=500))  # Set multi_class
    ])

    # Fit model on training data
    model_pipeline.fit(X_train, y_train)

    # Calculate and store F1 scores
    y_pred_train = model_pipeline.predict(X_train)
    y_pred_valid = model_pipeline.predict(X_valid)
    f1_train_array[C_index] = f1_score(y_train, y_pred_train, average='macro')
    f1_valid_array[C_index] = f1_score(y_valid, y_pred_valid, average='macro')
    print(f'C: {C}, F1 Score (Validation): {f1_valid_array[C_index]}')

# Plot F1 scores vs. regularization weights
plt.plot(Cs, f1_train_array, label='F1 Score (Train)')
plt.plot(Cs, f1_valid_array, label='F1 Score (Validation)')
plt.xscale('log')
plt.xlabel(r'Regularization weight $C$')
plt.ylabel('F1 Score')
plt.legend()
plt.title(r'Regularization Weight $C$ vs. F1 Score')
plt.show()

# Determine best C and re-fit model
best_C = Cs[np.nanargmax(f1_valid_array)]
print(f'Best C: {best_C}')

# Final Model Testing
model_pipeline = Pipeline(steps=[
    ('preprocessor', ColumnTransformer(
        transformers=[
            ('num', Pipeline([('scaler', StandardScaler())]), num_without_target),
            ('cat', Pipeline([('onehot', OneHotEncoder(handle_unknown='ignore'))]), categorical_final)
        ])),
    ('model', LogisticRegression(C=best_C, multi_class='multinomial', solver='lbfgs', max_iter=500))
])

model_pipeline.fit(X_train_valid, y_train_valid)
y_pred_test = model_pipeline.predict(X_test)

# Model Evaluation on Test Set
print('\nAccuracy: {:.4f}'.format(accuracy_score(y_test, y_pred_test)))
print('Precision (Macro): {:.4f}'.format(precision_score(y_test, y_pred_test, average="macro")))
print('Recall (Macro): {:.4f}'.format(recall_score(y_test, y_pred_test, average="macro")))
print('F1 Score (Macro): {:.4f}'.format(f1_score(y_test, y_pred_test, average="macro")))

# Balanced accuracy is useful for multi-class evaluation
print('Balanced Accuracy: {:.4f}'.format(balanced_accuracy_score(y_test, y_pred_test)))

# Confusion Matrix and Classification Report
confusion_mat = sklearn.metrics.confusion_matrix(y_test, y_pred_test, normalize='true')
sklearn.metrics.ConfusionMatrixDisplay(confusion_mat).plot(cmap=plt.cm.Blues)
plt.grid(False)
print(sklearn.metrics.classification_report(y_test, y_pred_test, digits=4))


# ## Artificial Neural Network Model

# In[114]:


# Separate the features (X) and the target variable (y)
X = df_classification.drop(columns=['Target_hour','microstructure', 'seedLocation'])
y = df_classification['Target_hour']

# Split the data into training, validation, and test sets
X_train_valid, X_test, y_train_valid, y_test = train_test_split(X, y, test_size=0.10, shuffle=True, random_state=0)

X_train, X_valid, y_train, y_valid = train_test_split(X_train_valid, y_train_valid, test_size=0.12, shuffle=True, random_state=0)

X_train=scal_encod_pipeline.fit_transform(X_train)
X_valid=scal_encod_pipeline.fit_transform(X_valid)
X_test=scal_encod_pipeline.fit_transform(X_test)

# Define the `Dense` layer.
dense_layer_1 = Dense(units=50, activation=relu)
dense_layer_2 = Dense(units=50, activation=relu)
out_layer = Dense(units=6, activation=softmax)

# Define the "virtual" input
input = Input(shape=X_train.shape[1:])

# Define the "virtual" output
output = dense_layer_1(input)
output = dense_layer_2(output)
output = out_layer(output)

# Define the neural network model.
model = Model(inputs=[input], outputs=[output], name='Multi_class_Classification')

# Output the summary of the model.
model.summary()

# COmpile the model
sgd = SGD(learning_rate=0.01)
ce = SparseCategoricalCrossentropy()
acc = SparseCategoricalAccuracy()
model.compile(optimizer=sgd, loss=ce, metrics=[acc])

# Train the model.
history = model.fit(X_train, y_train, batch_size=20, epochs=35, validation_data=(X_valid, y_valid))


# In[115]:


# Plot validation MSE, alwys nice to have plots to help us visualise things!
plt.plot(history.history['sparse_categorical_accuracy'], label='accuracy')
plt.plot(history.history['val_sparse_categorical_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.show()


# In[116]:


y_test_logit = model.predict(X_test)


# In[117]:


y_test_pred = np.argmax(y_test_logit, axis=1)
print(y_test_pred)


# In[118]:


disp = ConfusionMatrixDisplay(confusion_matrix(y_test, y_test_pred))
disp.plot(cmap=plt.cm.Blues)
plt.grid(False)
plt.show()

acc_test = accuracy_score(y_test, y_test_pred)
f1_test = f1_score(y_test, y_test_pred, average='macro')
print('The accuracy on the test data with the selected hyperparameter:', acc_test)
print('The F1 score on the test data with the selected hyperparameter:', f1_test)
pre_test = precision_score(y_test, y_test_pred, average='macro')
print('Precision on test data:', pre_test)
reca_test = recall_score(y_test, y_test_pred, average='macro')
print('Recall on test data:', reca_test)


# In[119]:


first_layer = Dense(units=128*4, activation=relu)
##############################
# Set proportion of nodes to deactivate with every training iteration (20% in this case)
second_layer = Dropout(0.2)
##############################
third_layer = Dense(units=128*4, activation=relu)
forth_layer = Dense(units=128*4, activation=relu)
fifth_layer = Dense(units=128*4, activation=relu)
out_layer = Dense(units=6, activation=softmax)

input = Input(shape=X_train.shape[1:])
output = first_layer(input)
output = second_layer(output)
output = third_layer(output)
output = forth_layer(output)
output = fifth_layer(output)
output = out_layer(output)

# Define the neural network model.
model = Model(inputs=[input], outputs=[output], name='Dropout')
model.summary()


# In[120]:


from tensorflow.keras.optimizers import Adam
# COmpile the model
adam = Adam(learning_rate=0.01)
ce = SparseCategoricalCrossentropy()
acc = SparseCategoricalAccuracy()
model.compile(optimizer=adam, loss=ce, metrics=[acc])

# Train the model.
history = model.fit(X_train, y_train, batch_size=20, epochs=35, validation_data=(X_valid, y_valid))


# In[121]:


# Plot validation MSE, alwys nice to have plots to help us visualise things!
plt.plot(history.history['sparse_categorical_accuracy'], label='accuracy')
plt.plot(history.history['val_sparse_categorical_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.show()


# In[122]:


y_test_logit = model.predict(X_test)
y_test_pred = np.argmax(y_test_logit, axis=1)


# In[123]:


disp = ConfusionMatrixDisplay(confusion_matrix(y_test, y_test_pred))
disp.plot(cmap=plt.cm.Blues)
plt.grid(False)
plt.show()

acc_test = accuracy_score(y_test, y_test_pred)
f1_test = f1_score(y_test, y_test_pred, average='macro')
print('The accuracy on the test data with the selected hyperparameter:', acc_test)
print('The F1 score on the test data with the selected hyperparameter:', f1_test)
pre_test = precision_score(y_test, y_test_pred, average='macro')
print('Precision on test data:', pre_test)
reca_test = recall_score(y_test, y_test_pred, average='macro')
print('Recall on test data:', reca_test)


# In[123]:




