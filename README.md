# Apple Product Price Prediction

## Project Overview

This project builds machine learning regression models to predict the **current price of Apple products in INR** using product, platform, pricing, discount, condition, stock, rating, review, and time-related information.

The dataset contains **80,000 records and 14 original features** covering Apple products listed on e-commerce platforms such as Amazon and Flipkart. The project includes exploratory data analysis, data preprocessing, feature engineering, model comparison, cross-validation, hyperparameter tuning, and feature importance analysis.

## Dataset Features

The original dataset contains the following features:

- **Date** – Date of the product listing/observation
- **Platform** – E-commerce platform such as Amazon or Flipkart
- **Product_Category** – Apple product category such as iPhone, iPad, Mac, or Watch
- **Model_Name** – Specific Apple product model
- **Condition** – Product condition such as New, Refurbished, or Renewed
- **Launch_Price_USD** – Original launch price in USD
- **Launch_Price_INR** – Original launch price in INR
- **Current_Price_USD** – Current selling price in USD
- **Current_Price_INR** – Current selling price in INR
- **Discount_Pct** – Discount percentage from launch price
- **Sale_Event** – Sale event information
- **Stock_Status** – Product availability status
- **Rating** – Customer rating
- **Reviews_Count** – Number of customer reviews

## Project Workflow

### 1. Exploratory Data Analysis

Performed univariate, bivariate, and multivariate analysis to understand the distribution and relationships among the variables.

The analysis included:

- Numerical and categorical feature analysis
- Distribution analysis
- Correlation analysis using a heatmap
- Missing-value analysis
- Duplicate checking
- Outlier detection using the IQR method

### 2. Data Preprocessing & Feature Engineering

The preprocessing pipeline included:

- Removing unnecessary/redundant price features
- Extracting useful information from the date feature
- Applying a log transformation to `Current_Price_INR` where required
- One-hot encoding categorical variables
- Preparing training and testing datasets

### 3. Machine Learning Models

The following regression algorithms were trained and evaluated:

- Linear Regression
- Decision Tree Regressor
- Random Forest Regressor
- XGBoost Regressor

Model performance was evaluated using:

- **R² Score**
- **Mean Absolute Error (MAE)**
- **Root Mean Squared Error (RMSE)**

### 4. Cross-Validation

Five-fold K-Fold cross-validation was performed to evaluate model stability.

| Model | Mean CV R² | CV Std. Dev. |
|---|---:|---:|
| Random Forest | 0.8655 | 0.0017 |
| XGBoost | 0.8675 | 0.0018 |

The low standard deviation indicates consistent performance across the five folds.

### 5. XGBoost Hyperparameter Tuning

`RandomizedSearchCV` with 5-fold cross-validation was used to optimize the XGBoost model.

The best parameters obtained were:

```python
{
    "subsample": 0.9,
    "n_estimators": 300,
    "min_child_weight": 3,
    "max_depth": 6,
    "learning_rate": 0.05,
    "colsample_bytree": 0.8
}
```

### 6. Tuned XGBoost Performance

| Metric | Result |
|---|---:|
| Training R² | 0.8836 |
| Testing R² | **0.8699** |
| Cross-Validation R² | 0.8675 |
| MAE | ₹11,198.44 |
| RMSE | ₹16,307.56 |

The tuned XGBoost model achieved a **testing R² of approximately 0.87**, explaining about 87% of the variation in Apple product prices on the test data.

### 7. Untuned vs Tuned XGBoost

| Model | Training R² | Testing R² | CV R² | MAE | RMSE |
|---|---:|---:|---:|---:|---:|
| Untuned XGBoost | 0.8987 | 0.8680 | 0.8675 | ₹10,951.32 | ₹16,425.20 |
| Tuned XGBoost | 0.8836 | **0.8699** | **0.8675** | ₹11,198.44 | **₹16,307.56** |

Hyperparameter tuning slightly improved the testing R² and RMSE while reducing the gap between training and testing performance.

## Feature Importance

Feature importance from the tuned XGBoost model was analyzed to understand which variables contributed most to the price predictions.

This step helps explain the model and identify the product and pricing characteristics that have the greatest influence on predicted Apple product prices.

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- XGBoost
- Jupyter Notebook

## Key Concepts Demonstrated

- Exploratory Data Analysis (EDA)
- Data Cleaning
- Outlier Detection
- Feature Engineering
- Log Transformation
- One-Hot Encoding
- Regression Modeling
- Model Evaluation
- K-Fold Cross-Validation
- Hyperparameter Tuning
- Feature Importance Analysis

## Conclusion

Multiple regression algorithms were compared for Apple product price prediction. Random Forest and XGBoost demonstrated strong and stable cross-validation performance. After hyperparameter tuning with RandomizedSearchCV, the XGBoost model achieved a **testing R² of 0.8699** and **RMSE of approximately ₹16,308**.

The project demonstrates an end-to-end machine learning regression workflow, from exploratory analysis and preprocessing through model evaluation, validation, tuning, and interpretation.
