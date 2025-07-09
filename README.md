# Prediction-Emergency-Admission
“Math 10 final project – Predicting emergency disposition categories using logistic regression and neural networks”


## Project Structure

This project is organized into the following modular notebooks, each representing a major stage in the workflow:

| Notebook                             | Description                                                                 |
|--------------------------------------|-----------------------------------------------------------------------------|
| `1_preprocessing_scaling.ipynb`      | Loads and cleans the dataset, performs label encoding and feature scaling to ensure multicollinearity is reduced. |
| `2_logistic_regression.ipynb`        | Trains and evaluates logistic regression models (binary) using cross_validation. Performs Binary Cross Entropy for nuanced model behavior. Rank orders significance of predictive variables | 
| `3_random_forest.ipynb`              | Builds and compares a Random Forest classifier using the same features. Rank orders significance of predictive variables     |
| `4_discussion_insights.ipynb`        | Provides reflections on model performance, class behavior, and insights.   |

