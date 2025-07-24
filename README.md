Heart Disease Prediction

This project uses Logistic Regression to predict whether a person has heart disease based on medical features.

-> Dataset
- The dataset used is a variant of the UCI Heart Disease dataset.
- The target variable `num` is binarized:  
  - `0` → No heart disease  
  - `1-4` → Has heart disease

-> How It Works
- Missing values are dropped for simplicity.
- Categorical columns are label-encoded.
- The model is trained on 80% of the data and tested on 20%.
- Accuracy for both training and test data is shown.
