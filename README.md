# Alphabet Soup Charity Deep Learning Model

## Overview

The purpose of this project was to develop a binary classification model using deep learning to predict whether an Alphabet Soup-funded organization would be successful. Using a dataset of historical funding applications, we designed and optimized a neural network model using TensorFlow and Keras to achieve an accuracy goal of over 75%.

---

## Data Preprocessing

- **Target Variable**: `IS_SUCCESSFUL`
  - Binary value indicating whether the funded organization was successful (`1`) or not (`0`).

- **Features**: All remaining columns after dropping unnecessary identifiers.

- **Removed Columns**:
  - `EIN` – Employer Identification Number, which is not useful for prediction.
  - `NAME` – Name of the organization, which doesn't contribute to the outcome.

- **Categorical Data Handling**:
  - Grouped rare values in `APPLICATION_TYPE` (fewer than 500) and `CLASSIFICATION` (fewer than 1000) into a new category `"Other"` to reduce dimensionality.
  - Encoded all categorical variables using `pd.get_dummies()`.

- **Train/Test Split**:
  - 75% training, 25% testing using `train_test_split`.

- **Feature Scaling**:
  - Scaled numerical features using `StandardScaler` from scikit-learn.

---

## Model Design and Optimization

### Initial Architecture:
- **Hidden Layers**: 2
- **Neurons**: 80 and 30
- **Activation Functions**: ReLU (hidden layers), Sigmoid (output)
- **Result**: ~72% accuracy (below target)

### Optimized Final Architecture:
- **Input Layer**: Based on total features (number of input neurons = feature columns)
- **Hidden Layers**:
  - Dense (128 units, ReLU)
  - Dense (64 units, ReLU)

- **Output Layer**: Dense (1 unit, Sigmoid)
- **Optimizer**: Adam
- **Loss Function**: Binary Crossentropy
- **Epochs**: 100
- **Final Accuracy**: 72.64% on test data

### Saved Model:
- Final model saved to `AlphabetSoupCharity_Optimization.h5`

---

## Summary and Recommendations

The optimized neural network achieved the accuracy of over 72.64%, outperforming the initial model. Preprocessing steps such as encoding, grouping rare values, and feature scaling were essential for model performance.

### Recommendations for Further Improvement:
- Use **Keras Tuner** to automate hyperparameter tuning (neurons, layers, learning rate).
- Try other model architectures like **Random Forest** or **Gradient Boosting** for comparison.


---

## Files in this Repository

- `AlphabetSoupCharity.ipynb`: Initial preprocessing and model design.
- `AlphabetSoupCharity_Optimization.ipynb`: Optimized model notebook.
- `AlphabetSoupCharity.h5`: Initial saved model.
- `AlphabetSoupCharity_Optimization.h5`: Final optimized model.
- `README.md`: This report.
