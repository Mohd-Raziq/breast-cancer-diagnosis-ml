# Breast Cancer Diagnosis: Machine Learning Pipeline

An end-to-end classification project that predicts whether a breast tumour is **malignant or benign** from 30 cell-nucleus measurements. It compares five models with cross-validation, tunes the best one, evaluates it on held-out data, and serves it through a Streamlit app.

> Educational project. It is not a medical tool and must not be used for diagnosis.

## Dataset
Wisconsin Diagnostic Breast Cancer dataset (bundled with scikit-learn): 569 samples, 30 numeric features, no missing values. Class balance is 212 malignant and 357 benign.

## Approach
1. **EDA:** class balance and a correlation heatmap of the features most related to the diagnosis.
2. **Split:** 80/20 stratified train/test split. The test set is not touched until the final evaluation.
3. **Model comparison:** Logistic Regression, SVM (RBF), KNN, Random Forest and Gradient Boosting, each in a pipeline with `StandardScaler`, scored with 5-fold stratified cross-validation on the training set only.
4. **Tuning:** `GridSearchCV` on the best model (ROC-AUC).
5. **Evaluation:** metrics, confusion matrix, ROC curve and permutation feature importance on the test set.

## Results

Cross-validation on the training set:

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.978 | 0.980 | 0.986 | 0.983 | 0.996 |
| SVM (RBF) | 0.969 | 0.973 | 0.979 | 0.976 | 0.996 |
| Gradient Boosting | 0.952 | 0.962 | 0.961 | 0.961 | 0.992 |
| Random Forest | 0.965 | 0.976 | 0.968 | 0.972 | 0.990 |
| KNN | 0.963 | 0.959 | 0.983 | 0.971 | 0.988 |

Held-out test set (114 samples), tuned Logistic Regression (`C=1`):

| Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|
| 0.983 | 0.986 | 0.986 | 0.986 | 0.995 |

A simple linear model beat the tree ensembles here. With 569 samples and strongly correlated features, extra model complexity did not help.

Figures are in `reports/figures/`: class balance, correlation heatmap, confusion matrix, ROC curve and feature importance.

## Run it
```bash
pip install -r requirements.txt
python src/train.py        # trains, evaluates, saves model and reports
streamlit run app.py       # opens the prediction app
```

## Structure
```
app.py                  Streamlit prediction app
src/train.py            training and evaluation pipeline
models/model.joblib     saved model (created by train.py)
reports/                metrics, model comparison table, figures
```

## Limitations
- Small dataset from a single source, so results may not transfer to other hospitals or imaging setups.
- The class is mildly imbalanced. In a real screening setting, recall on malignant cases would matter most.
- Predictions come from measurements that were already extracted from images. The model does not read images.

## Possible next steps
- Add SHAP explanations for individual predictions.
- Calibrate predicted probabilities and choose a decision threshold that favours malignant recall.
- Compare against a small neural network.
