
import json
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, RocCurveDisplay,
                             accuracy_score, classification_report, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import (GridSearchCV, StratifiedKFold,
                                     cross_validate, train_test_split)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

SEED = 42
ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "reports" / "figures"
FIG.mkdir(parents=True, exist_ok=True)
(ROOT / "models").mkdir(exist_ok=True)

# ---------------------------------------------------------------- data
data = load_breast_cancer(as_frame=True)
X, y = data.data, data.target          # target: 0 = malignant, 1 = benign
class_names = list(data.target_names)  # ['malignant', 'benign']
print(f"Samples: {X.shape[0]}, features: {X.shape[1]}")
print("Missing values:", int(X.isna().sum().sum()))

# ---------------------------------------------------------------- EDA
plt.figure(figsize=(4.5, 3.5))
sns.countplot(x=y.map(dict(enumerate(class_names))), color="#3b7ea1")
plt.title("Class balance")
plt.xlabel("")
plt.tight_layout()
plt.savefig(FIG / "class_balance.png", dpi=150)
plt.close()

top = X.corrwith(y).abs().sort_values(ascending=False).head(10).index
plt.figure(figsize=(8, 6.5))
sns.heatmap(X[top].corr(), cmap="coolwarm", center=0, annot=True,
            fmt=".2f", annot_kws={"size": 7})
plt.title("Correlation of the 10 features most related to the diagnosis")
plt.tight_layout()
plt.savefig(FIG / "correlation_heatmap.png", dpi=150)
plt.close()

# ---------------------------------------------------------------- split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=SEED)

# ---------------------------------------------------------------- compare models
def pipe(model):
    return Pipeline([("scale", StandardScaler()), ("model", model)])

candidates = {
    "Logistic Regression": pipe(LogisticRegression(max_iter=5000, random_state=SEED)),
    "SVM (RBF)": pipe(SVC(probability=True, random_state=SEED)),
    "KNN": pipe(KNeighborsClassifier()),
    "Random Forest": pipe(RandomForestClassifier(n_estimators=300, random_state=SEED)),
    "Gradient Boosting": pipe(GradientBoostingClassifier(random_state=SEED)),
}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
scoring = ["accuracy", "precision", "recall", "f1", "roc_auc"]

rows = []
for name, model in candidates.items():
    res = cross_validate(model, X_train, y_train, cv=cv, scoring=scoring)
    rows.append({"model": name, **{m: res[f"test_{m}"].mean() for m in scoring}})
comparison = pd.DataFrame(rows).sort_values("roc_auc", ascending=False).round(4)
comparison.to_csv(ROOT / "reports" / "model_comparison.csv", index=False)
print("\n5-fold CV on the training set:\n", comparison.to_string(index=False))

# ---------------------------------------------------------------- tune best
grids = {
    "Logistic Regression": {"model__C": [0.01, 0.1, 1, 10, 100]},
    "SVM (RBF)": {"model__C": [0.1, 1, 10], "model__gamma": ["scale", 0.01, 0.001]},
    "KNN": {"model__n_neighbors": [3, 5, 7, 11]},
    "Random Forest": {"model__max_depth": [None, 5, 10], "model__min_samples_leaf": [1, 2, 4]},
    "Gradient Boosting": {"model__learning_rate": [0.05, 0.1], "model__max_depth": [2, 3]},
}
best_name = comparison.iloc[0]["model"]
search = GridSearchCV(candidates[best_name], grids[best_name],
                      cv=cv, scoring="roc_auc", n_jobs=-1)
search.fit(X_train, y_train)
best = search.best_estimator_
print(f"\nTuned model: {best_name}  params: {search.best_params_}")

# ---------------------------------------------------------------- test evaluation
pred = best.predict(X_test)
proba = best.predict_proba(X_test)[:, 1]
metrics = {
    "model": best_name,
    "best_params": {k: (v if isinstance(v, (int, float, str, type(None))) else str(v))
                    for k, v in search.best_params_.items()},
    "cv_roc_auc": round(search.best_score_, 4),
    "test_accuracy": round(accuracy_score(y_test, pred), 4),
    "test_precision": round(precision_score(y_test, pred), 4),
    "test_recall": round(recall_score(y_test, pred), 4),
    "test_f1": round(f1_score(y_test, pred), 4),
    "test_roc_auc": round(roc_auc_score(y_test, proba), 4),
    "test_size": int(len(y_test)),
}
(ROOT / "reports" / "metrics.json").write_text(json.dumps(metrics, indent=2))
print("\nHeld-out test set:\n", classification_report(y_test, pred, target_names=class_names))

fig, ax = plt.subplots(figsize=(4.5, 4))
ConfusionMatrixDisplay.from_predictions(y_test, pred, display_labels=class_names,
                                        cmap="Blues", ax=ax, colorbar=False)
ax.set_title(f"Confusion matrix: {best_name}")
plt.tight_layout()
plt.savefig(FIG / "confusion_matrix.png", dpi=150)
plt.close()

fig, ax = plt.subplots(figsize=(4.8, 4))
RocCurveDisplay.from_predictions(y_test, proba, ax=ax, name=best_name)
ax.plot([0, 1], [0, 1], "k--", lw=1)
ax.set_title("ROC curve (test set)")
plt.tight_layout()
plt.savefig(FIG / "roc_curve.png", dpi=150)
plt.close()

imp = permutation_importance(best, X_test, y_test, n_repeats=20,
                             scoring="roc_auc", random_state=SEED)
imp_df = (pd.Series(imp.importances_mean, index=X.columns)
          .sort_values(ascending=False).head(10)[::-1])
plt.figure(figsize=(7, 4.5))
imp_df.plot.barh(color="#3b7ea1")
plt.xlabel("Drop in ROC-AUC when the feature is shuffled")
plt.title("Top 10 features (permutation importance)")
plt.tight_layout()
plt.savefig(FIG / "feature_importance.png", dpi=150)
plt.close()

# ---------------------------------------------------------------- save
joblib.dump({"model": best, "features": list(X.columns),
             "medians": X.median().to_dict(),
             "mins": X.min().to_dict(), "maxs": X.max().to_dict()},
            ROOT / "models" / "model.joblib")
print("\nSaved model and reports.")
