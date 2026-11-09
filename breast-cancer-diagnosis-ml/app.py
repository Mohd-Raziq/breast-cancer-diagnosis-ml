"""Streamlit app: enter tumour measurements and get a model prediction.

Run:  streamlit run app.py   (after running  python src/train.py  once)
"""
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Breast cancer diagnosis model", layout="centered")

MODEL_PATH = Path(__file__).parent / "models" / "model.joblib"
if not MODEL_PATH.exists():
    st.error("Model file not found. Run `python src/train.py` first.")
    st.stop()


@st.cache_resource
def load():
    return joblib.load(MODEL_PATH)


bundle = load()
model, features = bundle["model"], bundle["features"]

st.title("Breast cancer diagnosis model")
st.caption("Educational project built on the Wisconsin Diagnostic Breast Cancer "
           "dataset. It is not a medical tool and must not be used for diagnosis.")

st.write("Each measurement starts at the dataset median. Change the values and "
         "the prediction updates.")

groups = {"Mean values": "mean ", "Error values": " error", "Worst values": "worst "}
tabs = st.tabs(list(groups))
values = {}
for tab, (label, key) in zip(tabs, groups.items()):
    with tab:
        cols = st.columns(2)
        names = [f for f in features
                 if (f.startswith("mean ") if key == "mean " else
                     f.endswith(" error") if key == " error" else
                     f.startswith("worst "))]
        for i, name in enumerate(names):
            lo, hi = float(bundle["mins"][name]), float(bundle["maxs"][name])
            values[name] = cols[i % 2].slider(
                name, lo, hi, float(bundle["medians"][name]),
                step=(hi - lo) / 200, format="%.4f", key=name)

row = pd.DataFrame([values])[features]
prob_benign = float(model.predict_proba(row)[0, 1])

st.divider()
if prob_benign >= 0.5:
    st.success(f"Predicted: benign ({prob_benign:.1%} confidence)")
else:
    st.error(f"Predicted: malignant ({1 - prob_benign:.1%} confidence)")
st.progress(prob_benign, text=f"Probability of benign: {prob_benign:.1%}")
