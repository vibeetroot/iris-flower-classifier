# app.py — streamlit interface for the iris classifier
#
# prerequisites: run train_model.py first to generate the .pkl files
#   python train_model.py
#   streamlit run app.py

import pickle
import numpy as np
import streamlit as st

# ── load artifacts ────────────────────────────────────────────────────────────
# @st.cache_resource: loads once and caches — avoids reloading pkl on every rerun
# use this for models, scalers, database connections — anything expensive to load

@st.cache_resource
def load_artifacts():
    with open("model.pkl",    "rb") as f: model    = pickle.load(f)
    with open("scaler.pkl",   "rb") as f: scaler   = pickle.load(f)
    with open("metadata.pkl", "rb") as f: metadata = pickle.load(f)
    return model, scaler, metadata

model, scaler, metadata = load_artifacts()

feature_names  = metadata["feature_names"]
class_names    = metadata["class_names"]
feature_ranges = metadata["feature_ranges"]
test_accuracy  = metadata["test_accuracy"]

# ── page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="iris classifier",
    page_icon="🌸",
    layout="centered"
)

# ── header ────────────────────────────────────────────────────────────────────

st.title("🌸 iris flower classifier")
st.markdown(
    "enter the flower measurements using the sliders below. "
    "the model will predict the species and show confidence for each class."
)
st.caption(f"model: random forest | test accuracy: **{test_accuracy:.1%}**")

st.divider()

# ── sidebar — model info ──────────────────────────────────────────────────────

st.sidebar.header("about the model")
st.sidebar.markdown(f"""
- **algorithm** : random forest classifier  
- **features**  : 4 (sepal + petal dimensions)  
- **classes**   : {', '.join(class_names)}  
- **test accuracy** : {test_accuracy:.1%}
""")

st.sidebar.markdown("---")
st.sidebar.markdown("**how it works**")
st.sidebar.markdown(
    "1. your inputs are scaled using the same `StandardScaler` fitted on training data.\n"
    "2. the scaled values are passed to the trained `RandomForestClassifier`.\n"
    "3. the model returns a predicted class and probability for each class."
)

# ── input widgets ─────────────────────────────────────────────────────────────

st.subheader("flower measurements")

col1, col2 = st.columns(2)
inputs = {}

feature_list = list(feature_ranges.items())

# first two features in left column, last two in right column
with col1:
    for name, (min_val, max_val, default) in feature_list[:2]:
        inputs[name] = st.slider(
            name,
            min_value=float(min_val),
            max_value=float(max_val),
            value=float(default),
            step=0.1
        )

with col2:
    for name, (min_val, max_val, default) in feature_list[2:]:
        inputs[name] = st.slider(
            name,
            min_value=float(min_val),
            max_value=float(max_val),
            value=float(default),
            step=0.1
        )

# ── prediction ────────────────────────────────────────────────────────────────

st.divider()
st.subheader("prediction")

if st.button("🔍 predict species", use_container_width=True):

    # build input array in the same feature order as training
    input_array  = np.array([[inputs[name] for name in feature_names]])

    # apply the same scaler used during training — critical step
    input_scaled = scaler.transform(input_array)

    # get prediction and probabilities
    pred_index   = model.predict(input_scaled)[0]
    probabilities = model.predict_proba(input_scaled)[0]

    predicted_species = class_names[pred_index]
    confidence        = probabilities[pred_index]

    # display result
    species_emojis = {"setosa": "🌺", "versicolor": "🌷", "virginica": "🌸"}
    emoji = species_emojis.get(predicted_species, "🌼")

    st.success(f"{emoji}  predicted species: **iris {predicted_species}**")

    m1, m2 = st.columns(2)
    m1.metric("confidence", f"{confidence:.1%}")
    m2.metric("predicted class index", pred_index)

    # probability bar chart — one bar per class
    st.markdown("**prediction probabilities across all classes:**")
    prob_dict = {name: float(p) for name, p in zip(class_names, probabilities)}
    st.bar_chart(prob_dict, height=200)

    # show the actual scaled input — educational: helps students see what the model receives
    with st.expander("🔬 see what the model actually received (scaled input)"):
        st.write("raw input (what you entered):")
        st.write({name: round(val, 1) for name, val in inputs.items()})
        st.write("scaled input (after applying StandardScaler):")
        st.write({name: round(float(val), 4) for name, val in zip(feature_names, input_scaled[0])})

else:
    st.info("adjust the sliders above and click **predict species** to see the result.")