
import json
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import streamlit as st

ROOT = Path(__file__).resolve().parent

st.set_page_config(
    page_title="ORBITAL | Earth Observation",
    page_icon="🛰️",
    layout="wide"
)

st.title("🛰️ ORBITAL")
st.subheader("Reinforcement Learning for Information-Efficient Earth Observation")
st.write(
    "A research prototype combining multispectral satellite imagery, "
    "spectral clustering, and tabular reinforcement learning."
)

st.warning(
    "EuroSAT imagery is real. Weather regimes, cloud risks, energy costs, "
    "and observation transitions in the RL environment are simulated. "
    "This is not a physical orbital simulator."
)


@st.cache_resource
def load_models():
    with open(ROOT / "orbital_models.pkl", "rb") as f:
        models = pickle.load(f)

    metadata_path = ROOT / "metadata.json"
    metadata = (
        json.loads(metadata_path.read_text())
        if metadata_path.exists() else {}
    )
    return models, metadata


try:
    models, metadata = load_models()
    q_table = models["q_learning"]
    sarsa_table = models["sarsa"]
    scaler = models["scaler"]
    cluster_model = models["region_model"]
    centers = np.asarray(models["target_features"])
    distances = np.asarray(models["distance_matrix"])
except Exception as exc:
    st.error(f"Unable to load saved models: {exc}")
    st.stop()


def spectral_features(image):
    image = np.asarray(image, dtype=np.float32)

    band_means = image.mean(axis=(0, 1))
    green = image[:, :, 2]
    red = image[:, :, 3]
    nir = image[:, :, 7]
    swir = image[:, :, 11]
    eps = 1e-8

    ndvi = (nir - red) / (nir + red + eps)
    ndwi = (green - nir) / (green + nir + eps)
    ndbi = (swir - nir) / (swir + nir + eps)

    return np.concatenate([
        band_means,
        [
            ndvi.mean(), ndvi.std(),
            ndwi.mean(), ndwi.std(),
            ndbi.mean(), ndbi.std()
        ]
    ])


@st.cache_data(ttl=3600)
def fetch_eurosat_samples():
    """Fetch a small set of patches from the public dataset viewer."""
    api = "https://datasets-server.huggingface.co/rows"
    dataset_name = "blanchon/EuroSAT_MSI"

    # Spread requests across the dataset rather than downloading it all.
    offsets = [0, 1600, 3200, 4800, 6400, 8000, 9600, 11200]
    samples = []

    for offset in offsets:
        response = requests.get(
            api,
            params={
                "dataset": dataset_name,
                "config": "default",
                "split": "train",
                "offset": offset,
                "length": 1,
            },
            timeout=45,
        )
        response.raise_for_status()
        rows = response.json().get("rows", [])

        for item in rows:
            row = item.get("row", {})
            image = row.get("image")

            if isinstance(image, dict):
                image = image.get("array", image.get("data"))

            if image is None:
                continue

            image = np.asarray(image)
            if image.shape == (64, 64, 13):
                samples.append({
                    "image": image,
                    "label": str(row.get("label", "Unknown")),
                    "filename": str(row.get("filename", "Unknown")),
                })

    if not samples:
        raise ValueError("The dataset viewer returned no usable image arrays.")

    return samples


def stretch(band):
    band = np.asarray(band, dtype=float)
    lo, hi = np.percentile(band, [2, 98])
    if hi <= lo:
        return np.zeros_like(band)
    return np.clip((band - lo) / (hi - lo), 0, 1)


def composite(image, bands):
    return np.dstack([stretch(image[:, :, b]) for b in bands])


def spectral_index(image, a, b):
    a = image[:, :, a].astype(float)
    b = image[:, :, b].astype(float)
    return (a - b) / (a + b + 1e-8)


# Overview
c1, c2, c3, c4 = st.columns(4)
c1.metric("Spectral targets", len(distances))
c2.metric("Features per patch", scaler.n_features_in_)
c3.metric("Q-learning states", len(q_table))
c4.metric("SARSA states", len(sarsa_table))

st.caption(
    "The state counts describe saved Q-table entries, not the number "
    "of training episodes."
)

tab1, tab2, tab3, tab4 = st.tabs([
    "Satellite imagery",
    "Spectral analysis",
    "RL models",
    "Metadata"
])


with tab1:
    st.header("Live dataset retrieval")

    if st.button("Load real EuroSAT image patches"):
        try:
            st.session_state["orbital_samples"] = fetch_eurosat_samples()
        except Exception as exc:
            st.error(
                "Could not retrieve images from the public dataset viewer. "
                f"Please retry later. Details: {exc}"
            )

    samples = st.session_state.get("orbital_samples", [])

    if samples:
        selected = st.selectbox(
            "Select a satellite image",
            range(len(samples)),
            format_func=lambda i: samples[i]["filename"]
        )
        item = samples[selected]
        image = item["image"]

        try:
            features = spectral_features(image).reshape(1, -1)
            scaled = scaler.transform(features)
            target = int(cluster_model.predict(scaled)[0])

            st.success(f"Predicted spectral cluster: Target {target}")
            st.caption(
                "This is a cluster assignment from your saved model, "
                "not a verified geographic location."
            )
        except Exception as exc:
            st.error(f"Feature extraction or clustering failed: {exc}")
            target = None

        left, right = st.columns(2)
        with left:
            st.write("**True-colour composite (B4, B3, B2)**")
            st.image(
                composite(image, [3, 2, 1]),
                use_container_width=True
            )
        with right:
            st.write("**False-colour composite (B8, B4, B3)**")
            st.image(
                composite(image, [7, 3, 2]),
                use_container_width=True
            )

        fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
        for ax, pair, title in zip(
            axes,
            [(7, 3), (2, 7), (11, 7)],
            ["NDVI", "NDWI", "NDBI"]
        ):
            ax.imshow(
                spectral_index(image, *pair),
                cmap="RdYlGn",
                vmin=-1,
                vmax=1
            )
            ax.set_title(title)
            ax.axis("off")

        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        st.write("Source label:", item["label"])
        st.write("Source filename:", item["filename"])
    else:
        st.info(
            "Click the button to retrieve a small sample from the public "
            "EuroSAT dataset. No complete dataset download is required."
        )


with tab2:
    st.header("Spectral target analysis")

    target = st.selectbox(
        "Select a target cluster",
        range(len(centers)),
        key="target_select"
    )

    st.write(f"**Target {target} centroid**")
    st.line_chart(pd.DataFrame({
        "Standardized centroid": centers[target]
    }))

    st.write("**Pairwise target distance matrix**")
    st.dataframe(
        pd.DataFrame(
            distances,
            index=[f"Target {i}" for i in range(len(distances))],
            columns=[f"Target {i}" for i in range(len(distances))]
        ).round(3),
        use_container_width=True
    )

    st.caption(
        "Centroids use standardized features. Clusters represent spectral "
        "similarity and should not automatically be interpreted as land-cover classes."
    )


with tab3:
    st.header("Saved reinforcement learning agents")

    policy_name = st.radio(
        "Select saved agent",
        ["Q-learning", "SARSA"],
        horizontal=True
    )
    table = q_table if policy_name == "Q-learning" else sarsa_table

    st.metric("Saved state-action entries", len(table))
    st.write(
        f"The {policy_name} Q-table is loaded from your saved pickle file."
    )

    if table:
        states = list(table.keys())
        state_idx = st.number_input(
            "Inspect saved state index",
            min_value=0,
            max_value=len(states) - 1,
            value=0,
            step=1
        )
        state = states[int(state_idx)]
        values = np.asarray(table[state])

        st.write("**State representation**")
        st.code(repr(state))
        st.write("**Action values**")
        st.bar_chart(pd.DataFrame({
            "Action value": values
        }))
        st.write("Greedy action:", int(np.argmax(values)))
    else:
        st.info("This Q-table is empty.")


with tab4:
    st.header("Experiment metadata")
    st.json(metadata)

    st.write("**Saved model artifacts**")
    for key, value in models.items():
        if key in ("q_learning", "sarsa"):
            st.write(f"- {key}: {len(value)} states")
        elif hasattr(value, "shape"):
            st.write(f"- {key}: shape {value.shape}")
        else:
            st.write(f"- {key}: {type(value).__name__}")

    st.info(
        "Training curves and policy evaluation CSVs were not included in "
        "the uploaded repository files. This dashboard does not invent "
        "those results; add the original files from Colab to display them."
    )

st.divider()
st.caption(
    "ORBITAL research prototype | EuroSAT multispectral data | "
    "Tabular reinforcement learning"
)
