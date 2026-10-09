
import json
import pickle
import time
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
    layout="wide",
)

st.title("🛰️ ORBITAL")
st.subheader("Reinforcement Learning for Information-Efficient Earth Observation")
st.write(
    "A research prototype combining real multispectral satellite imagery, "
    "unsupervised spectral clustering, and tabular reinforcement learning."
)

st.warning(
    "Scientific scope: EuroSAT imagery is real. Cloud risks, weather regimes, "
    "energy costs, and observation transitions are simulated. This is not "
    "a physical orbital simulator or a validated mission planner."
)


# ---------------------------------------------------------
# 1. LOAD YOUR SAVED MODELS
# ---------------------------------------------------------

@st.cache_resource
def load_models():
    model_path = ROOT / "orbital_models.pkl"
    metadata_path = ROOT / "metadata.json"

    if not model_path.exists():
        raise FileNotFoundError("orbital_models.pkl is missing from the repository.")

    # Only load pickle files from your own trusted repository.
    with open(model_path, "rb") as file:
        models = pickle.load(file)

    metadata = {}
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text())

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
    st.error(f"Could not load saved models: {exc}")
    st.info(
        "Check that orbital_models.pkl and metadata.json are in the "
        "repository root and that the scikit-learn version is compatible."
    )
    st.stop()


# ---------------------------------------------------------
# 2. FEATURE EXTRACTION
# ---------------------------------------------------------

def spectral_features(image):
    image = np.asarray(image, dtype=np.float32)

    if image.shape != (64, 64, 13):
        raise ValueError(
            f"Expected an image with shape (64, 64, 13), got {image.shape}"
        )

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
            ndbi.mean(), ndbi.std(),
        ],
    ])


# ---------------------------------------------------------
# 3. ROBUST ONLINE EURO SAT IMAGE RETRIEVAL
# ---------------------------------------------------------

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_eurosat_samples():
    """
    Try the Hugging Face dataset viewer with retries.
    If that fails, try loading the dataset with the datasets library.
    Only a small number of image patches are retained.
    """
    import requests

    dataset_name = "blanchon/EuroSAT_MSI"
    api = "https://datasets-server.huggingface.co/rows"

    # First strategy: viewer API, with retry and partial-success support.
    offsets = [0, 100, 500, 1000, 1600, 3200, 4800, 6400, 8000, 9600, 11200]
    samples = []
    errors = []

    session = requests.Session()

    for offset in offsets:
        for attempt in range(3):
            try:
                response = session.get(
                    api,
                    params={
                        "dataset": dataset_name,
                        "config": "default",
                        "split": "train",
                        "offset": offset,
                        "length": 1,
                    },
                    timeout=30,
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
                            "filename": str(
                                row.get("filename", f"patch_{offset}")
                            ),
                        })

                break

            except Exception as exc:
                errors.append(f"Viewer offset {offset}: {exc}")
                time.sleep(1.0 + attempt)

    if samples:
        return samples

    # Second strategy: official Hugging Face datasets library.
    try:
        from datasets import load_dataset

        dataset = load_dataset(
            dataset_name,
            split="train",
            streaming=True,
        )

        fallback_samples = []

        for i, row in enumerate(dataset):
            image = row.get("image")

            if isinstance(image, dict):
                image = image.get("array", image.get("data"))

            if image is None:
                continue

            image = np.asarray(image)

            if image.shape == (64, 64, 13):
                fallback_samples.append({
                    "image": image,
                    "label": str(row.get("label", "Unknown")),
                    "filename": str(row.get("filename", f"patch_{i}")),
                })

            if len(fallback_samples) >= 8:
                break

        if fallback_samples:
            return fallback_samples

    except Exception as exc:
        errors.append(f"Datasets-library fallback: {exc}")

    raise RuntimeError(
        "Both online retrieval methods failed. The public dataset may be "
        "temporarily unavailable or require a different configuration. "
        "Try again later. Details: " + " | ".join(errors[-3:])
    )


# ---------------------------------------------------------
# 4. VISUALIZATION HELPERS
# ---------------------------------------------------------

def stretch(band):
    band = np.asarray(band, dtype=float)
    low, high = np.nanpercentile(band, [2, 98])

    if high <= low:
        return np.zeros_like(band)

    return np.clip((band - low) / (high - low), 0, 1)


def composite(image, bands):
    return np.dstack([
        stretch(image[:, :, band])
        for band in bands
    ])


def spectral_index(image, a, b):
    a = image[:, :, a].astype(float)
    b = image[:, :, b].astype(float)
    return (a - b) / (a + b + 1e-8)


# ---------------------------------------------------------
# 5. PROJECT OVERVIEW
# ---------------------------------------------------------

col1, col2, col3, col4 = st.columns(4)

col1.metric("Spectral targets", len(distances))
col2.metric("Features per patch", scaler.n_features_in_)
col3.metric("Q-learning states", len(q_table))
col4.metric("SARSA states", len(sarsa_table))

st.caption(
    "State counts refer to saved Q-table entries, not training episodes."
)

tab_images, tab_spectra, tab_rl, tab_metadata = st.tabs([
    "🛰️ Satellite imagery",
    "📊 Spectral analysis",
    "🧠 RL models",
    "📁 Metadata",
])


# ---------------------------------------------------------
# 6. SATELLITE IMAGERY
# ---------------------------------------------------------

with tab_images:
    st.header("Real multispectral satellite imagery")

    if st.button("Load EuroSAT image patches", type="primary"):
        with st.spinner("Retrieving images from Hugging Face..."):
            try:
                st.session_state["orbital_samples"] = fetch_eurosat_samples()
                st.success(
                    f"Loaded {len(st.session_state['orbital_samples'])} image patches."
                )
            except Exception as exc:
                st.error(str(exc))

    samples = st.session_state.get("orbital_samples", [])

    if samples:
        selected_index = st.selectbox(
            "Select image patch",
            range(len(samples)),
            format_func=lambda i: (
                f"Patch {i + 1} — {samples[i]['filename']}"
            ),
        )

        item = samples[selected_index]
        image = item["image"]

        try:
            features = spectral_features(image).reshape(1, -1)
            scaled_features = scaler.transform(features)
            predicted_target = int(
                cluster_model.predict(scaled_features)[0]
            )

            st.success(f"Predicted spectral cluster: Target {predicted_target}")

        except Exception as exc:
            predicted_target = None
            st.error(f"Could not classify this patch: {exc}")

        left, right = st.columns(2)

        with left:
            st.write("**True-colour composite (B4, B3, B2)**")
            st.image(
                composite(image, [3, 2, 1]),
                use_container_width=True,
            )

        with right:
            st.write("**False-colour composite (B8, B4, B3)**")
            st.image(
                composite(image, [7, 3, 2]),
                use_container_width=True,
            )

        st.subheader("Spectral index maps")

        fig, axes = plt.subplots(1, 3, figsize=(12, 4))

        index_specs = [
            (7, 3, "NDVI"),
            (2, 7, "NDWI"),
            (11, 7, "NDBI"),
        ]

        for ax, (a, b, title) in zip(axes, index_specs):
            result = spectral_index(image, a, b)
            plot = ax.imshow(result, cmap="RdYlGn", vmin=-1, vmax=1)
            ax.set_title(title)
            ax.axis("off")
            fig.colorbar(plot, ax=ax, fraction=0.046)

        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        st.write("Dataset label:", item["label"])
        st.write("Image filename:", item["filename"])

    else:
        st.info(
            "Click 'Load EuroSAT image patches'. If Hugging Face is "
            "temporarily unavailable, you can still use the spectral "
            "analysis and saved RL model tabs."
        )


# ---------------------------------------------------------
# 7. SPECTRAL ANALYSIS
# ---------------------------------------------------------

with tab_spectra:
    st.header("Spectral target analysis")

    target_id = st.selectbox(
        "Choose a spectral target",
        range(len(centers)),
    )

    st.subheader(f"Target {target_id}: feature centroid")

    centroid_df = pd.DataFrame({
        "Standardized centroid": centers[target_id],
    })
    st.line_chart(centroid_df)

    st.subheader("Pairwise target distances")

    distance_df = pd.DataFrame(
        distances,
        index=[f"Target {i}" for i in range(len(distances))],
        columns=[f"Target {i}" for i in range(len(distances))],
    )

    st.dataframe(
        distance_df.round(3),
        use_container_width=True,
    )

    st.caption(
        "Clusters represent spectral similarity. They are not automatically "
        "verified land-cover labels or geographic regions."
    )

    summary_file = ROOT / "orbital_spectral_targets.csv"

    if summary_file.exists():
        st.subheader("Saved target summary")
        st.dataframe(
            pd.read_csv(summary_file),
            use_container_width=True,
        )


# ---------------------------------------------------------
# 8. INSPECT SAVED Q-LEARNING AND SARSA
# ---------------------------------------------------------

with tab_rl:
    st.header("Saved reinforcement learning agents")

    chosen_policy = st.radio(
        "Agent",
        ["Q-learning", "SARSA"],
        horizontal=True,
    )

    table = q_table if chosen_policy == "Q-learning" else sarsa_table

    st.metric("States in saved Q-table", len(table))

    if table:
        state_list = list(table.keys())

        state_index = st.number_input(
            "Choose a saved state index",
            min_value=0,
            max_value=len(state_list) - 1,
            value=0,
            step=1,
        )

        selected_state = state_list[int(state_index)]
        action_values = np.asarray(table[selected_state])

        st.write("**State representation**")
        st.code(repr(selected_state))

        st.write("**Action values**")
        st.bar_chart(pd.DataFrame({
            "Q-value": action_values,
        }))

        st.write(
            "Greedy action selected for this saved state:",
            int(np.argmax(action_values)),
        )

        st.caption(
            "This inspects stored values; it does not claim to reproduce "
            "a complete evaluation episode."
        )

    else:
        st.warning("The selected Q-table is empty.")


# ---------------------------------------------------------
# 9. METADATA AND FILES
# ---------------------------------------------------------

with tab_metadata:
    st.header("Saved model metadata")
    st.json(metadata)

    st.subheader("Loaded artifacts")

    for name, value in models.items():
        if name in ("q_learning", "sarsa"):
            st.write(f"- {name}: {len(value)} states")
        elif hasattr(value, "shape"):
            st.write(f"- {name}: shape {value.shape}")
        else:
            st.write(f"- {name}: {type(value).__name__}")

    st.subheader("Optional result files")

    result_files = [
        "orbital_spectral_targets.csv",
        "orbital_policy_rewards.csv",
        "orbital_policy_diversity.csv",
        "orbital_learning_curve.csv",
        "orbital_results.png",
        "orbital_rl_comparison.png",
    ]

    for filename in result_files:
        path = ROOT / filename

        if not path.exists():
            st.caption(f"Not uploaded: {filename}")
            continue

        if path.suffix == ".csv":
            st.markdown(f"**{filename}**")
            st.dataframe(
                pd.read_csv(path),
                use_container_width=True,
            )

        elif path.suffix == ".png":
            st.markdown(f"**{filename}**")
            st.image(str(path), use_container_width=True)


st.divider()
st.caption(
    "ORBITAL | EuroSAT multispectral imagery | Spectral clustering | "
    "Tabular reinforcement learning"
)
