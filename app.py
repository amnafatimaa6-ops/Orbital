
import io
import os
import pickle
from datetime import date, timedelta

import numpy as np
import pandas as pd
import requests
import streamlit as st
import streamlit.components.v1 as components
import matplotlib.pyplot as plt
from PIL import Image

# ==========================================================
# ORBITAL — EARTH OBSERVATORY
# ==========================================================

st.set_page_config(
    page_title="ORBITAL | Earth Observatory",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded",
)

NASA_WMS = (
    "https://gibs.earthdata.nasa.gov/"
    "wms/epsg4326/best/wms.cgi"
)

# ----------------------------- DESIGN -----------------------------

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Manrope:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Manrope', sans-serif;
}

.stApp {
    background: radial-gradient(ellipse at 70% 0%, #17384b 0%, #08121e 48%);
    color: #edf5fb;
}

[data-testid="stSidebar"] {
    background: #0b1724;
    border-right: 1px solid #263b4c;
}

.block-container {
    max-width: 1450px;
    padding-top: 1.5rem;
    padding-bottom: 3rem;
}

.eyebrow {
    font-family: 'DM Mono', monospace;
    color: #71dfc9;
    font-size: 11px;
    letter-spacing: .15em;
    text-transform: uppercase;
}

.hero {
    padding: 32px;
    border-radius: 20px;
    background: linear-gradient(115deg, #112a3d, #102130 60%, #12404a);
    border: 1px solid #315367;
    margin-bottom: 24px;
}

.hero h1 {
    font-size: clamp(34px, 4vw, 54px);
    letter-spacing: -.055em;
    line-height: 1.08;
    margin: 10px 0;
}

.hero p {
    color: #c0d2df;
    max-width: 850px;
    line-height: 1.8;
}

.section {
    margin-top: 34px;
    margin-bottom: 18px;
    border-bottom: 1px solid #294052;
    padding-bottom: 12px;
}

.section h2 {
    margin: 5px 0;
    font-size: 26px;
    letter-spacing: -.04em;
}

.panel {
    background: #0d1b29;
    border: 1px solid #263c4e;
    padding: 18px;
    border-radius: 15px;
    margin-bottom: 12px;
}

.muted {
    color: #a9bdcc;
    font-size: 13px;
    line-height: 1.7;
}

div[data-testid="stMetric"] {
    background: #0d1b29;
    border: 1px solid #263c4e;
    padding: 14px;
    border-radius: 13px;
}

.stButton button {
    border-radius: 9px;
    font-weight: 700;
    min-height: 42px;
}

a {
    color: #71dfc9 !important;
}
</style>
""", unsafe_allow_html=True)


def section(number, title, subtitle):
    st.markdown(
        f"""
        <div class="section">
            <div class="eyebrow">{number} / OBSERVATORY MODULE</div>
            <h2>{title}</h2>
            <div class="muted">{subtitle}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def encode_image(image):
    """Convert a PIL image into downloadable JPEG bytes."""
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=92)
    return buffer.getvalue()


def load_model_artifact():
    path = "orbital_models.pkl"

    if not os.path.exists(path):
        return None, f"{path} was not found beside app.py."

    try:
        with open(path, "rb") as file:
            bundle = pickle.load(file)

        if not isinstance(bundle, dict):
            return None, "The model artifact is not a dictionary."

        return bundle, None

    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"


def get_field(bundle, names, default=None):
    if not isinstance(bundle, dict):
        return default

    for name in names:
        if name in bundle:
            return bundle[name]

    return default


# ----------------------------- NASA IMAGERY -----------------------------

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_nasa_image(layer, selected_date):
    """Fetch a global NASA GIBS map image."""
    params = {
        "SERVICE": "WMS",
        "REQUEST": "GetMap",
        "VERSION": "1.1.1",
        "LAYERS": layer,
        "STYLES": "",
        "SRS": "EPSG:4326",
        "BBOX": "-180,-90,180,90",
        "WIDTH": "1600",
        "HEIGHT": "800",
        "FORMAT": "image/jpeg",
        "TIME": selected_date,
    }

    response = requests.get(
        NASA_WMS,
        params=params,
        timeout=45,
        headers={"User-Agent": "ORBITAL-Earth-Observatory/1.0"},
    )
    response.raise_for_status()

    content_type = response.headers.get("content-type", "")

    if "image" not in content_type.lower():
        raise RuntimeError(
            "NASA GIBS did not return an image. "
            "The selected layer or date may be unavailable."
        )

    try:
        return Image.open(io.BytesIO(response.content)).convert("RGB")
    except Exception as exc:
        raise RuntimeError(
            "The server response could not be decoded as an image."
        ) from exc


# ----------------------------- EUROSAT DATA -----------------------------

@st.cache_data(ttl=3600, show_spinner=False)
def load_eurosat_samples(limit=20):
    """Load sample images from EuroSAT MSI."""
    from datasets import load_dataset

    dataset = load_dataset(
        "blanchon/EuroSAT_MSI",
        split="train",
        streaming=True,
    )

    samples = []

    for i, row in enumerate(dataset):
        image = row.get("image")

        if image is None:
            continue

        samples.append({
            "id": i,
            "image": np.asarray(image),
            "label": str(row.get("label", "Unknown")),
            "filename": str(row.get("filename", f"sample_{i}")),
        })

        if len(samples) >= limit:
            break

    return samples


def image_hwc(image):
    """Normalize image layout to height × width × channels."""
    arr = np.asarray(image)

    if arr.ndim == 2:
        return arr.astype(np.float32)

    if arr.ndim != 3:
        raise ValueError(f"Unexpected image dimensions: {arr.shape}")

    # Convert channel-first arrays such as (13, 64, 64).
    if arr.shape[0] in (3, 4, 13) and arr.shape[-1] not in (3, 4, 13):
        arr = np.moveaxis(arr, 0, -1)

    return arr.astype(np.float32)


def stretch(band):
    """Percentile contrast stretch for visualization."""
    band = np.asarray(band, dtype=float)
    low, high = np.nanpercentile(band, [2, 98])

    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        return np.zeros_like(band)

    return np.clip((band - low) / (high - low), 0, 1)


def make_composite(image, mode):
    arr = image_hwc(image)

    if arr.ndim != 3 or arr.shape[-1] < 12:
        raise ValueError(
            "Expected an image containing the required multispectral bands."
        )

    if mode == "Natural-colour approximation":
        bands = [2, 1, 0]
    elif mode == "Vegetation false colour":
        bands = [7, 3, 2]
    else:
        bands = [11, 7, 3]

    return np.stack(
        [stretch(arr[:, :, band]) for band in bands],
        axis=-1,
    )


def calculate_indices(image):
    arr = image_hwc(image)

    if arr.ndim != 3 or arr.shape[-1] < 12:
        raise ValueError("The selected image does not contain enough bands.")

    green = arr[:, :, 2]
    red = arr[:, :, 3]
    nir = arr[:, :, 7]
    swir = arr[:, :, 11]
    eps = 1e-8

    return {
        "NDVI": (nir - red) / (nir + red + eps),
        "NDWI": (green - nir) / (green + nir + eps),
        "NDBI": (swir - nir) / (swir + nir + eps),
    }


def calculate_features(image):
    arr = image_hwc(image)
    indices = calculate_indices(image)

    band_means = np.nanmean(arr, axis=(0, 1))[:13].tolist()

    index_features = []
    for name in ["NDVI", "NDWI", "NDBI"]:
        values = indices[name]
        index_features.extend([
            float(np.nanmean(values)),
            float(np.nanstd(values)),
        ])

    return np.asarray(band_means + index_features)


# ----------------------------- MODEL -----------------------------

bundle, model_error = load_model_artifact()

q_learning = get_field(
    bundle, ["q_learning", "Q_v2", "Q", "q_table"], {}
)
sarsa = get_field(
    bundle, ["sarsa", "Q_sarsa", "SARSA", "sarsa_table"], {}
)
scaler = get_field(bundle, ["scaler"])
region_model = get_field(bundle, ["region_model"])
target_features = get_field(bundle, ["target_features"])
distance_matrix = get_field(bundle, ["distance_matrix"])


# ==========================================================
# SIDEBAR
# ==========================================================

with st.sidebar:
    st.markdown(
        '<div class="eyebrow">ORBITAL / EARTH LAB</div>',
        unsafe_allow_html=True,
    )
    st.title("Observation settings")

    st.markdown("#### NASA imagery")

    layer_options = {
        "Terra MODIS · True colour":
            "MODIS_Terra_CorrectedReflectance_TrueColor",
        "Suomi NPP VIIRS · True colour":
            "VIIRS_SNPP_CorrectedReflectance_TrueColor",
    }

    layer_label = st.selectbox(
        "Satellite product",
        list(layer_options.keys()),
    )
    nasa_layer = layer_options[layer_label]

    selected_date = st.date_input(
        "Imagery date",
        value=date.today() - timedelta(days=3),
        min_value=date(2002, 1, 1),
        max_value=date.today() - timedelta(days=1),
    )

    st.caption(
        "Some imagery products may not be available for every date."
    )

    st.divider()
    st.markdown("#### Research artifact")

    if bundle is not None:
        st.success("Model artifact loaded")
        st.metric("Q-learning states", len(q_learning or {}))
        st.metric("SARSA states", len(sarsa or {}))
    else:
        st.warning("Model artifact unavailable")
        st.caption(model_error)

    st.divider()
    st.markdown("[NASA Worldview](https://worldview.earthdata.nasa.gov/)")
    st.markdown("[NASA GIBS documentation](https://nasa-gibs.github.io/gibs-api-docs/)")
    st.markdown("[EuroSAT MSI dataset](https://huggingface.co/datasets/blanchon/EuroSAT_MSI)")


# ==========================================================
# HEADER
# ==========================================================

st.markdown("""
<div class="hero">
    <div class="eyebrow">EARTH OBSERVATION · REMOTE SENSING · AI</div>
    <h1>ORBITAL Earth Observatory</h1>
    <p>
        Explore satellite imagery, reveal spectral patterns across Earth's
        surface, and investigate reinforcement learning for information-efficient
        Earth observation.
    </p>
    <div class="eyebrow">
        NASA IMAGERY &nbsp; / &nbsp; MULTISPECTRAL ANALYSIS &nbsp; / &nbsp; RL RESEARCH
    </div>
</div>
""", unsafe_allow_html=True)


# ==========================================================
# 01 — EARTH IMAGERY
# ==========================================================

section(
    "01",
    "Earth from orbit",
    "Explore NASA satellite imagery before entering the multispectral lab.",
)

image_tab, worldview_tab = st.tabs([
    "Satellite image",
    "Interactive NASA Worldview",
])

with image_tab:
    st.markdown(
        f"**Product:** `{layer_label}`  ·  "
        f"**Requested date:** `{selected_date.isoformat()}`"
    )

    if st.button("Load satellite imagery", type="primary"):
        st.session_state["load_nasa_image"] = True

    if (
        st.session_state.get("load_nasa_image", False)
        or "nasa_image_bytes" in st.session_state
    ):
        with st.spinner("Requesting satellite imagery from NASA GIBS…"):
            try:
                image = fetch_nasa_image(
                    nasa_layer,
                    selected_date.isoformat(),
                )

                st.session_state["nasa_image_bytes"] = encode_image(image)
                st.session_state["nasa_image_key"] = (
                    nasa_layer,
                    selected_date.isoformat(),
                )

            except Exception as exc:
                st.error(
                    "NASA imagery could not be loaded for this product/date. "
                    "Try a different date or open the interactive Worldview tab."
                )
                with st.expander("Technical details"):
                    st.code(f"{type(exc).__name__}: {exc}")

    current_key = (nasa_layer, selected_date.isoformat())
    stored_key = st.session_state.get("nasa_image_key")

    if (
        st.session_state.get("nasa_image_bytes")
        and stored_key == current_key
    ):
        image_bytes = st.session_state["nasa_image_bytes"]

        st.image(
            image_bytes,
            caption=(
                f"NASA GIBS visualization · {layer_label} · "
                f"{selected_date.isoformat()}"
            ),
            use_container_width=True,
        )

        st.download_button(
            "Download displayed NASA image",
            data=image_bytes,
            file_name=f"orbital_nasa_{selected_date.isoformat()}.jpg",
            mime="image/jpeg",
        )

    st.markdown("""
    <div class="muted">
        This is a global satellite visualization, not street-level imagery.
        Cloud cover, resolution, lighting, and acquisition time influence
        what is visible.
    </div>
    """, unsafe_allow_html=True)

with worldview_tab:
    st.markdown(
        "Pan, zoom, change dates, and explore NASA's interactive Earth-observation layers."
    )

    components.iframe(
        "https://worldview.earthdata.nasa.gov/",
        height=700,
        scrolling=True,
    )

    st.caption(
        "The embedded viewer is operated by NASA. If the embed is blocked "
        "by your browser or deployment, open NASA Worldview directly."
    )

    st.link_button(
        "Open NASA Worldview",
        "https://worldview.earthdata.nasa.gov/",
    )


# ==========================================================
# 02 — MULTISPECTRAL LAB
# ==========================================================

section(
    "02",
    "Multispectral analysis lab",
    "Compare EuroSAT MSI band composites and explore vegetation, water, and built-up indices.",
)

st.markdown("""
<div class="panel">
    <b>What remote sensing reveals</b>
    <div class="muted">
        Satellites measure reflected energy at different wavelengths.
        False-colour composites make spectral differences easier to see,
        while indices such as NDVI, NDWI, and NDBI summarize selected band
        relationships. They are useful indicators, not definitive classifications.
    </div>
</div>
""", unsafe_allow_html=True)

samples = None
dataset_error = None

try:
    with st.spinner("Loading EuroSAT multispectral samples…"):
        samples = load_eurosat_samples(20)
except Exception as exc:
    dataset_error = f"{type(exc).__name__}: {exc}"

if samples:
    sample_names = [
        f"{sample['id']:03d} · {sample['filename']}"
        for sample in samples
    ]

    chosen = st.selectbox("Select a satellite sample", sample_names)
    sample = samples[sample_names.index(chosen)]

    composite_mode = st.radio(
        "Spectral composite",
        [
            "Natural-colour approximation",
            "Vegetation false colour",
            "SWIR / NIR / Red",
        ],
        horizontal=True,
    )

    try:
        image = sample["image"]
        composite_image = make_composite(image, composite_mode)
        indices = calculate_indices(image)
        features = calculate_features(image)

        left, right = st.columns([1.15, 0.85], gap="large")

        with left:
            st.image(
                composite_image,
                caption=f"{sample['filename']} · {composite_mode}",
                use_container_width=True,
            )
            st.caption(
                "The colours are generated from selected spectral bands. "
                "False-colour imagery does not represent human vision."
            )

        with right:
            st.markdown("#### Spectral measurements")

            a, b, c = st.columns(3)
            a.metric("Mean NDVI", f"{np.nanmean(indices['NDVI']):.3f}")
            b.metric("Mean NDWI", f"{np.nanmean(indices['NDWI']):.3f}")
            c.metric("Mean NDBI", f"{np.nanmean(indices['NDBI']):.3f}")

            st.markdown(
                f"<div class='muted'>Dataset label: <b>{sample['label']}</b>"
                "<br>The label is metadata from the dataset, not a location "
                "identified from the image.</div>",
                unsafe_allow_html=True,
            )

        st.markdown("#### Spectral index maps")

        cols = st.columns(3)
        palettes = {
            "NDVI": "RdYlGn",
            "NDWI": "BrBG",
            "NDBI": "PuOr",
        }

        for col, name in zip(cols, ["NDVI", "NDWI", "NDBI"]):
            with col:
                fig, ax = plt.subplots(figsize=(5, 4))
                plot = ax.imshow(
                    indices[name],
                    cmap=palettes[name],
                    vmin=-1,
                    vmax=1,
                )
                ax.set_title(name)
                ax.axis("off")
                fig.colorbar(plot, ax=ax, fraction=0.046, pad=0.04)
                fig.tight_layout()
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)

        with st.expander("Inspect all 19 extracted features"):
            names = (
                [f"Band {i} mean" for i in range(1, 14)]
                + [
                    "NDVI mean", "NDVI std",
                    "NDWI mean", "NDWI std",
                    "NDBI mean", "NDBI std",
                ]
            )

            feature_df = pd.DataFrame({
                "Feature": names,
                "Value": features,
            })

            st.dataframe(feature_df, use_container_width=True, hide_index=True)

            st.download_button(
                "Download spectral features",
                feature_df.to_csv(index=False).encode("utf-8"),
                file_name="orbital_spectral_features.csv",
                mime="text/csv",
            )

    except Exception as exc:
        st.error(f"Could not process the selected sample: {exc}")

else:
    st.warning(
        "EuroSAT MSI samples could not be loaded. The NASA imagery viewer "
        "is independent of this dataset."
    )

    if dataset_error:
        with st.expander("Dataset loading details"):
            st.code(dataset_error)

    st.markdown(
        "Check that `datasets` appears in `requirements.txt`, then reboot "
        "the Streamlit app after the dependency installation completes."
    )


# ==========================================================
# 03 — SPECTRAL CLUSTERS
# ==========================================================

section(
    "03",
    "Spectral target explorer",
    "Explore the unsupervised feature groups saved by your research notebook.",
)

if bundle is not None:
    centers = target_features
    distances = distance_matrix

    if centers is not None:
        centers = np.asarray(centers)

        if centers.ndim == 2 and len(centers) > 0:
            target_id = st.selectbox(
                "Choose a spectral target",
                list(range(min(8, len(centers)))),
            )

            centroid = centers[target_id].reshape(1, -1)

            try:
                if (
                    scaler is not None
                    and centroid.shape[1] == scaler.n_features_in_
                ):
                    centroid = scaler.inverse_transform(centroid)
            except Exception:
                pass

            if centroid.shape[1] >= 13:
                fig, ax = plt.subplots(figsize=(10, 4))
                ax.plot(
                    range(1, 14),
                    centroid.ravel()[:13],
                    marker="o",
                    linewidth=2,
                )
                ax.set_xlabel("Spectral band")
                ax.set_ylabel("Centroid feature value")
                ax.set_title(f"Spectral profile · Target {target_id}")
                ax.grid(alpha=0.25)
                fig.tight_layout()
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)

    if distances is not None:
        distances = np.asarray(distances)

        fig, ax = plt.subplots(figsize=(7, 5))
        plot = ax.imshow(distances, cmap="magma")
        ax.set_title("Pairwise spectral-target distances")
        ax.set_xlabel("Target")
        ax.set_ylabel("Target")
        fig.colorbar(plot, ax=ax)
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    st.caption(
        "These are spectral clusters in feature space, not geographic regions. "
        "A cluster ID alone does not establish whether a location is forest, water, or city."
    )

else:
    st.warning("The saved model artifact is unavailable. Check the app logs.")


# ==========================================================
# 04 — REINFORCEMENT LEARNING
# ==========================================================

section(
    "04",
    "Reinforcement-learning research",
    "Inspect the saved agents and understand the observation-planning experiment.",
)

st.markdown("""
<div class="panel">
    <h3>What is the agent learning?</h3>
    <div class="muted">
        Under a limited observation budget, an agent must choose which spectral
        target to observe. A novelty-aware reward can favor observations that
        add information beyond what has already been seen.
    </div>
</div>
""", unsafe_allow_html=True)

if bundle is not None:
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Q-learning")
        st.metric("Stored states", len(q_learning or {}))
        st.caption("Off-policy temporal-difference learning.")

    with col2:
        st.markdown("#### SARSA")
        st.metric("Stored states", len(sarsa or {}))
        st.caption("On-policy temporal-difference learning.")

    st.warning(
        "The original OrbitalEnvV2 state encoding and action-selection logic "
        "must be restored before this app can honestly report new missions "
        "as evaluations of the trained Q-learning and SARSA agents. The model "
        "tables load, but this dashboard does not fabricate learned-policy results."
    )

else:
    st.warning("The model artifact could not be loaded.")


# ==========================================================
# 05 — METHODS AND SOURCES
# ==========================================================

section(
    "05",
    "Methods and data provenance",
    "Distinguish the source imagery, the spectral experiment, and the RL model.",
)

with st.expander("Project methodology and limitations"):
    st.markdown("""
    **NASA imagery**
    - Global satellite visualization is requested from NASA GIBS.
    - NASA Worldview provides a separate interactive Earth-observation viewer.
    - NASA imagery is not automatically fed into the EuroSAT-trained clustering model.

    **EuroSAT MSI**
    - Multispectral samples come from `blanchon/EuroSAT_MSI`.
    - Features comprise 13 band means and six statistics from NDVI, NDWI, and NDBI.
    - Index maps are spectral indicators and should not be treated as verified labels.

    **Spectral clustering**
    - The model's groups represent similarity in the stored feature space.
    - Cluster IDs are arbitrary and do not encode geographic coordinates.

    **Reinforcement learning**
    - Saved Q-learning and SARSA tables are loaded from `orbital_models.pkl`.
    - Faithful policy evaluation requires the exact training environment, state
      representation, transition rules, and reward function.
    - This prototype is not an operational satellite controller.

    **Acknowledgement**
    NASA GIBS imagery services are provided through NASA's Earth Science Data
    and Information System.
    """)

st.divider()

st.markdown("""
<div style="text-align:center;color:#91a8b9;font-size:12px;padding:12px">
    ORBITAL · EARTH OBSERVATORY · REMOTE-SENSING RESEARCH
</div>
""", unsafe_allow_html=True)
