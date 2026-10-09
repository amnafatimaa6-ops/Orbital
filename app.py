
from __future__ import annotations

import io
import pickle
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import streamlit as st
import matplotlib.pyplot as plt

from PIL import Image
from sklearn.preprocessing import StandardScaler


# ============================================================
# ORBITAL — EARTH OBSERVATORY
# ============================================================

st.set_page_config(
    page_title="ORBITAL | Earth Observatory",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = Path(__file__).resolve().parent
NASA_WMS = "https://gibs.earthdata.nasa.gov/wms/epsg4326/best/wms.cgi"
WORLDVIEW_URL = "https://worldview.earthdata.nasa.gov/"
EUROSAT_URL = "https://huggingface.co/datasets/blanchon/EuroSAT_MSI"

NASA_PRODUCTS = {
    "Suomi NPP VIIRS · True colour":
        "VIIRS_SNPP_CorrectedReflectance_TrueColor",
    "Terra MODIS · True colour":
        "MODIS_Terra_CorrectedReflectance_TrueColor",
}


# ============================================================
# STYLE
# ============================================================

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Space+Grotesk:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Space Grotesk', sans-serif;
    }

    .stApp {
        background:
            radial-gradient(ellipse at 10% 0%, #132c3b 0%, transparent 40%),
            linear-gradient(180deg, #08121b 0%, #0b1119 100%);
        color: #eaf2f6;
    }

    [data-testid="stSidebar"] {
        background: #0c1721;
        border-right: 1px solid #203442;
    }

    h1, h2, h3 {
        color: #eff8ff !important;
        letter-spacing: -0.035em;
    }

    .eyebrow {
        color: #7eddd0;
        font-family: 'DM Mono', monospace;
        font-size: 0.75rem;
        letter-spacing: 0.12em;
        text-transform: uppercase;
    }

    .hero {
        padding: 2.1rem 0 1.3rem 0;
        border-bottom: 1px solid #233946;
        margin-bottom: 2rem;
    }

    .hero h1 {
        font-size: clamp(2.4rem, 6vw, 4.7rem);
        line-height: 1;
        margin: 0.5rem 0 1rem 0;
    }

    .hero p {
        color: #a9bac5;
        max-width: 780px;
        font-size: 1.08rem;
    }

    .module {
        font-family: 'DM Mono', monospace;
        color: #79d9cd;
        font-size: 0.72rem;
        letter-spacing: 0.11em;
        margin-bottom: 0.35rem;
    }

    .muted {
        color: #a9bac5;
    }

    .panel {
        background: #101e29;
        border: 1px solid #243b49;
        border-radius: 12px;
        padding: 1rem 1.2rem;
        margin: 0.4rem 0 1rem 0;
    }

    div[data-testid="stMetric"] {
        background: #101e29;
        border: 1px solid #243b49;
        border-radius: 10px;
        padding: 0.8rem;
    }

    div[data-testid="stMetricValue"] {
        color: #8be0d2;
    }

    .stButton button, .stLinkButton a {
        border-radius: 8px;
    }

    hr {
        border-color: #233946;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# HELPERS
# ============================================================

def first_existing_path(paths):
    for path in paths:
        if path.exists():
            return path
    return None


def load_model_artifact():
    """Load the saved model bundle without stopping the whole app."""
    artifact_path = first_existing_path(
        [
            BASE_DIR / "orbital_models.pkl",
            BASE_DIR / "orbital_models" / "orbital_models.pkl",
            BASE_DIR / "orbital_models" / "orbital_models.pickle",
        ]
    )

    if artifact_path is None:
        return None, (
            "Model artifact not found. Expected orbital_models.pkl "
            "in the repository root or orbital_models/ directory."
        )

    try:
        with artifact_path.open("rb") as file:
            artifact = pickle.load(file)

        if not isinstance(artifact, dict):
            return None, "The model artifact loaded, but it is not a dictionary."

        return artifact, f"Loaded {artifact_path.name}"

    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"


def get_field(artifact, names, default=None):
    if not isinstance(artifact, dict):
        return default

    for name in names:
        if name in artifact:
            return artifact[name]

    return default


def encode_image(image):
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=92)
    return buffer.getvalue()


def validate_image_response(response):
    """Reject error pages and invalid image payloads."""
    response.raise_for_status()

    content_type = response.headers.get("Content-Type", "").lower()
    if not content_type.startswith("image/"):
        raise RuntimeError(
            "The server response was not an image "
            f"(Content-Type: {content_type or 'missing'})."
        )

    with Image.open(io.BytesIO(response.content)) as check:
        check.verify()

    with Image.open(io.BytesIO(response.content)) as loaded:
        image = loaded.convert("RGB")

    if image.width < 100 or image.height < 100:
        raise RuntimeError("The returned image is unexpectedly small.")

    return image


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_nasa_image(requested_layer, selected_date):
    """
    Try the requested GIBS product and then alternate products/dates.
    Returns: image, actual layer, actual date, fallback_used.
    """
    requested_date = date.fromisoformat(str(selected_date))

    other_layer = (
        "MODIS_Terra_CorrectedReflectance_TrueColor"
        if requested_layer == "VIIRS_SNPP_CorrectedReflectance_TrueColor"
        else "VIIRS_SNPP_CorrectedReflectance_TrueColor"
    )

    attempts = [
        (requested_layer, requested_date),
        (other_layer, requested_date),
        (requested_layer, requested_date - timedelta(days=1)),
        (other_layer, requested_date - timedelta(days=1)),
        (requested_layer, requested_date - timedelta(days=2)),
        (other_layer, requested_date - timedelta(days=2)),
    ]

    errors = []

    for layer, image_date in attempts:
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
            "TIME": image_date.isoformat(),
        }

        try:
            response = requests.get(
                NASA_WMS,
                params=params,
                timeout=(8, 30),
                headers={
                    "User-Agent": "ORBITAL-Earth-Observatory/1.1"
                },
            )

            image = validate_image_response(response)

            fallback_used = (
                layer != requested_layer
                or image_date != requested_date
            )

            return image, layer, image_date.isoformat(), fallback_used

        except Exception as exc:
            errors.append(
                f"{layer} / {image_date.isoformat()}: "
                f"{type(exc).__name__}: {exc}"
            )

    raise RuntimeError(
        "NASA GIBS requests failed for all attempted products and dates.\n\n"
        + "\n".join(errors)
    )


@st.cache_data(ttl=86400, show_spinner=False)
def load_eurosat_samples(limit=20):
    """Load a small sample of EuroSAT MSI using streaming."""
    from datasets import load_dataset

    dataset = load_dataset(
        "blanchon/EuroSAT_MSI",
        split="train",
        streaming=True,
    )

    samples = []
    for sample in dataset:
        samples.append(sample)
        if len(samples) >= limit:
            break

    return samples


def image_hwc(value):
    """Convert a multispectral sample to H × W × bands."""
    if isinstance(value, Image.Image):
        array = np.asarray(value)
    elif isinstance(value, dict):
        if "array" in value:
            array = np.asarray(value["array"])
        elif "path" in value and value["path"]:
            array = np.asarray(Image.open(value["path"]))
        else:
            raise ValueError("Unrecognized image dictionary.")
    else:
        array = np.asarray(value)

    if array.ndim != 3:
        raise ValueError(
            f"Expected a 3-D multispectral array; got shape {array.shape}."
        )

    # EuroSAT arrays may use either HWC or CHW.
    if array.shape[-1] <= 20:
        result = array
    elif array.shape[0] <= 20:
        result = np.moveaxis(array, 0, -1)
    else:
        raise ValueError(
            f"Could not identify spectral-band axis in {array.shape}."
        )

    return result.astype(np.float32)


def stretch(array):
    """Percentile stretch for display, independent of source data range."""
    array = np.asarray(array, dtype=np.float32)
    result = np.zeros_like(array, dtype=np.float32)

    if array.ndim == 2:
        low, high = np.nanpercentile(array, [2, 98])
        if high <= low:
            return result
        return np.clip((array - low) / (high - low), 0, 1)

    for channel in range(array.shape[-1]):
        plane = array[..., channel]
        low, high = np.nanpercentile(plane, [2, 98])
        if high > low:
            result[..., channel] = np.clip(
                (plane - low) / (high - low), 0, 1
            )

    return result


def make_composite(cube, band_indices):
    available = cube.shape[-1]
    if max(band_indices) >= available:
        raise ValueError(
            f"This sample has {available} bands, but the composite "
            f"requires band index {max(band_indices)}."
        )

    selected = cube[..., list(band_indices)]
    return stretch(selected)


def calculate_indices(cube):
    """
    Band indices assume EuroSAT MSI band ordering used in this project:
    green=2, red=3, NIR=7, SWIR=11.
    """
    if cube.shape[-1] <= 11:
        raise ValueError(
            "At least 12 bands are needed to calculate these indices."
        )

    green = cube[..., 2]
    red = cube[..., 3]
    nir = cube[..., 7]
    swir = cube[..., 11]
    eps = 1e-8

    ndvi = (nir - red) / (nir + red + eps)
    ndwi = (green - nir) / (green + nir + eps)
    ndbi = (swir - nir) / (swir + nir + eps)

    return {
        "NDVI": np.clip(ndvi, -1, 1),
        "NDWI": np.clip(ndwi, -1, 1),
        "NDBI": np.clip(ndbi, -1, 1),
    }


def calculate_features(cube):
    """13 band means + mean and standard deviation of 3 indices."""
    indices = calculate_indices(cube)
    features = [float(np.nanmean(cube[..., band]))
                for band in range(cube.shape[-1])]

    for name in ("NDVI", "NDWI", "NDBI"):
        index = indices[name]
        features.extend(
            [
                float(np.nanmean(index)),
                float(np.nanstd(index)),
            ]
        )

    return np.asarray(features, dtype=np.float32)


def find_image(sample):
    for key in ("image", "img", "pixel_values"):
        if key in sample:
            return sample[key], key
    raise KeyError(
        "Could not find an image field. Available fields: "
        + ", ".join(sample.keys())
    )


def make_index_figure(index_name, array):
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(array, cmap="RdYlGn", vmin=-1, vmax=1)
    ax.set_title(index_name)
    ax.set_axis_off()
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    return fig


def display_feature_table(features):
    table = pd.DataFrame(
        {
            "Feature": [
                *[f"Band {i + 1} mean" for i in range(13)],
                "NDVI mean", "NDVI std",
                "NDWI mean", "NDWI std",
                "NDBI mean", "NDBI std",
            ],
            "Value": features,
        }
    )
    st.dataframe(table, use_container_width=True, hide_index=True)


# ============================================================
# LOAD ARTIFACT
# ============================================================

artifact, artifact_status = load_model_artifact()

q_learning = get_field(
    artifact,
    ["Q_v2", "Q_learning", "q_learning", "q_table"],
    {},
)

sarsa = get_field(
    artifact,
    ["Q_sarsa", "SARSA", "sarsa", "sarsa_table"],
    {},
)

scaler = get_field(
    artifact,
    ["scaler", "feature_scaler"],
)

region_model = get_field(
    artifact,
    ["region_model", "kmeans", "cluster_model"],
)

target_features = get_field(
    artifact,
    ["target_features", "features", "X_features"],
)

distance_matrix = get_field(
    artifact,
    ["distance_matrix", "distances"],
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.markdown('<div class="eyebrow">ORBITAL / EARTH LAB</div>',
                unsafe_allow_html=True)
    st.title("Observation settings")

    st.markdown("#### NASA imagery")

    max_date = date.today() - timedelta(days=1)
    min_date = date(2002, 1, 1)
    default_date = min(date(2026, 10, 6), max_date)

    selected_date = st.date_input(
        "Requested observation date",
        value=default_date,
        min_value=min_date,
        max_value=max_date,
        help="Imagery availability varies by product and date.",
    )

    layer_label = st.selectbox(
        "Satellite product",
        options=list(NASA_PRODUCTS.keys()),
        index=0,
    )
    nasa_layer = NASA_PRODUCTS[layer_label]

    st.caption(
        "If a product/date fails, ORBITAL tries another supported "
        "product and nearby dates."
    )

    st.divider()
    st.markdown("#### Research artifact")

    if artifact is not None:
        st.success("Model artifact loaded")
    else:
        st.warning("Model artifact unavailable")

    st.caption(artifact_status)

    st.metric("Q-learning states", len(q_learning or {}))
    st.metric("SARSA states", len(sarsa or {}))

    st.divider()
    st.markdown(f"[NASA Worldview]({WORLDVIEW_URL})")
    st.markdown(
        "[NASA GIBS documentation]"
        "(https://nasa-gibs.github.io/gibs-api-docs/)"
    )
    st.markdown(f"[EuroSAT MSI dataset]({EUROSAT_URL})")


# ============================================================
# HERO
# ============================================================

st.markdown(
    """
    <div class="hero">
        <div class="eyebrow">Earth observation · Remote sensing · AI</div>
        <h1>ORBITAL Earth Observatory</h1>
        <p>
            Explore satellite imagery, reveal spectral patterns across
            Earth's surface, and investigate reinforcement learning for
            information-efficient Earth observation.
        </p>
        <div class="eyebrow">
            NASA imagery &nbsp; / &nbsp; Multispectral analysis
            &nbsp; / &nbsp; RL research
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# 01 — EARTH FROM ORBIT
# ============================================================

st.markdown(
    '<div class="module">01 / OBSERVATORY MODULE</div>',
    unsafe_allow_html=True,
)
st.header("Earth from orbit")
st.write(
    "Explore NASA satellite imagery before entering the multispectral lab."
)

image_tab, worldview_tab = st.tabs(
    ["Satellite image", "Interactive NASA Worldview"]
)

with image_tab:
    st.markdown(
        f"**Requested product:** `{layer_label}` · "
        f"**Requested date:** `{selected_date.isoformat()}`"
    )

    try:
        with st.spinner("Connecting to NASA GIBS and loading imagery…"):
            nasa_image, actual_layer, actual_date, used_fallback = (
                fetch_nasa_image(
                    nasa_layer,
                    selected_date.isoformat(),
                )
            )

        st.image(
            nasa_image,
            caption=(
                f"NASA GIBS visualization · {actual_layer} · {actual_date}"
            ),
            use_container_width=True,
        )

        st.download_button(
            "Download satellite image",
            data=encode_image(nasa_image),
            file_name=f"orbital_nasa_{actual_date}.jpg",
            mime="image/jpeg",
        )

        if used_fallback:
            st.info(
                f"The requested product/date was unavailable. "
                f"Showing the successful fallback: {actual_layer}, "
                f"{actual_date}."
            )

        st.caption(
            "This is a global satellite visualization, not street-level "
            "imagery. Cloud cover, resolution, lighting, and acquisition "
            "time influence what is visible."
        )

    except Exception as exc:
        st.error(
            "NASA imagery could not be loaded after the fallback attempts. "
            "This may be a temporary NASA service, network, or date/product "
            "availability issue."
        )

        with st.expander("NASA connection details"):
            st.code(f"{type(exc).__name__}: {exc}")

        st.link_button(
            "Open NASA Worldview",
            WORLDVIEW_URL,
        )

        st.info(
            "The rest of ORBITAL can still work even when the NASA "
            "image service is unavailable."
        )

with worldview_tab:
    st.markdown(
        "Use NASA Worldview for interactive browsing, date changes, "
        "and available imagery layers."
    )

    st.link_button(
        "Open NASA Worldview in a new tab",
        WORLDVIEW_URL,
    )

    st.components.v1.iframe(
        WORLDVIEW_URL,
        height=680,
        scrolling=True,
    )


# ============================================================
# 02 — MULTISPECTRAL ANALYSIS LAB
# ============================================================

st.divider()
st.markdown(
    '<div class="module">02 / OBSERVATORY MODULE</div>',
    unsafe_allow_html=True,
)
st.header("Multispectral analysis lab")
st.write(
    "Compare EuroSAT MSI band composites and explore vegetation, "
    "water, and built-up indices."
)

st.markdown(
    """
    <div class="panel">
        <strong>What remote sensing reveals</strong><br><br>
        Satellites measure reflected energy at different wavelengths.
        False-colour composites make spectral differences easier to see,
        while NDVI, NDWI, and NDBI summarize selected band relationships.
        These are useful indicators, not definitive classifications.
    </div>
    """,
    unsafe_allow_html=True,
)

try:
    with st.spinner("Loading EuroSAT MSI samples…"):
        samples = load_eurosat_samples(limit=30)

    if not samples:
        raise RuntimeError("The dataset returned no samples.")

    sample_options = []
    for index, sample in enumerate(samples):
        name = sample.get("filename", sample.get("image_name", f"Sample {index + 1}"))
        sample_options.append(f"{index + 1:02d} · {name}")

    selected_sample_index = st.selectbox(
        "Choose a multispectral sample",
        options=list(range(len(samples))),
        format_func=lambda i: sample_options[i],
    )

    sample = samples[selected_sample_index]
    image_value, image_key = find_image(sample)
    cube = image_hwc(image_value)

    if cube.shape[-1] < 12:
        raise ValueError(
            f"Selected image has only {cube.shape[-1]} bands; "
            "this project expects at least 12."
        )

    composite_tab1, composite_tab2, composite_tab3 = st.tabs(
        [
            "Natural-colour approximation",
            "Vegetation false colour",
            "SWIR / NIR / Red",
        ]
    )

    with composite_tab1:
        st.image(
            make_composite(cube, (2, 1, 0)),
            caption="Band combination: 3 / 2 / 1 (display approximation)",
            use_container_width=True,
        )

    with composite_tab2:
        st.image(
            make_composite(cube, (7, 3, 2)),
            caption="Band combination: 8 / 4 / 3 (false colour)",
            use_container_width=True,
        )

    with composite_tab3:
        st.image(
            make_composite(cube, (11, 7, 3)),
            caption="Band combination: 12 / 8 / 4 (SWIR / NIR / Red)",
            use_container_width=True,
        )

    st.caption(
        "The selected band combinations follow this project's assumed "
        "EuroSAT MSI band ordering. Colours are stretched for visualization; "
        "they are not raw sensor display values."
    )

    indices = calculate_indices(cube)
    features = calculate_features(cube)

    st.markdown("#### Spectral measurements")

    metric_cols = st.columns(3)
    for column, name in zip(metric_cols, ["NDVI", "NDWI", "NDBI"]):
        column.metric(name, f"{np.nanmean(indices[name]):.3f}")

    label = sample.get("label", sample.get("labels", "Not supplied"))
    st.caption(
        f"Dataset label: {label}. This is dataset metadata, not a location "
        "identified from the image."
    )

    st.markdown("#### Spectral index maps")

    map_cols = st.columns(3)
    for column, name in zip(map_cols, ["NDVI", "NDWI", "NDBI"]):
        with column:
            st.pyplot(
                make_index_figure(name, indices[name]),
                use_container_width=True,
            )
            st.caption(
                {
                    "NDVI": "Vegetation-related spectral contrast",
                    "NDWI": "Water-related spectral contrast",
                    "NDBI": "Built-up-related spectral contrast",
                }[name]
            )

    with st.expander("Inspect all extracted features"):
        display_feature_table(features)

    st.download_button(
        "Download extracted features as CSV",
        data=pd.DataFrame(
            [features],
            columns=[
                *[f"band_{i + 1}_mean" for i in range(cube.shape[-1])],
                "ndvi_mean", "ndvi_std",
                "ndwi_mean", "ndwi_std",
                "ndbi_mean", "ndbi_std",
            ],
        ).to_csv(index=False).encode("utf-8"),
        file_name="orbital_spectral_features.csv",
        mime="text/csv",
    )

except Exception as exc:
    st.error("The EuroSAT multispectral lab could not load its sample.")
    st.write(
        "Check the dataset connection, dataset schema, and installed "
        "`datasets` package. NASA imagery and the artifact status are "
        "handled independently."
    )
    with st.expander("Multispectral loading details"):
        st.code(f"{type(exc).__name__}: {exc}")
    st.link_button("Open EuroSAT MSI dataset", EUROSAT_URL)


# ============================================================
# 03 — SPECTRAL TARGET EXPLORER
# ============================================================

st.divider()
st.markdown(
    '<div class="module">03 / OBSERVATORY MODULE</div>',
    unsafe_allow_html=True,
)
st.header("Spectral target explorer")
st.write(
    "Inspect the unsupervised feature groups saved by your research notebook."
)

if artifact is None:
    st.warning(
        "Load `orbital_models.pkl` to explore the saved spectral model."
    )

else:
    if region_model is not None and hasattr(region_model, "cluster_centers_"):
        centers = np.asarray(region_model.cluster_centers_)

        st.markdown("#### Stored cluster centres")

        fig, ax = plt.subplots(figsize=(12, 4))
        image = ax.imshow(centers, aspect="auto", cmap="viridis")
        ax.set_xlabel("Feature index")
        ax.set_ylabel("Cluster ID")
        ax.set_title("Cluster centres in stored feature space")
        fig.colorbar(image, ax=ax, label="Stored feature value")
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)

        st.caption(
            "These are feature-space clusters, not geographic regions. "
            "A cluster ID does not identify a city, forest, or water body."
        )

        selected_cluster = st.selectbox(
            "Inspect a cluster centre",
            options=list(range(len(centers))),
        )

        st.dataframe(
            pd.DataFrame(
                {
                    "Feature index": np.arange(centers.shape[1]),
                    "Centre value": centers[selected_cluster],
                }
            ),
            use_container_width=True,
            hide_index=True,
        )

        st.download_button(
            "Download cluster centres",
            data=pd.DataFrame(centers).to_csv(index=False).encode("utf-8"),
            file_name="orbital_cluster_centres.csv",
            mime="text/csv",
        )

    else:
        st.info(
            "A cluster model with `cluster_centers_` was not found in the "
            "artifact. Verify the saved model key and artifact structure."
        )

    if distance_matrix is not None:
        try:
            distances = np.asarray(distance_matrix, dtype=float)

            if distances.ndim == 2:
                st.markdown("#### Stored target-distance matrix")

                fig, ax = plt.subplots(figsize=(7, 5))
                plot = ax.imshow(distances, cmap="magma", aspect="auto")
                ax.set_xlabel("Target index")
                ax.set_ylabel("Target index")
                ax.set_title("Stored pairwise distances")
                fig.colorbar(plot, ax=ax)
                fig.tight_layout()
                st.pyplot(fig, use_container_width=True)

        except Exception as exc:
            st.warning(f"Could not display the distance matrix: {exc}")


# ============================================================
# 04 — REINFORCEMENT-LEARNING RESEARCH
# ============================================================

st.divider()
st.markdown(
    '<div class="module">04 / OBSERVATORY MODULE</div>',
    unsafe_allow_html=True,
)
st.header("Reinforcement-learning research")
st.write(
    "Inspect the saved agents and understand the observation-planning "
    "experiment."
)

st.markdown("### What is the agent learning?")
st.write(
    "Under a limited observation budget, an agent chooses which spectral "
    "target to observe. A novelty-aware reward can favour observations "
    "that add information beyond what has already been seen."
)

rl_cols = st.columns(2)

with rl_cols[0]:
    st.markdown("#### Q-learning")
    st.metric("Stored states", len(q_learning or {}))
    st.caption("Off-policy temporal-difference learning.")

with rl_cols[1]:
    st.markdown("#### SARSA")
    st.metric("Stored states", len(sarsa or {}))
    st.caption("On-policy temporal-difference learning.")

st.markdown(
    """
    <div class="panel">
        <strong>Evaluation status</strong><br><br>
        The saved Q-learning and SARSA tables are loaded, but the original
        <code>OrbitalEnvV2</code> state encoding, action selection, transition
        rules, and reward function must be restored before new missions can
        be reported as evaluations of these trained agents.<br><br>
        This dashboard does not fabricate learned-policy performance.
    </div>
    """,
    unsafe_allow_html=True,
)

with st.expander("Inspect artifact keys"):
    if isinstance(artifact, dict):
        st.write("Saved keys:")
        st.code("\n".join(str(key) for key in artifact.keys()))

        st.write("Q-learning table type:", type(q_learning).__name__)
        st.write("SARSA table type:", type(sarsa).__name__)

        if isinstance(q_learning, dict) and q_learning:
            st.write("Example Q-learning state:")
            st.code(repr(next(iter(q_learning.keys())))[:1000])

        if isinstance(sarsa, dict) and sarsa:
            st.write("Example SARSA state:")
            st.code(repr(next(iter(sarsa.keys())))[:1000])


# ============================================================
# 05 — METHODS AND PROVENANCE
# ============================================================

st.divider()
st.markdown(
    '<div class="module">05 / OBSERVATORY MODULE</div>',
    unsafe_allow_html=True,
)
st.header("Methods and data provenance")

with st.expander("Project methodology and limitations", expanded=True):
    st.markdown(
        """
        **NASA imagery**
        - Global satellite visualization is requested from NASA GIBS.
        - NASA Worldview provides a separate interactive Earth-observation viewer.
        - NASA imagery is not automatically fed into the EuroSAT-trained clustering model.

        **EuroSAT MSI**
        - Multispectral samples come from `blanchon/EuroSAT_MSI`.
        - The feature pipeline uses 13 band means and six statistics from NDVI, NDWI, and NDBI.
        - Index maps are spectral indicators, not verified land-cover labels.

        **Spectral clustering**
        - Groups represent similarity in the stored feature space.
        - Cluster IDs are arbitrary and do not encode geographic coordinates.

        **Reinforcement learning**
        - Saved Q-learning and SARSA tables are loaded from `orbital_models.pkl`.
        - Faithful policy evaluation requires the exact training environment, state representation, transition rules, and reward function.
        - This prototype is not an operational satellite controller.
        """
    )

st.caption(
    "NASA GIBS imagery services are provided through NASA's Earth Science "
    "Data and Information System. EuroSAT and RL experiments are separate "
    "components of this research prototype."
)

st.markdown("---")
st.markdown(
    '<div class="eyebrow">ORBITAL · EARTH OBSERVATION · REMOTE SENSING · AI</div>',
    unsafe_allow_html=True,
)
