
from datetime import date, timedelta
from pathlib import Path
import io
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import streamlit as st
from PIL import Image, ImageOps


# ============================================================
# ORBITAL
# Reinforcement Learning for Information-Efficient Earth Observation
# Research prototype — not operational satellite guidance
# ============================================================

st.set_page_config(
    page_title="ORBITAL | Earth Intelligence",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = Path(__file__).resolve().parent
NASA_WMS = "https://gibs.earthdata.nasa.gov/wms/epsg4326/best/wms.cgi"
NASA_WORLDVIEW = "https://worldview.earthdata.nasa.gov/"
NASA_EARTHDATA = "https://www.earthdata.nasa.gov/"
EUROSAT_URL = "https://huggingface.co/datasets/blanchon/EuroSAT_MSI"

NASA_LAYERS = {
    "VIIRS — Suomi NPP True Color": "VIIRS_SNPP_CorrectedReflectance_TrueColor",
    "MODIS — Terra True Color": "MODIS_Terra_CorrectedReflectance_TrueColor",
}

EPS = 1e-8


# ============================================================
# STYLING
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
            radial-gradient(ellipse at 15% 0%, #122b43 0%, transparent 38%),
            linear-gradient(180deg, #07111e 0%, #091522 100%);
        color: #e7f1fa;
    }

    section[data-testid="stSidebar"] {
        background: #0a1624;
        border-right: 1px solid #21364a;
    }

    .orbital-hero {
        padding: 30px 30px 26px 30px;
        border: 1px solid #28445b;
        border-radius: 20px;
        background: linear-gradient(130deg, #10283d, #0b1928 65%, #122b37);
        margin-bottom: 22px;
    }

    .orbital-eyebrow {
        color: #78d9c3;
        font-family: 'DM Mono', monospace;
        font-size: 12px;
        letter-spacing: 2px;
        text-transform: uppercase;
    }

    .orbital-title {
        font-size: clamp(38px, 6vw, 68px);
        line-height: 1;
        font-weight: 700;
        letter-spacing: -3px;
        margin: 14px 0;
        color: #f3f8ff;
    }

    .orbital-subtitle {
        color: #a8bfd1;
        font-size: 17px;
        max-width: 780px;
        line-height: 1.65;
    }

    .orbital-card {
        border: 1px solid #223b50;
        border-radius: 14px;
        padding: 18px;
        background: rgba(12, 29, 45, 0.8);
        min-height: 120px;
    }

    .orbital-label {
        color: #8caabd;
        font-size: 11px;
        letter-spacing: 1.4px;
        text-transform: uppercase;
        font-family: 'DM Mono', monospace;
    }

    .orbital-value {
        color: #e9f6ff;
        font-size: 25px;
        font-weight: 700;
        margin-top: 9px;
    }

    .orbital-note {
        color: #91a9ba;
        font-size: 12px;
        margin-top: 5px;
    }

    h1, h2, h3 {
        color: #eaf4ff !important;
    }

    p, li, label {
        color: #c5d6e3;
    }

    div[data-testid="stMetric"] {
        background: #102235;
        padding: 14px;
        border: 1px solid #223b50;
        border-radius: 12px;
    }

    div[data-testid="stMetricValue"] {
        color: #7ce0ca;
    }

    .stButton > button,
    .stLinkButton > a {
        border-radius: 10px;
        font-weight: 600;
    }

    a {
        color: #79dbc8 !important;
    }

    code, .orbital-mono {
        font-family: 'DM Mono', monospace;
    }

    hr {
        border-color: #223b50;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# MODEL ARTIFACT
# ============================================================

def load_model_artifact():
    candidates = [
        BASE_DIR / "orbital_models.pkl",
        BASE_DIR / "orbital_models.pickle",
        BASE_DIR / "orbital_models" / "orbital_models.pkl",
        BASE_DIR / "orbital_models" / "orbital_models.pickle",
    ]

    for path in candidates:
        if path.exists():
            try:
                with path.open("rb") as file:
                    artifact = pickle.load(file)

                if isinstance(artifact, dict):
                    return artifact, path, None

                return {}, path, "Artifact exists but is not a dictionary."

            except Exception as exc:
                return {}, path, f"{type(exc).__name__}: {exc}"

    return {}, None, "No model artifact found in the expected locations."


def get_field(artifact, names, default=None):
    for name in names:
        if name in artifact and artifact[name] is not None:
            return artifact[name]
    return default


artifact, artifact_path, artifact_error = load_model_artifact()

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

cluster_model = get_field(
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
# IMAGE UTILITIES
# ============================================================

def validate_image_response(response):
    response.raise_for_status()

    content_type = response.headers.get("Content-Type", "").lower()

    if not content_type.startswith("image/"):
        raise ValueError(
            f"Expected an image, received Content-Type: {content_type or 'unknown'}"
        )

    image = Image.open(io.BytesIO(response.content))
    image.load()

    if image.width < 100 or image.height < 100:
        raise ValueError(
            f"NASA returned an unexpectedly small image: "
            f"{image.width} × {image.height}"
        )

    return image.convert("RGB")


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_nasa_image(requested_layer, selected_date):
    """
    Fetch a global NASA GIBS image, trying alternative layers
    and recent dates if the first request fails.
    """
    preferred_layer = NASA_LAYERS.get(
        requested_layer,
        "VIIRS_SNPP_CorrectedReflectance_TrueColor",
    )

    alternate_layer = (
        "MODIS_Terra_CorrectedReflectance_TrueColor"
        if "VIIRS" in preferred_layer
        else "VIIRS_SNPP_CorrectedReflectance_TrueColor"
    )

    requested_day = date.fromisoformat(str(selected_date))

    attempts = [
        (preferred_layer, requested_day),
        (alternate_layer, requested_day),
        (preferred_layer, requested_day - timedelta(days=1)),
        (alternate_layer, requested_day - timedelta(days=1)),
        (preferred_layer, requested_day - timedelta(days=2)),
        (alternate_layer, requested_day - timedelta(days=2)),
    ]

    errors = []

    for layer, image_day in attempts:
        if image_day < date(2002, 1, 1):
            continue

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
            "TRANSPARENT": "FALSE",
            "TIME": image_day.isoformat(),
        }

        try:
            response = requests.get(
                NASA_WMS,
                params=params,
                timeout=30,
                headers={"User-Agent": "ORBITAL-Earth-Observation-Research/1.0"},
            )

            image = validate_image_response(response)

            fallback_used = (
                layer != preferred_layer or image_day != requested_day
            )

            return (
                image,
                layer,
                image_day.isoformat(),
                fallback_used,
            )

        except Exception as exc:
            errors.append(
                f"{layer} on {image_day.isoformat()}: "
                f"{type(exc).__name__}: {exc}"
            )

    details = "\n".join(errors[-6:])

    raise RuntimeError(
        "NASA GIBS did not return a valid image after several attempts.\n"
        "This may be a temporary service, network, or layer-availability issue.\n\n"
        + details
    )


def image_hwc(value):
    """Convert a sample's multispectral image to H × W × bands."""
    if isinstance(value, Image.Image):
        arr = np.asarray(value)
    else:
        arr = np.asarray(value)

    arr = np.squeeze(arr)

    if arr.ndim != 3:
        raise ValueError(
            f"Expected a 3D multispectral image, received shape {arr.shape}"
        )

    # Convert bands × height × width to height × width × bands
    if arr.shape[0] <= 20 and arr.shape[-1] > 20:
        arr = np.moveaxis(arr, 0, -1)

    if arr.shape[-1] < 4:
        raise ValueError(
            f"Expected multiple spectral bands, received shape {arr.shape}"
        )

    return arr.astype(np.float32)


def stretch(array):
    """Percentile stretch each channel for visual display."""
    array = np.asarray(array, dtype=np.float32)
    output = np.zeros_like(array, dtype=np.float32)

    if array.ndim == 2:
        low, high = np.nanpercentile(array, [2, 98])
        if high <= low:
            return np.zeros_like(array)
        return np.clip((array - low) / (high - low), 0, 1)

    for band in range(array.shape[-1]):
        channel = array[..., band]
        low, high = np.nanpercentile(channel, [2, 98])

        if high > low:
            output[..., band] = np.clip(
                (channel - low) / (high - low),
                0,
                1,
            )

    return output


def make_composite(cube, band_indices):
    indices = [
        min(max(int(i), 0), cube.shape[-1] - 1)
        for i in band_indices
    ]

    rgb = cube[..., indices]
    return stretch(rgb)


def calculate_indices(cube):
    """
    Expected EuroSAT MSI band ordering:
    green=2, red=3, NIR=7, SWIR=11.
    Confirm the dataset's band ordering before interpreting indices
    scientifically.
    """
    bands = cube.shape[-1]

    if bands < 12:
        raise ValueError(
            f"These index settings need at least 12 bands; found {bands}."
        )

    green = cube[..., 2]
    red = cube[..., 3]
    nir = cube[..., 7]
    swir = cube[..., 11]

    ndvi = (nir - red) / (nir + red + EPS)
    ndwi = (green - nir) / (green + nir + EPS)
    ndbi = (swir - nir) / (swir + nir + EPS)

    return {
        "NDVI": ndvi,
        "NDWI": ndwi,
        "NDBI": ndbi,
    }


def calculate_features(cube):
    """Create 19 features: 13 band means and 6 index statistics."""
    band_means = np.nanmean(cube, axis=(0, 1))

    indices = calculate_indices(cube)

    features = list(band_means)

    for index_name in ["NDVI", "NDWI", "NDBI"]:
        index = indices[index_name]
        features.extend([
            float(np.nanmean(index)),
            float(np.nanstd(index)),
        ])

    return np.asarray(features, dtype=np.float32), indices


def make_index_figure(index, title):
    fig, ax = plt.subplots(figsize=(7, 4))
    im = ax.imshow(index, cmap="RdYlGn", vmin=-1, vmax=1)
    ax.set_title(title)
    ax.set_axis_off()
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    return fig


# ============================================================
# EUROSAT DATASET
# ============================================================

@st.cache_data(ttl=86400, show_spinner=False)
def load_eurosat_samples(limit=20):
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

    if not samples:
        raise RuntimeError("The dataset stream returned no samples.")

    return samples


def find_image(sample):
    for key in ["image", "img", "pixel_values"]:
        if key in sample and sample[key] is not None:
            return sample[key], key

    available = ", ".join(map(str, sample.keys()))

    raise KeyError(
        f"No image field was found. Available dataset fields: {available}"
    )


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.markdown("## 🌍 ORBITAL")
    st.caption("EARTH INTELLIGENCE / RESEARCH")

    st.divider()

    st.markdown("### Observation settings")

    today = date.today()
    latest_date = today - timedelta(days=1)

    default_date = min(date(2026, 10, 6), latest_date)
    default_date = max(default_date, date(2002, 1, 1))

    selected_date = st.date_input(
        "Requested imagery date",
        value=default_date,
        min_value=date(2002, 1, 1),
        max_value=latest_date,
        help="NASA GIBS may not provide every layer on every date.",
    )

    selected_layer = st.selectbox(
        "Satellite imagery layer",
        list(NASA_LAYERS.keys()),
    )

    st.divider()

    st.markdown("### Research artifacts")

    if artifact_path:
        st.success(f"Artifact detected: `{artifact_path.name}`")
    else:
        st.warning("No model artifact detected.")

    col_a, col_b = st.columns(2)

    with col_a:
        st.metric(
            "Q-learning states",
            len(q_learning) if hasattr(q_learning, "__len__") else 0,
        )

    with col_b:
        st.metric(
            "SARSA states",
            len(sarsa) if hasattr(sarsa, "__len__") else 0,
        )

    st.divider()

    st.markdown("### External resources")
    st.markdown(f"[NASA Worldview ↗]({NASA_WORLDVIEW})")
    st.markdown(f"[EuroSAT MSI ↗]({EUROSAT_URL})")
    st.markdown(f"[NASA Earthdata ↗]({NASA_EARTHDATA})")

    st.caption("Research prototype — not operational satellite guidance.")


# ============================================================
# HERO
# ============================================================

st.markdown(
    """
    <div class="orbital-hero">
        <div class="orbital-eyebrow">
            Earth observation / Reinforcement learning / Spectral intelligence
        </div>
        <div class="orbital-title">ORBITAL</div>
        <div class="orbital-subtitle">
            Reinforcement learning for information-efficient Earth observation.
            Explore satellite imagery, multispectral signatures, spectral
            clustering, and a research prototype for observation policies.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

hero_a, hero_b, hero_c, hero_d = st.columns(4)

with hero_a:
    st.markdown(
        """
        <div class="orbital-card">
            <div class="orbital-label">Research domain</div>
            <div class="orbital-value">Earth</div>
            <div class="orbital-note">Remote sensing</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with hero_b:
    st.markdown(
        """
        <div class="orbital-card">
            <div class="orbital-label">Learning paradigm</div>
            <div class="orbital-value">RL</div>
            <div class="orbital-note">Q-learning / SARSA</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with hero_c:
    st.markdown(
        """
        <div class="orbital-card">
            <div class="orbital-label">Spectral analysis</div>
            <div class="orbital-value">MSI</div>
            <div class="orbital-note">Multispectral imagery</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with hero_d:
    st.markdown(
        """
        <div class="orbital-card">
            <div class="orbital-label">System status</div>
            <div class="orbital-value">R&D</div>
            <div class="orbital-note">Non-operational prototype</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# SECTION 1 — NASA SATELLITE IMAGERY
# ============================================================

st.divider()
st.header("01 / Earth from orbit")
st.write(
    "Retrieve a global satellite-image layer from NASA GIBS, "
    "or open NASA's full interactive map in a separate browser tab."
)

tab_satellite, tab_worldview = st.tabs([
    "Satellite image",
    "Interactive NASA Worldview",
])


with tab_satellite:
    st.markdown("### Global satellite imagery")

    st.caption(
        "The image is retrieved from NASA GIBS. It is a global map image, "
        "not a live video feed."
    )

    load_button = st.button(
        "↻ Load / refresh satellite image",
        key="load_nasa",
        type="primary",
    )

    if "orbital_nasa_image" not in st.session_state:
        st.session_state["orbital_nasa_image"] = None

    if load_button:
        st.session_state["orbital_nasa_image"] = None
        st.cache_data.clear()

    if (
        st.session_state["orbital_nasa_image"] is None
        or load_button
    ):
        with st.spinner("Requesting imagery from NASA GIBS..."):
            try:
                result = fetch_nasa_image(
                    selected_layer,
                    selected_date.isoformat(),
                )
                st.session_state["orbital_nasa_image"] = result
                st.session_state["orbital_nasa_error"] = None

            except Exception as exc:
                st.session_state["orbital_nasa_error"] = str(exc)

    nasa_result = st.session_state.get("orbital_nasa_image")

    if nasa_result:
        nasa_image, actual_layer, actual_date, fallback_used = nasa_result

        st.image(
            nasa_image,
            use_container_width=True,
            caption=f"NASA GIBS · {actual_layer} · {actual_date}",
        )

        if fallback_used:
            st.info(
                f"The requested layer/date was unavailable. "
                f"Showing the available fallback: {actual_date}."
            )

        image_buffer = io.BytesIO()
        nasa_image.save(image_buffer, format="JPEG")

        st.download_button(
            "Download satellite image",
            data=image_buffer.getvalue(),
            file_name=f"orbital_nasa_{actual_date}.jpg",
            mime="image/jpeg",
        )

        st.caption(
            "Source: NASA Global Imagery Browse Services (GIBS). "
            "Check the selected layer and date before scientific use."
        )

    else:
        error_message = st.session_state.get(
            "orbital_nasa_error",
            "No satellite image has been loaded.",
        )

        st.warning(
            "NASA imagery could not be loaded automatically. "
            "The rest of the research dashboard can still be used."
        )

        with st.expander("Technical details"):
            st.code(str(error_message))

        st.link_button(
            "Open NASA Worldview instead ↗",
            NASA_WORLDVIEW,
            use_container_width=True,
        )


with tab_worldview:
    st.markdown("### NASA Worldview — interactive map")

    st.write(
        "NASA Worldview offers interactive satellite imagery and "
        "time controls. Its full map may not work inside a Streamlit "
        "embedded frame, so this dashboard opens it directly."
    )

    st.link_button(
        "🌍 OPEN NASA WORLDVIEW",
        NASA_WORLDVIEW,
        use_container_width=True,
    )

    st.info(
        "Click the button above to open NASA Worldview in a separate "
        "browser tab. The map's interactive scripts and controls run "
        "on NASA's own website rather than inside this dashboard."
    )

    with st.expander("What can I explore there?"):
        st.markdown(
            """
            - Global satellite imagery and daily observations
            - Clouds, smoke, fires, and environmental events where layers are available
            - Time controls for viewing imagery from different dates
            - Multiple NASA Earth-observation data layers
            """
        )

    st.caption(
        "If NASA Worldview does not load in a separate tab either, "
        "the cause may be browser, network, or NASA service availability."
    )


# ============================================================
# SECTION 2 — MULTISPECTRAL ANALYSIS
# ============================================================

st.divider()
st.header("02 / Multispectral analysis")

st.write(
    "Inspect multispectral samples from EuroSAT MSI and derive "
    "simple spectral indices for exploratory analysis."
)

st.markdown(
    f"Dataset: [EuroSAT MSI on Hugging Face]({EUROSAT_URL})"
)

if st.button("Load multispectral samples", key="load_eurosat"):
    st.session_state["orbital_eurosat_samples"] = None
    st.session_state["orbital_eurosat_error"] = None

if "orbital_eurosat_samples" not in st.session_state:
    st.session_state["orbital_eurosat_samples"] = None

if (
    st.session_state["orbital_eurosat_samples"] is None
    and st.session_state.get("orbital_eurosat_error") is None
):
    with st.spinner("Loading EuroSAT MSI samples..."):
        try:
            st.session_state["orbital_eurosat_samples"] = (
                load_eurosat_samples(limit=20)
            )
        except Exception as exc:
            st.session_state["orbital_eurosat_error"] = (
                f"{type(exc).__name__}: {exc}"
            )

samples = st.session_state.get("orbital_eurosat_samples")
eurosat_error = st.session_state.get("orbital_eurosat_error")

if eurosat_error:
    st.error("The multispectral dataset could not be loaded.")
    with st.expander("Dataset error details"):
        st.code(eurosat_error)

    st.link_button(
        "Open the dataset page ↗",
        EUROSAT_URL,
    )

elif samples:
    sample_labels = [
        f"Sample {i + 1}"
        for i in range(len(samples))
    ]

    chosen_sample_index = st.selectbox(
        "Choose a sample",
        range(len(samples)),
        format_func=lambda i: sample_labels[i],
    )

    sample = samples[chosen_sample_index]

    try:
        raw_image, image_key = find_image(sample)
        cube = image_hwc(raw_image)
        features, indices = calculate_features(cube)

        label_value = sample.get(
            "label",
            sample.get("labels", "Not provided"),
        )

        st.caption(
            f"Image field: `{image_key}` · "
            f"Cube shape: `{cube.shape}` · "
            f"Dataset label: `{label_value}`"
        )

        st.markdown("#### Spectral composites")

        composite_a, composite_b, composite_c = st.tabs([
            "Natural-color approximation",
            "False-color vegetation",
            "Shortwave infrared",
        ])

        with composite_a:
            rgb = make_composite(cube, [3, 2, 1])
            st.image(
                rgb,
                use_container_width=True,
                caption="Approximate RGB composite; band mapping must be verified.",
            )

        with composite_b:
            rgb = make_composite(cube, [7, 3, 2])
            st.image(
                rgb,
                use_container_width=True,
                caption="False-color composite using NIR, red, and green positions.",
            )

        with composite_c:
            if cube.shape[-1] > 11:
                rgb = make_composite(cube, [11, 7, 3])
                st.image(
                    rgb,
                    use_container_width=True,
                    caption="SWIR/NIR/red composite.",
                )
            else:
                st.warning("This sample does not contain the expected SWIR band.")

        st.markdown("#### Spectral indices")

        m1, m2, m3 = st.columns(3)

        m1.metric("Mean NDVI", f"{np.nanmean(indices['NDVI']):.3f}")
        m2.metric("Mean NDWI", f"{np.nanmean(indices['NDWI']):.3f}")
        m3.metric("Mean NDBI", f"{np.nanmean(indices['NDBI']):.3f}")

        idx_a, idx_b, idx_c = st.tabs(["NDVI", "NDWI", "NDBI"])

        with idx_a:
            fig = make_index_figure(indices["NDVI"], "NDVI")
            st.pyplot(fig)
            plt.close(fig)

        with idx_b:
            fig = make_index_figure(indices["NDWI"], "NDWI")
            st.pyplot(fig)
            plt.close(fig)

        with idx_c:
            fig = make_index_figure(indices["NDBI"], "NDBI")
            st.pyplot(fig)
            plt.close(fig)

        st.markdown("#### Extracted spectral features")

        band_count = cube.shape[-1]
        feature_names = [
            f"band_{i + 1}_mean"
            for i in range(band_count)
        ]

        for index_name in ["NDVI", "NDWI", "NDBI"]:
            feature_names.extend([
                f"{index_name.lower()}_mean",
                f"{index_name.lower()}_std",
            ])

        feature_table = pd.DataFrame({
            "Feature": feature_names,
            "Value": features,
        })

        st.dataframe(
            feature_table,
            use_container_width=True,
            hide_index=True,
        )

        csv_data = feature_table.to_csv(index=False).encode("utf-8")

        st.download_button(
            "Download extracted features (CSV)",
            data=csv_data,
            file_name=f"orbital_features_sample_{chosen_sample_index + 1}.csv",
            mime="text/csv",
        )

        st.caption(
            "Exploratory features only. Confirm the dataset's band order, "
            "scaling, and radiometric meaning before interpreting indices "
            "as scientific measurements."
        )

    except Exception as exc:
        st.error("This sample could not be processed.")
        with st.expander("Sample processing error"):
            st.code(f"{type(exc).__name__}: {exc}")

else:
    st.info("Load the dataset to explore multispectral images.")


# ============================================================
# SECTION 3 — SPECTRAL TARGET EXPLORER
# ============================================================

st.divider()
st.header("03 / Spectral target explorer")

st.write(
    "Inspect the spectral clustering model included in the saved "
    "artifact. Clusters are unsupervised spectral groupings; they are "
    "not automatically verified land-cover categories."
)

if cluster_model is not None and hasattr(cluster_model, "cluster_centers_"):
    centers = np.asarray(cluster_model.cluster_centers_)

    st.metric("Spectral clusters", centers.shape[0])

    fig, ax = plt.subplots(figsize=(11, 4))

    for cluster_id, center in enumerate(centers):
        ax.plot(
            np.arange(1, len(center) + 1),
            center,
            marker=".",
            linewidth=1.4,
            label=f"Cluster {cluster_id}",
        )

    ax.set_xlabel("Feature index")
    ax.set_ylabel("Scaled feature value")
    ax.set_title("Spectral cluster centers")
    ax.grid(alpha=0.2)
    ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()

    st.pyplot(fig)
    plt.close(fig)

    selected_cluster = st.selectbox(
        "Inspect cluster",
        range(centers.shape[0]),
        format_func=lambda i: f"Cluster {i}",
    )

    st.dataframe(
        pd.DataFrame({
            "Feature index": np.arange(len(centers[selected_cluster])),
            "Center value": centers[selected_cluster],
        }),
        use_container_width=True,
        hide_index=True,
    )

else:
    st.info(
        "No compatible cluster-center model was found in the artifact. "
        "Check that `region_model` or `kmeans` was saved."
    )

if distance_matrix is not None:
    try:
        distances = np.asarray(distance_matrix, dtype=float)

        if distances.ndim == 2:
            st.markdown("#### Target-distance matrix")

            fig, ax = plt.subplots(figsize=(7, 5))
            image = ax.imshow(distances, cmap="viridis", aspect="auto")
            ax.set_xlabel("Target")
            ax.set_ylabel("Target")
            ax.set_title("Stored target-distance matrix")
            fig.colorbar(image, ax=ax, label="Distance")
            fig.tight_layout()

            st.pyplot(fig)
            plt.close(fig)

        else:
            st.caption("The stored distance matrix is not two-dimensional.")

    except Exception as exc:
        st.warning(f"Could not display distance matrix: {exc}")


# ============================================================
# SECTION 4 — REINFORCEMENT LEARNING RESEARCH
# ============================================================

st.divider()
st.header("04 / Reinforcement learning research")

st.write(
    "ORBITAL's research prototype explores observation selection "
    "under a simulated budget. The saved Q-tables alone do not establish "
    "real-world satellite performance."
)

rl_a, rl_b, rl_c = st.columns(3)

with rl_a:
    st.metric(
        "Q-learning states",
        len(q_learning) if hasattr(q_learning, "__len__") else 0,
    )

with rl_b:
    st.metric(
        "SARSA states",
        len(sarsa) if hasattr(sarsa, "__len__") else 0,
    )

with rl_c:
    st.metric(
        "Saved artifact",
        "Loaded" if artifact_path else "Missing",
    )

st.markdown("#### Artifact inspection")

if artifact:
    artifact_summary = []

    for key, value in artifact.items():
        try:
            if hasattr(value, "shape"):
                description = f"shape={value.shape}"
            elif hasattr(value, "__len__") and not isinstance(value, str):
                description = f"length={len(value)}"
            else:
                description = type(value).__name__

        except Exception:
            description = type(value).__name__

        artifact_summary.append({
            "Key": str(key),
            "Type": type(value).__name__,
            "Details": description,
        })

    st.dataframe(
        pd.DataFrame(artifact_summary),
        use_container_width=True,
        hide_index=True,
    )

else:
    st.warning(
        artifact_error or "No model artifact is available."
    )

st.info(
    "To report current policy rewards, comparison plots, or held-out "
    "performance, run the evaluation in the research notebook and load "
    "its actual outputs. This dashboard does not fabricate evaluation "
    "results or claim that simulated actions control a real satellite."
)


# ============================================================
# SECTION 5 — METHODS AND PROVENANCE
# ============================================================

st.divider()
st.header("05 / Methods and provenance")

with st.expander("Satellite imagery"):
    st.markdown(
        """
        NASA GIBS provides browse imagery through an image-service request.
        The dashboard tries alternative recent dates and supported layers
        when the initial request fails.

        Imagery is not a live satellite feed. Availability depends on the
        product, observation date, service status, and network access.
        """
    )
    st.markdown(f"[NASA GIBS documentation]({NASA_EARTHDATA})")

with st.expander("Multispectral feature extraction"):
    st.markdown(
        """
        The exploratory pipeline computes per-band means and summary
        statistics for NDVI, NDWI, and NDBI. These indices depend on correct
        band mapping and suitable input reflectance values.

        Display composites use percentile stretching for visualization.
        A visually attractive composite is not, by itself, a calibrated
        scientific product.
        """
    )
    st.markdown(f"[EuroSAT MSI dataset]({EUROSAT_URL})")

with st.expander("Reinforcement learning"):
    st.markdown(
        """
        Q-learning and SARSA are temporal-difference reinforcement learning
        methods. In this project, saved state-action values belong to a
        simulated observation-selection environment.

        Simulation results must be distinguished from actual satellite
        operations. Real operational guidance would require validated
        constraints, flight dynamics, telemetry, safety review, and
        mission-specific authorization.
        """
    )

st.divider()

st.caption(
    "ORBITAL · Research prototype · Earth observation and reinforcement learning"
)
st.caption(
    "External services and datasets are maintained by their respective providers. "
    "This application is not affiliated with NASA."
)
