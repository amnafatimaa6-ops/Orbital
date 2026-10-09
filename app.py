
from __future__ import annotations

import pickle
import warnings
from pathlib import Path
from datetime import datetime, timedelta
from io import BytesIO

import numpy as np
import pandas as pd
import requests
import streamlit as st
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

warnings.filterwarnings("ignore", category=FutureWarning)

# ============================================================
# ORBITAL — Earth Intelligence Research Prototype
# ============================================================

st.set_page_config(
    page_title="ORBITAL | Earth Intelligence",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded",
)

ROOT = Path(__file__).resolve().parent

ARTIFACT_CANDIDATES = [
    ROOT / "orbital_models.pkl",
    ROOT / "orbital_models" / "orbital_models.pkl",
]

NASA_WORLDVIEW = "https://worldview.earthdata.nasa.gov/"
EUROSAT_URL = "https://huggingface.co/datasets/blanchon/EuroSAT_MSI"

# GIBS imagery layers
NASA_LAYERS = [
    "MODIS_Terra_CorrectedReflectance_TrueColor",
    "VIIRS_SNPP_CorrectedReflectance_TrueColor",
]

# ============================================================
# Styling
# ============================================================

st.markdown(
    """
    <style>
    .stApp {
        background:
            radial-gradient(ellipse at top left,
                rgba(24, 73, 103, 0.24), transparent 42%),
            #08111d;
        color: #e8f0f8;
    }

    [data-testid="stSidebar"] {
        background: #0b1725;
        border-right: 1px solid #243448;
    }

    .orbital-kicker {
        color: #78c9e8;
        letter-spacing: 0.22em;
        font-size: 0.72rem;
        font-weight: 700;
    }

    .orbital-title {
        font-size: clamp(2.5rem, 6vw, 4.8rem);
        font-weight: 800;
        letter-spacing: 0.12em;
        line-height: 1.1;
        margin: 0.2rem 0;
    }

    .orbital-subtitle {
        color: #9fb2c7;
        letter-spacing: 0.13em;
        font-size: 0.8rem;
    }

    .orbital-panel {
        padding: 1rem 1.15rem;
        border: 1px solid #263a50;
        border-radius: 14px;
        background: rgba(15, 29, 45, 0.78);
        margin: 0.5rem 0 1rem 0;
    }

    .muted {
        color: #9fb2c7;
    }

    a {
        color: #80d8f4 !important;
    }

    div[data-testid="stMetric"] {
        background: rgba(17, 34, 52, 0.75);
        border: 1px solid #263a50;
        border-radius: 12px;
        padding: 0.9rem;
    }

    div[data-testid="stAlert"] {
        border-radius: 10px;
    }

    .stButton button {
        border-radius: 9px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# Utility functions
# ============================================================

def find_artifact() -> Path | None:
    for candidate in ARTIFACT_CANDIDATES:
        if candidate.is_file():
            return candidate
    return None


def first_value(mapping, names):
    """Return the first matching key from a dictionary."""
    if not isinstance(mapping, dict):
        return None

    for name in names:
        if name in mapping:
            return mapping[name]

    return None


def safe_count(value) -> int:
    if isinstance(value, dict):
        return len(value)

    if isinstance(value, (list, tuple)):
        return len(value)

    return 0


def numeric_matrix(value):
    """Return a numeric 2D array, or None if incompatible."""
    if value is None:
        return None

    try:
        array = np.asarray(value, dtype=float)
        if array.ndim == 2 and array.shape[0] > 0:
            if array.shape[1] > 0 and np.isfinite(array).all():
                return array
    except (TypeError, ValueError, OverflowError):
        pass

    return None


def load_artifact():
    """
    Load only the trusted project artifact.

    Pickle files can execute code while loading. Never load an
    artifact from an untrusted source.
    """
    path = find_artifact()

    if path is None:
        return {}, "Artifact file not found.", None

    try:
        with path.open("rb") as file:
            artifact = pickle.load(file)

        if not isinstance(artifact, dict):
            return (
                {},
                "The artifact loaded, but its top-level object is "
                f"{type(artifact).__name__}, not a dictionary.",
                path,
            )

        return artifact, "Artifact loaded successfully.", path

    except ModuleNotFoundError as exc:
        message = (
            f"{type(exc).__name__}: {exc}. "
            "Check NumPy and scikit-learn versions used when saving "
            "the artifact. Recreate the artifact in a compatible "
            "environment if necessary."
        )
        return {}, message, path

    except (ImportError, AttributeError, ValueError, TypeError,
            pickle.UnpicklingError, EOFError, OSError) as exc:
        message = f"{type(exc).__name__}: {exc}"
        return {}, message, path

    except Exception as exc:
        message = f"{type(exc).__name__}: {exc}"
        return {}, message, path


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_nasa_image(date_string: str, layer: str):
    """Fetch a NASA GIBS browse image."""
    endpoint = (
        "https://gibs.earthdata.nasa.gov/wms/epsg4326/best/wms.cgi"
    )

    params = {
        "SERVICE": "WMS",
        "REQUEST": "GetMap",
        "VERSION": "1.1.1",
        "LAYERS": layer,
        "STYLES": "",
        "FORMAT": "image/jpeg",
        "SRS": "EPSG:4326",
        "BBOX": "-180,-90,180,90",
        "WIDTH": 1200,
        "HEIGHT": 600,
        "TIME": date_string,
    }

    response = requests.get(
        endpoint,
        params=params,
        timeout=25,
        headers={"User-Agent": "ORBITAL-Earth-Research/1.0"},
    )
    response.raise_for_status()

    image = Image.open(BytesIO(response.content)).convert("RGB")
    return image


def get_q_table(artifact):
    return first_value(
        artifact,
        ["q_table", "q_table_dict", "q_values", "Q_table", "Q"],
    )


def get_sarsa_table(artifact):
    return first_value(
        artifact,
        ["sarsa_table", "sarsa_q_table", "sarsa_values",
         "SARSA_table", "SARSA"],
    )


def get_cluster_model(artifact):
    return first_value(
        artifact,
        ["kmeans", "kmeans_model", "cluster_model",
         "clustering_model", "model", "kmeans_clustering"],
    )


def get_distance_matrix(artifact):
    return numeric_matrix(
        first_value(
            artifact,
            ["distance_matrix", "dist_matrix", "target_distances",
             "distances", "distance_mat"],
        )
    )


def get_scaler(artifact):
    return first_value(
        artifact,
        ["scaler", "standard_scaler", "feature_scaler"],
    )


def describe_model(model):
    if model is None:
        return "Not found"

    return type(model).__name__


def calculate_indices(cube):
    """
    Estimate spectral indices from a channel-last multispectral cube.

    Band positions must be verified against dataset metadata before
    treating these estimates as scientific measurements.
    """
    cube = np.asarray(cube, dtype=np.float32)

    if cube.ndim != 3 or cube.shape[-1] < 4:
        raise ValueError("Expected an H × W × bands multispectral cube.")

    # These positions are provisional. Verify channel ordering
    # for the exact dataset configuration before scientific use.
    red = cube[..., 3]
    green = cube[..., 2]
    nir = cube[..., 7]
    swir = cube[..., 11] if cube.shape[-1] > 11 else cube[..., -1]

    def normalized_difference(a, b):
        denominator = a + b
        return np.divide(
            a - b,
            denominator,
            out=np.zeros_like(a, dtype=np.float32),
            where=np.abs(denominator) > 1e-8,
        )

    ndvi = normalized_difference(nir, red)
    ndwi = normalized_difference(green, nir)
    ndbi = normalized_difference(swir, nir)

    return {
        "NDVI": ndvi,
        "NDWI": ndwi,
        "NDBI": ndbi,
    }


def run_simulation(distance_matrix, policy, steps=40, seed=42):
    """
    Simple educational simulator over a distance matrix.
    This is not satellite flight-control software.
    """
    matrix = numeric_matrix(distance_matrix)

    if matrix is None or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("A valid square distance matrix is required.")

    n_targets = matrix.shape[0]
    if n_targets < 2:
        raise ValueError("At least two targets are required.")

    rng = np.random.default_rng(seed)
    current = 0
    visited = {current}
    route = [current]
    total_distance = 0.0
    rewards = []

    q_table = policy if isinstance(policy, dict) else {}

    for step in range(steps):
        candidates = [
            i for i in range(n_targets)
            if i != current
        ]

        if not candidates:
            break

        if isinstance(q_table, dict) and current in q_table:
            try:
                values = np.asarray(q_table[current], dtype=float).ravel()
                if len(values) == n_targets:
                    action = max(
                        candidates,
                        key=lambda idx: values[idx],
                    )
                else:
                    action = min(
                        candidates,
                        key=lambda idx: matrix[current, idx],
                    )
            except (ValueError, TypeError, IndexError):
                action = min(
                    candidates,
                    key=lambda idx: matrix[current, idx],
                )
        elif policy == "random":
            action = int(rng.choice(candidates))
        elif policy == "greedy":
            action = min(
                candidates,
                key=lambda idx: matrix[current, idx],
            )
        else:
            action = min(
                candidates,
                key=lambda idx: matrix[current, idx],
            )

        distance = float(matrix[current, action])
        total_distance += distance
        reward = -distance
        if action not in visited:
            reward += 1.0

        rewards.append(reward)
        current = action
        route.append(current)
        visited.add(current)

    return {
        "route": route,
        "total_distance": total_distance,
        "unique_targets": len(visited),
        "rewards": rewards,
    }


# ============================================================
# Load artifact once per app session
# ============================================================

if "artifact_loaded" not in st.session_state:
    (
        st.session_state.artifact,
        st.session_state.artifact_message,
        st.session_state.artifact_path,
    ) = load_artifact()
    st.session_state.artifact_loaded = True

artifact = st.session_state.artifact
artifact_message = st.session_state.artifact_message
artifact_path = st.session_state.artifact_path

q_table = get_q_table(artifact)
sarsa_table = get_sarsa_table(artifact)
cluster_model = get_cluster_model(artifact)
distance_matrix = get_distance_matrix(artifact)
scaler = get_scaler(artifact)

# ============================================================
# Header and sidebar
# ============================================================

st.markdown(
    '<div class="orbital-kicker">EARTH INTELLIGENCE / RESEARCH PROTOTYPE</div>',
    unsafe_allow_html=True,
)
st.markdown('<div class="orbital-title">🌍 ORBITAL</div>',
            unsafe_allow_html=True)
st.markdown(
    '<div class="orbital-subtitle">'
    'EARTH OBSERVATION · REMOTE SENSING · REINFORCEMENT LEARNING'
    '</div>',
    unsafe_allow_html=True,
)

st.divider()

with st.sidebar:
    st.markdown("## 🌍 ORBITAL")
    st.caption("Earth Intelligence / Research Prototype")

    page = st.radio(
        "Navigation",
        [
            "Overview",
            "Earth from Orbit",
            "Spectral Lab",
            "Target Explorer",
            "Mission Simulator",
            "Policy Benchmark",
            "Methods",
        ],
        label_visibility="collapsed",
    )

    st.divider()
    st.markdown("### Data sources")
    st.markdown(f"[NASA Worldview ↗]({NASA_WORLDVIEW})")
    st.markdown(f"[EuroSAT MSI ↗]({EUROSAT_URL})")
    st.caption("Not affiliated with NASA.")

    st.divider()
    st.markdown("### Saved models")
    st.caption(
        "Artifact: "
        + (str(artifact_path.relative_to(ROOT))
           if artifact_path is not None else "Not found")
    )

    if st.button("Reload artifact", use_container_width=True):
        st.session_state.pop("artifact_loaded", None)
        st.rerun()

# ============================================================
# Overview
# ============================================================

if page == "Overview":
    st.markdown("## Mission overview")
    st.write(
        "Explore satellite browse imagery, inspect multispectral "
        "measurements, examine feature-space clusters, and evaluate "
        "tabular agents in a simulated observation environment."
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Q states", safe_count(q_table))
    c2.metric("SARSA states", safe_count(sarsa_table))
    c3.metric(
        "Clustering model",
        "Loaded" if cluster_model is not None else "Unavailable",
    )
    c4.metric(
        "Distance matrix",
        "Ready" if distance_matrix is not None else "Unavailable",
    )

    st.markdown("### Saved-model diagnostics")
    if artifact_path is not None:
        st.caption(f"Artifact path: `{artifact_path}`")
    else:
        st.warning(
            "No orbital_models.pkl file was found. Expected it in the "
            "repository root or the orbital_models/ folder."
        )

    if artifact_message == "Artifact loaded successfully.":
        st.success(artifact_message)
    else:
        st.error(artifact_message)

    with st.expander("Artifact keys and detected types"):
        if artifact:
            for key, value in artifact.items():
                st.write(f"**{key}** — `{type(value).__name__}`")
        else:
            st.info(
                "No artifact objects are available. Resolve the loading "
                "error before using saved-model features."
            )

    st.markdown("### Research modules")
    left, right = st.columns(2)

    with left:
        st.markdown(
            """
            <div class="orbital-panel">
                <h4>🛰️ Earth from Orbit</h4>
                <p class="muted">NASA GIBS browse imagery and recent dates.</p>
            </div>
            <div class="orbital-panel">
                <h4>🧪 Spectral Lab</h4>
                <p class="muted">Multispectral observations and exploratory indices.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with right:
        st.markdown(
            """
            <div class="orbital-panel">
                <h4>🎯 Target Explorer</h4>
                <p class="muted">Explore saved clustering models when compatible.</p>
            </div>
            <div class="orbital-panel">
                <h4>🤖 Policy Benchmark</h4>
                <p class="muted">Compare policies in a simulated environment.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

# ============================================================
# Earth from Orbit
# ============================================================

elif page == "Earth from Orbit":
    st.markdown("## Satellite imagery")
    st.caption("NASA GIBS browse imagery. Availability varies by layer and date.")

    default_date = datetime.utcnow().date() - timedelta(days=3)
    chosen_date = st.date_input(
        "Observation date",
        value=default_date,
        min_value=datetime(2012, 1, 1).date(),
        max_value=datetime.utcnow().date(),
    )
    date_string = chosen_date.strftime("%Y-%m-%d")

    selected_layer = st.selectbox(
        "Imagery layer",
        NASA_LAYERS,
        format_func=lambda x: x.replace("_", " "),
    )

    if st.button("Fetch satellite imagery", type="primary"):
        image = None
        errors = []

        layers_to_try = [selected_layer] + [
            layer for layer in NASA_LAYERS if layer != selected_layer
        ]

        with st.spinner("Requesting imagery from NASA GIBS..."):
            for layer in layers_to_try:
                try:
                    image = fetch_nasa_image(date_string, layer)
                    st.session_state.nasa_image = image
                    st.session_state.nasa_layer = layer
                    st.session_state.nasa_date = date_string
                    break
                except Exception as exc:
                    errors.append(f"{layer}: {type(exc).__name__}: {exc}")

        if image is None:
            st.error("Could not retrieve imagery for the selected date.")
            with st.expander("Fetch diagnostics"):
                for error in errors:
                    st.write(error)
        else:
            st.success(
                "Imagery retrieved: "
                + st.session_state.nasa_layer.replace("_", " ")
            )

    if "nasa_image" in st.session_state:
        st.image(
            st.session_state.nasa_image,
            caption=(
                f"NASA GIBS · {st.session_state.nasa_layer} · "
                f"{st.session_state.nasa_date}"
            ),
            use_container_width=True,
        )

    st.markdown(f"[Open NASA Worldview ↗]({NASA_WORLDVIEW})")
    st.caption(
        "Browse imagery is for exploration; it is not a substitute for "
        "validated scientific analysis or operational imagery products."
    )

# ============================================================
# Spectral Lab
# ============================================================

elif page == "Spectral Lab":
    st.markdown("## Spectral Lab")
    st.write(
        "Load a small sample from EuroSAT MSI to inspect the available "
        "multispectral channels. Dataset configurations can vary."
    )

    st.markdown(f"[EuroSAT MSI dataset ↗]({EUROSAT_URL})")

    try:
        from datasets import load_dataset
    except ImportError:
        load_dataset = None

    if load_dataset is None:
        st.error("The `datasets` package is unavailable in this environment.")
    else:
        if st.button("Load a small EuroSAT sample", type="primary"):
            try:
                with st.spinner("Loading a small dataset sample..."):
                    ds = load_dataset(
                        "blanchon/EuroSAT_MSI",
                        split="train",
                        streaming=True,
                    )
                    sample = next(iter(ds.take(1)))
                    st.session_state.eurosat_sample = sample
                st.success("Sample loaded.")
            except Exception as exc:
                st.error(f"Dataset loading failed: {type(exc).__name__}: {exc}")

    sample = st.session_state.get("eurosat_sample")

    if sample:
        st.markdown("### Sample metadata")
        st.write("Available fields:", list(sample.keys()))

        for key, value in sample.items():
            if isinstance(value, (str, int, float, bool)):
                st.write(f"**{key}:** {value}")
            elif isinstance(value, dict):
                st.write(f"**{key}:** {list(value.keys())}")
            else:
                st.write(f"**{key}:** `{type(value).__name__}`")

        st.info(
            "The exact image/band field and channel ordering must be "
            "confirmed from the loaded dataset metadata before computing "
            "NDVI, NDWI, or NDBI. No scientific index is inferred from an "
            "unknown field layout."
        )
    else:
        st.info(
            "Load a sample to inspect the dataset schema. The dataset is "
            "streamed to avoid downloading the full dataset."
        )

    st.markdown("### Spectral-index reference")
    st.markdown(
        """
        - **NDVI:** commonly uses near-infrared and red reflectance.
        - **NDWI:** several definitions exist; the band selection must be stated.
        - **NDBI:** commonly uses short-wave infrared and near-infrared reflectance.

        These indices require correctly identified spectral bands and
        suitable reflectance data. They are not ground-truth labels.
        """
    )

# ============================================================
# Target Explorer
# ============================================================

elif page == "Target Explorer":
    st.markdown("## Spectral target explorer")
    st.caption("Target IDs represent feature-space clusters, not geographic coordinates.")

    if cluster_model is None:
        st.warning(
            "No compatible clustering model was found in the artifact. "
            "Check the saved artifact keys and dependency versions."
        )
    else:
        st.success(f"Detected clustering model: {describe_model(cluster_model)}")

        n_clusters = getattr(cluster_model, "n_clusters", None)
        if n_clusters is not None:
            st.metric("Configured clusters", int(n_clusters))

        centers = getattr(cluster_model, "cluster_centers_", None)
        if centers is not None:
            centers = np.asarray(centers)
            st.markdown("### Cluster centers")
            st.dataframe(
                pd.DataFrame(
                    centers,
                    index=[f"Cluster {i}" for i in range(len(centers))],
                    columns=[f"Feature {i + 1}" for i in range(centers.shape[1])],
                ),
                use_container_width=True,
            )

            if centers.ndim == 2 and centers.shape[0] > 1:
                if centers.shape[1] >= 2:
                    fig, ax = plt.subplots(figsize=(8, 5))
                    ax.scatter(centers[:, 0], centers[:, 1], s=75)
                    for idx, point in enumerate(centers):
                        ax.annotate(
                            str(idx),
                            (point[0], point[1]),
                            xytext=(5, 5),
                            textcoords="offset points",
                        )
                    ax.set_xlabel("Feature 1")
                    ax.set_ylabel("Feature 2")
                    ax.set_title("Saved cluster centers")
                    ax.grid(alpha=0.2)
                    st.pyplot(fig)
                    plt.close(fig)
        else:
            st.info(
                "The model loaded, but it does not expose `cluster_centers_`. "
                "It may be a different clustering implementation."
            )

    st.markdown("### Feature-space limitations")
    st.write(
        "A cluster is a grouping in feature space. It is not a geographic "
        "location, a confirmed land-cover class, or a validated target."
    )

# ============================================================
# Mission Simulator
# ============================================================

elif page == "Mission Simulator":
    st.markdown("## Mission simulator")
    st.caption(
        "Educational simulation only — not operational satellite guidance."
    )

    if distance_matrix is None or distance_matrix.shape[0] != distance_matrix.shape[1]:
        st.warning(
            "A valid saved square target-distance matrix is needed to run "
            "the simulation. Check artifact keys and matrix dimensions."
        )
    else:
        st.success(
            f"Distance matrix ready: {distance_matrix.shape[0]} × "
            f"{distance_matrix.shape[1]}"
        )

        policy_name = st.selectbox(
            "Policy",
            ["greedy", "random", "q_learning", "sarsa"],
        )
        steps = st.slider("Simulation steps", 5, 200, 40)

        selected_policy = (
            q_table if policy_name == "q_learning"
            else sarsa_table if policy_name == "sarsa"
            else policy_name
        )

        if policy_name in ("q_learning", "sarsa") and not isinstance(
            selected_policy, dict
        ):
            st.warning(
                f"No compatible saved {policy_name} table was found. "
                "Select greedy or random, or repair the saved artifact."
            )
        elif st.button("Run simulation", type="primary"):
            try:
                result = run_simulation(
                    distance_matrix,
                    selected_policy,
                    steps=steps,
                )

                c1, c2, c3 = st.columns(3)
                c1.metric("Visited targets", result["unique_targets"])
                c2.metric("Route length", f'{result["total_distance"]:.3f}')
                c3.metric("Steps completed", len(result["rewards"]))

                st.markdown("### Simulated route")
                st.write(" → ".join(map(str, result["route"])))

                if result["rewards"]:
                    fig, ax = plt.subplots(figsize=(9, 4))
                    ax.plot(result["rewards"])
                    ax.set_xlabel("Step")
                    ax.set_ylabel("Simulated reward")
                    ax.set_title("Reward per step")
                    ax.grid(alpha=0.25)
                    st.pyplot(fig)
                    plt.close(fig)

            except Exception as exc:
                st.error(f"Simulation failed: {type(exc).__name__}: {exc}")

# ============================================================
# Policy Benchmark
# ============================================================

elif page == "Policy Benchmark":
    st.markdown("## Policy benchmark")
    st.caption(
        "Results describe this simulator only and do not establish "
        "real-world policy superiority."
    )

    if distance_matrix is None or distance_matrix.shape[0] != distance_matrix.shape[1]:
        st.warning(
            "A valid saved target-distance matrix is needed for policy "
            "benchmarking. Load or recreate the original artifact."
        )
    else:
        st.write(
            f"Matrix size: {distance_matrix.shape[0]} targets."
        )

        if st.button("Run benchmark", type="primary"):
            policies = {
                "Random": "random",
                "Greedy": "greedy",
            }

            if isinstance(q_table, dict) and q_table:
                policies["Q-learning"] = q_table

            if isinstance(sarsa_table, dict) and sarsa_table:
                policies["SARSA"] = sarsa_table

            records = []

            for name, policy in policies.items():
                distances = []
                rewards = []
                coverage = []

                for seed in range(10):
                    try:
                        result = run_simulation(
                            distance_matrix,
                            policy,
                            steps=min(40, distance_matrix.shape[0] * 2),
                            seed=seed,
                        )
                        distances.append(result["total_distance"])
                        rewards.append(
                            float(np.mean(result["rewards"]))
                            if result["rewards"] else 0.0
                        )
                        coverage.append(result["unique_targets"])
                    except Exception:
                        continue

                if distances:
                    records.append(
                        {
                            "Policy": name,
                            "Mean route distance": np.mean(distances),
                            "Mean reward per step": np.mean(rewards),
                            "Mean unique targets": np.mean(coverage),
                            "Runs": len(distances),
                        }
                    )

            if records:
                frame = pd.DataFrame(records)
                st.dataframe(frame, use_container_width=True)

                fig, ax = plt.subplots(figsize=(9, 4))
                ax.bar(frame["Policy"], frame["Mean route distance"])
                ax.set_ylabel("Mean route distance")
                ax.set_title("Simulated policy comparison")
                ax.tick_params(axis="x", rotation=15)
                ax.grid(axis="y", alpha=0.2)
                st.pyplot(fig)
                plt.close(fig)

                st.caption(
                    "Random runs use different seeds. The benchmark is "
                    "a small simulator experiment, not a statistical "
                    "claim about real satellite operations."
                )
            else:
                st.error("No benchmark runs completed successfully.")

# ============================================================
# Methods
# ============================================================

elif page == "Methods":
    st.markdown("## Methods, data, and limitations")

    st.markdown("### NASA imagery")
    st.write(
        "NASA GIBS WMS provides browse imagery. The application attempts "
        "alternative layers when a selected layer cannot be fetched. "
        "NASA Worldview opens separately."
    )
    st.markdown(f"[NASA Worldview / GIBS ↗]({NASA_WORLDVIEW})")

    st.markdown("### EuroSAT MSI and spectral indices")
    st.write(
        "EuroSAT MSI is a multispectral dataset. The exact field layout, "
        "channel ordering, reflectance scaling, and band names must be "
        "verified before computing scientific indices."
    )
    st.markdown(f"[EuroSAT MSI on Hugging Face ↗]({EUROSAT_URL})")

    st.markdown("### Clustering and reinforcement learning")
    st.markdown(
        """
        - Clusters represent feature-space groups, not geographic coordinates.
        - Cloud probability and energy cost require explicit simulation assumptions.
        - Saved tabular agents may encounter states not present during training.
        - Benchmark results describe the simulator only.
        - A small number of runs does not establish statistical superiority.
        """
    )

    st.markdown("### Model artifact")
    st.write(
        f"Artifact: `{artifact_path}`"
        if artifact_path is not None
        else "Artifact: not found"
    )
    st.write(f"Q-learning entries: {safe_count(q_table)}")
    st.write(f"SARSA entries: {safe_count(sarsa_table)}")
    st.write(f"Clustering model: {describe_model(cluster_model)}")
    st.write(
        "Distance matrix: "
        + (
            f"{distance_matrix.shape[0]} × {distance_matrix.shape[1]}"
            if distance_matrix is not None
            else "unavailable"
        )
    )

    if artifact_message == "Artifact loaded successfully.":
        st.success(artifact_message)
    else:
        st.error(artifact_message)

    st.warning(
        "Pickle artifacts should only be loaded from trusted sources. "
        "Estimator compatibility depends on the versions used when "
        "the artifact was created."
    )

    st.markdown("### Research boundaries")
    st.write(
        "ORBITAL is an Earth-observation and machine-learning research "
        "prototype. It is not affiliated with NASA and is not intended "
        "for operational satellite guidance."
    )

# ============================================================
# Footer
# ============================================================

st.divider()
st.caption(
    "ORBITAL · Earth-observation research prototype · "
    "Not affiliated with NASA · Not operational satellite guidance."
)
