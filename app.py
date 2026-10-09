
# ORBITAL — Earth Intelligence / Research Prototype
# Streamlit application with graceful model loading and simulation fallbacks.

from __future__ import annotations

import os
import pickle
import warnings
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import streamlit as st
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=UserWarning)

# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------

st.set_page_config(
    page_title="ORBITAL | Earth Intelligence",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="collapsed",
)

APP_DIR = Path(__file__).resolve().parent
MODEL_PATHS = [
    APP_DIR / "orbital_models.pkl",
    APP_DIR / "orbital_models" / "orbital_models.pkl",
]

NASA_WMS = (
    "https://gibs.earthdata.nasa.gov/wms/epsg3857/best/wms.cgi"
)
NASA_WORLDVIEW = "https://worldview.earthdata.nasa.gov/"
EUROSAT_URL = "https://huggingface.co/datasets/blanchon/EuroSAT_MSI"

# ---------------------------------------------------------
# VISUAL STYLE
# ---------------------------------------------------------

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Space+Grotesk:wght@400;500;600;700&display=swap');

    .stApp {
        background: #080d16;
        color: #e8eef7;
        font-family: 'Space Grotesk', sans-serif;
    }
    [data-testid="stHeader"] {
        background: rgba(8,13,22,0.96);
    }
    [data-testid="stMetric"] {
        background: #101a29;
        border: 1px solid #26364b;
        padding: 15px;
        border-radius: 10px;
    }
    .orbital-eyebrow {
        color: #65d9c2;
        font-family: 'DM Mono', monospace;
        font-size: 0.76rem;
        letter-spacing: 0.16rem;
    }
    .orbital-title {
        font-size: clamp(2.2rem, 5vw, 4rem);
        font-weight: 700;
        letter-spacing: -0.06em;
        color: #f4f7fc;
        margin-bottom: 0;
    }
    .orbital-subtitle {
        color: #9caec5;
        font-size: 0.95rem;
    }
    .orbital-card {
        background: #101a29;
        border: 1px solid #26364b;
        padding: 18px;
        border-radius: 10px;
        margin-bottom: 10px;
    }
    .orbital-muted {
        color: #9caec5;
        font-size: 0.9rem;
    }
    div[data-testid="stAlert"] {
        border-radius: 9px;
    }
    a {
        color: #65d9c2 !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------
# SAFE MODEL LOADING
# ---------------------------------------------------------

def find_model_file():
    for path in MODEL_PATHS:
        if path.is_file():
            return path
    return None


@st.cache_resource(show_spinner=False)
def load_saved_artifact():
    """
    Loads a trusted local pickle file.
    Errors are recorded internally and never rendered as raw
    tracebacks in the public interface.
    """
    path = find_model_file()

    if path is None:
        return {
            "artifact": None,
            "status": "missing",
            "message": "No saved model artifact was found.",
            "path": None,
        }

    try:
        with path.open("rb") as file:
            artifact = pickle.load(file)

        if not isinstance(artifact, dict):
            return {
                "artifact": None,
                "status": "invalid",
                "message": "The saved artifact has an unsupported structure.",
                "path": str(path),
            }

        return {
            "artifact": artifact,
            "status": "loaded",
            "message": "Saved model artifact loaded successfully.",
            "path": str(path),
        }

    except Exception:
        # Do not display exception details to app visitors.
        return {
            "artifact": None,
            "status": "incompatible",
            "message": (
                "The saved models are incompatible with this "
                "runtime and could not be loaded."
            ),
            "path": str(path),
        }


def first_present(mapping, names):
    if not isinstance(mapping, dict):
        return None

    for name in names:
        value = mapping.get(name)
        if value is not None:
            return value

    return None


def count_states(table):
    if table is None:
        return None

    try:
        if isinstance(table, dict):
            return len(table)

        array = np.asarray(table)
        if array.ndim >= 2:
            return int(array.shape[0])

    except Exception:
        pass

    return None


def extract_models(artifact):
    if not isinstance(artifact, dict):
        return {
            "q_table": None,
            "sarsa_table": None,
            "cluster_model": None,
            "distance_matrix": None,
            "scaler": None,
        }

    return {
        "q_table": first_present(
            artifact,
            ["q_table", "q_learning_table", "q_learning", "Q", "q_values"],
        ),
        "sarsa_table": first_present(
            artifact,
            ["sarsa_table", "sarsa", "sarsa_values", "SARSA"],
        ),
        "cluster_model": first_present(
            artifact,
            ["cluster_model", "kmeans", "kmeans_model", "clustering_model"],
        ),
        "distance_matrix": first_present(
            artifact,
            ["distance_matrix", "distances", "target_distances", "cost_matrix"],
        ),
        "scaler": first_present(
            artifact,
            ["scaler", "feature_scaler", "standard_scaler"],
        ),
    }


def valid_distance_matrix(value):
    if value is None:
        return None

    try:
        matrix = np.asarray(value, dtype=float)

        if (
            matrix.ndim == 2
            and matrix.shape[0] == matrix.shape[1]
            and matrix.shape[0] >= 2
            and np.isfinite(matrix).all()
            and (matrix >= 0).all()
        ):
            return matrix

    except Exception:
        pass

    return None


# ---------------------------------------------------------
# SIMULATED MISSION ENVIRONMENT
# ---------------------------------------------------------

def make_simulated_distances(n_targets=8, seed=42):
    """
    Deterministic synthetic distances for a demonstrative
    simulator. These are not real satellite orbital distances.
    """
    rng = np.random.default_rng(seed)
    points = rng.uniform(0, 100, size=(n_targets, 2))
    delta = points[:, None, :] - points[None, :, :]
    distances = np.sqrt(np.sum(delta ** 2, axis=2))
    np.fill_diagonal(distances, 0.0)
    return distances


def run_mission(policy, distances, start=0, max_steps=12, seed=7):
    rng = np.random.default_rng(seed)
    n = distances.shape[0]
    current = int(start) % n
    visited = {current}
    route = [current]
    total_distance = 0.0
    reward_total = 0.0

    for _ in range(max_steps):
        candidates = [i for i in range(n) if i not in visited]

        if not candidates:
            break

        if policy == "Random":
            nxt = int(rng.choice(candidates))

        elif policy in ("Greedy", "Fallback baseline"):
            nxt = min(candidates, key=lambda i: distances[current, i])

        elif policy == "Q-learning" and Q_TABLE is not None:
            nxt = choose_saved_action(Q_TABLE, current, candidates, rng)

        elif policy == "SARSA" and SARSA_TABLE is not None:
            nxt = choose_saved_action(SARSA_TABLE, current, candidates, rng)

        else:
            nxt = min(candidates, key=lambda i: distances[current, i])

        step_cost = float(distances[current, nxt])
        total_distance += step_cost
        reward_total -= step_cost
        route.append(nxt)
        visited.add(nxt)
        current = nxt

    return {
        "route": route,
        "distance": total_distance,
        "reward": reward_total,
        "targets_visited": len(visited),
    }


def choose_saved_action(table, state, candidates, rng):
    """
    Supports common dictionary-based tabular Q representations.
    Falls back to a valid action if the trained state is absent.
    """
    try:
        if isinstance(table, dict):
            row = table.get(state, table.get(str(state)))

            if isinstance(row, dict):
                options = [
                    (action, float(value))
                    for action, value in row.items()
                    if str(action).isdigit()
                    and int(action) in candidates
                ]
                if options:
                    return int(max(options, key=lambda x: x[1])[0])

            if row is not None:
                values = np.asarray(row, dtype=float).ravel()
                available = [
                    (i, values[i])
                    for i in candidates
                    if i < len(values) and np.isfinite(values[i])
                ]
                if available:
                    return int(max(available, key=lambda x: x[1])[0])

        array = np.asarray(table, dtype=float)
        if array.ndim == 2 and state < array.shape[0]:
            available = [
                i for i in candidates if i < array.shape[1]
            ]
            if available:
                return int(max(available, key=lambda i: array[state, i]))

    except Exception:
        pass

    return int(min(candidates, key=lambda i: abs(i - state)))


def benchmark_policies(distances, repeats=20):
    records = []

    for policy in ["Random", "Greedy"]:
        scores = []

        for seed in range(repeats):
            result = run_mission(
                policy,
                distances,
                seed=seed,
                max_steps=min(12, distances.shape[0] - 1),
            )
            scores.append(result["distance"])

        records.append({
            "Policy": policy,
            "Mean distance": float(np.mean(scores)),
            "Std. deviation": float(np.std(scores)),
            "Runs": repeats,
            "Evaluation": "Simulated baseline",
        })

    for policy, table in [
        ("Q-learning", Q_TABLE),
        ("SARSA", SARSA_TABLE),
    ]:
        if table is None:
            continue

        scores = []

        for seed in range(repeats):
            result = run_mission(
                policy,
                distances,
                seed=seed,
                max_steps=min(12, distances.shape[0] - 1),
            )
            scores.append(result["distance"])

        records.append({
            "Policy": policy,
            "Mean distance": float(np.mean(scores)),
            "Std. deviation": float(np.std(scores)),
            "Runs": repeats,
            "Evaluation": "Saved agent",
        })

    return pd.DataFrame(records)


# ---------------------------------------------------------
# NASA IMAGERY
# ---------------------------------------------------------

@st.cache_data(ttl=1800, show_spinner=False)
def fetch_nasa_image(layer, image_date):
    params = {
        "SERVICE": "WMS",
        "REQUEST": "GetMap",
        "VERSION": "1.1.1",
        "LAYERS": layer,
        "STYLES": "",
        "FORMAT": "image/jpeg",
        "SRS": "EPSG:4326",
        "BBOX": "-180,-85,180,85",
        "WIDTH": "1200",
        "HEIGHT": "570",
        "TIME": image_date,
    }

    try:
        response = requests.get(
            NASA_WMS,
            params=params,
            timeout=25,
            headers={"User-Agent": "ORBITAL-Earth-Intelligence/1.0"},
        )
        content_type = response.headers.get("content-type", "")

        if response.ok and "image" in content_type.lower():
            return response.content

    except Exception:
        pass

    return None


def get_nasa_browse_image():
    today = date.today()

    layers = [
        "VIIRS_SNPP_CorrectedReflectance_TrueColor",
        "MODIS_Terra_CorrectedReflectance_TrueColor",
    ]

    for day_offset in range(0, 8):
        day = (today - timedelta(days=day_offset)).strftime("%Y-%m-%d")

        for layer in layers:
            image = fetch_nasa_image(layer, day)
            if image:
                return image, day.replace("-", ""), layer

    return None, None, None


# ---------------------------------------------------------
# OPTIONAL EUROSAT ACCESS
# ---------------------------------------------------------

@st.cache_data(ttl=3600, show_spinner=False)
def load_eurosat_sample(max_rows=100):
    """
    Loads a small sample only when the optional datasets package
    and remote dataset are available. No raw errors are exposed.
    """
    try:
        from datasets import load_dataset

        dataset = load_dataset(
            "blanchon/EuroSAT_MSI",
            split=f"train[:{max_rows}]",
        )

        rows = []

        for item in dataset:
            row = {}

            for key, value in item.items():
                if isinstance(value, (int, float, np.integer, np.floating)):
                    row[key] = float(value)

            if row:
                rows.append(row)

        if rows:
            return pd.DataFrame(rows), None

        return None, "The dataset did not return tabular spectral values."

    except Exception:
        return None, (
            "EuroSAT data is temporarily unavailable. "
            "Try again later or check the dataset connection."
        )


# ---------------------------------------------------------
# LOAD ARTIFACT AND DERIVE SAFE STATE
# ---------------------------------------------------------

artifact_result = load_saved_artifact()
ARTIFACT = artifact_result["artifact"]
MODELS = extract_models(ARTIFACT)

Q_TABLE = MODELS["q_table"]
SARSA_TABLE = MODELS["sarsa_table"]
CLUSTER_MODEL = MODELS["cluster_model"]
DISTANCE_MATRIX = valid_distance_matrix(MODELS["distance_matrix"])

q_count = count_states(Q_TABLE)
sarsa_count = count_states(SARSA_TABLE)

if DISTANCE_MATRIX is not None:
    ACTIVE_DISTANCES = DISTANCE_MATRIX
    SIMULATION_MODE = "Saved distance matrix"
else:
    ACTIVE_DISTANCES = make_simulated_distances()
    SIMULATION_MODE = "Synthetic demonstration environment"


# ---------------------------------------------------------
# HEADER
# ---------------------------------------------------------

left, right = st.columns([3, 1])

with left:
    st.markdown(
        '<div class="orbital-eyebrow">EARTH INTELLIGENCE / RESEARCH PROTOTYPE</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="orbital-title">🌍 ORBITAL</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="orbital-subtitle">'
        'Explore Earth observation, spectral analysis, and reinforcement learning.'
        '</div>',
        unsafe_allow_html=True,
    )

with right:
    st.markdown(" ")
    st.link_button("NASA Worldview ↗", NASA_WORLDVIEW, width="stretch")
    st.link_button("EuroSAT MSI ↗", EUROSAT_URL, width="stretch")

st.divider()

# ---------------------------------------------------------
# STATUS STRIP — NO RAW EXCEPTIONS
# ---------------------------------------------------------

status_col, q_col, sarsa_col, mode_col = st.columns(4)

with status_col:
    if artifact_result["status"] == "loaded":
        st.metric("Saved models", "Loaded")
    else:
        st.metric("Saved models", "Unavailable")

with q_col:
    st.metric(
        "Q-learning states",
        str(q_count) if q_count is not None else "Not loaded",
    )

with sarsa_col:
    st.metric(
        "SARSA states",
        str(sarsa_count) if sarsa_count is not None else "Not loaded",
    )

with mode_col:
    st.metric(
        "Mission environment",
        "Saved data" if DISTANCE_MATRIX is not None else "Demo mode",
    )

if artifact_result["status"] != "loaded":
    with st.expander("Model status and troubleshooting", expanded=False):
        st.info(
            "The application is still available. Saved models could not be "
            "loaded, so model-dependent features may be limited."
        )
        st.caption(
            "To restore trained agents, use an artifact saved with compatible "
            "Python, NumPy, and scikit-learn versions."
        )
        if artifact_result.get("path"):
            st.caption(f"Artifact location: {artifact_result['path']}")

st.caption(
    "NASA imagery is provided through NASA GIBS. "
    "ORBITAL is an independent research prototype and is not affiliated with NASA."
)

st.divider()

# ---------------------------------------------------------
# NAVIGATION
# ---------------------------------------------------------

tabs = st.tabs([
    "Overview",
    "Earth from Orbit",
    "Spectral Lab",
    "Target Explorer",
    "Mission Simulator",
    "Policy Benchmark",
    "Methods",
])

# ---------------------------------------------------------
# OVERVIEW
# ---------------------------------------------------------

with tabs[0]:
    st.subheader("Earth intelligence dashboard")

    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown("**Earth observation**")
        st.write("Browse satellite imagery served by NASA GIBS.")

    with c2:
        st.markdown("**Multispectral research**")
        st.write("Explore spectral data and vegetation or water indices.")

    with c3:
        st.markdown("**Reinforcement learning**")
        st.write("Compare mission policies in a simulated environment.")

    st.divider()

    st.markdown("#### System status")

    status_df = pd.DataFrame([
        {
            "Component": "NASA GIBS",
            "Status": "Available on request",
        },
        {
            "Component": "EuroSAT MSI",
            "Status": "Available on request",
        },
        {
            "Component": "Saved model artifact",
            "Status": (
                "Loaded"
                if artifact_result["status"] == "loaded"
                else "Unavailable — app remains usable"
            ),
        },
        {
            "Component": "Mission simulation",
            "Status": SIMULATION_MODE,
        },
        {
            "Component": "Clustering",
            "Status": (
                "Saved model detected"
                if CLUSTER_MODEL is not None
                else "Saved model unavailable"
            ),
        },
    ])

    st.dataframe(status_df, hide_index=True, width="stretch")

    st.info(
        "Simulated mission results are educational demonstrations, "
        "not real satellite guidance or operational flight instructions."
    )

# ---------------------------------------------------------
# EARTH FROM ORBIT
# ---------------------------------------------------------

with tabs[1]:
    st.subheader("Satellite imagery")

    if st.button("Refresh NASA imagery", key="refresh_nasa"):
        fetch_nasa_image.clear()
        st.rerun()

    with st.spinner("Requesting browse imagery from NASA GIBS..."):
        nasa_image, nasa_date, nasa_layer = get_nasa_browse_image()

    if nasa_image:
        st.image(
            nasa_image,
            caption=(
                f"NASA GIBS browse imagery · {nasa_date} · {nasa_layer}"
            ),
            width="stretch",
        )
        st.caption(
            "Global browse imagery. Cloud cover, image availability, "
            "and acquisition dates vary by layer."
        )
    else:
        st.warning(
            "NASA imagery is temporarily unavailable. "
            "Open NASA Worldview to explore current imagery."
        )
        st.link_button("Open NASA Worldview", NASA_WORLDVIEW)

    st.markdown("#### Data source")
    st.write("NASA Global Imagery Browse Services (GIBS).")
    st.link_button("NASA GIBS / Worldview ↗", NASA_WORLDVIEW)

# ---------------------------------------------------------
# SPECTRAL LAB
# ---------------------------------------------------------

with tabs[2]:
    st.subheader("Spectral Lab")
    st.write(
        "Inspect numeric spectral fields available in a small EuroSAT MSI sample."
    )

    if st.button("Load EuroSAT sample", key="load_eurosat"):
        st.session_state["orbital_load_eurosat"] = True

    if st.session_state.get("orbital_load_eurosat", False):
        with st.spinner("Loading a small EuroSAT sample..."):
            spectral_df, spectral_message = load_eurosat_sample()

        if spectral_df is not None and not spectral_df.empty:
            st.success(f"Loaded {len(spectral_df)} sample rows.")
            st.dataframe(spectral_df.head(20), width="stretch")

            numeric_cols = spectral_df.select_dtypes(
                include=np.number
            ).columns.tolist()

            if numeric_cols:
                chosen = st.selectbox(
                    "Choose a numeric feature",
                    numeric_cols,
                    key="spectral_feature",
                )

                fig, ax = plt.subplots(figsize=(9, 3.5))
                ax.hist(
                    spectral_df[chosen].dropna(),
                    bins=20,
                    edgecolor="black",
                )
                ax.set_title(f"Distribution: {chosen}")
                ax.set_xlabel(chosen)
                ax.set_ylabel("Frequency")
                fig.tight_layout()
                st.pyplot(fig)
                plt.close(fig)
            else:
                st.info("No numeric spectral columns were found in this sample.")

        else:
            st.warning(
                spectral_message
                or "Spectral data is currently unavailable."
            )
            st.link_button("Open EuroSAT MSI dataset ↗", EUROSAT_URL)
    else:
        st.info(
            "Select “Load EuroSAT sample” to request a small sample. "
            "No dataset is downloaded until you request it."
        )

    st.divider()
    st.markdown("#### About spectral indices")
    st.write(
        "NDVI, NDWI, and NDBI require correctly identified spectral bands. "
        "This interface does not calculate those indices unless the necessary "
        "band names and channel mapping are available."
    )

# ---------------------------------------------------------
# TARGET EXPLORER
# ---------------------------------------------------------

with tabs[3]:
    st.subheader("Spectral target explorer")
    st.caption(
        "Target IDs refer to feature-space clusters, not geographic coordinates."
    )

    if CLUSTER_MODEL is not None:
        st.success("A saved clustering object was found.")

        st.info(
            "The model is available, but predictions require the exact feature "
            "columns and preprocessing used during training."
        )

        if st.button("Check clustering interface", key="check_cluster"):
            try:
                if hasattr(CLUSTER_MODEL, "n_clusters"):
                    st.write(
                        f"Configured cluster count: {CLUSTER_MODEL.n_clusters}"
                    )
                elif hasattr(CLUSTER_MODEL, "n_clusters_"):
                    st.write(
                        f"Detected cluster count: {CLUSTER_MODEL.n_clusters_}"
                    )
                else:
                    st.write("Cluster count is not exposed by this estimator.")
            except Exception:
                st.info("Cluster metadata is not available for this model.")
    else:
        st.info(
            "No compatible saved clustering model is available. "
            "This panel remains usable, but model-based cluster predictions "
            "are disabled until the trained artifact is restored."
        )

    st.markdown("#### Synthetic feature-space demonstration")

    demo_count = st.slider(
        "Number of demonstration points",
        min_value=20,
        max_value=200,
        value=80,
        step=20,
        key="target_demo_count",
    )

    rng = np.random.default_rng(11)
    demo_points = rng.normal(size=(demo_count, 2))

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.scatter(
        demo_points[:, 0],
        demo_points[:, 1],
        s=25,
        alpha=0.75,
    )
    ax.set_title("Synthetic feature-space points")
    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Feature 2")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.caption(
        "These points are generated for interface demonstration only. "
        "They are not satellite observations or predictions from the saved model."
    )

# ---------------------------------------------------------
# MISSION SIMULATOR
# ---------------------------------------------------------

with tabs[4]:
    st.subheader("Mission simulator")

    if DISTANCE_MATRIX is None:
        st.info(
            "The saved target-distance matrix is unavailable. "
            "You can still explore the simulator using a synthetic demonstration "
            "environment; results are not from a trained orbital model."
        )
    else:
        st.success("Using the distance matrix from the saved artifact.")

    sim_col1, sim_col2 = st.columns(2)

    with sim_col1:
        start_target = st.number_input(
            "Starting target ID",
            min_value=0,
            max_value=ACTIVE_DISTANCES.shape[0] - 1,
            value=0,
            step=1,
        )

    with sim_col2:
        max_steps = st.slider(
            "Maximum observations",
            min_value=1,
            max_value=min(20, ACTIVE_DISTANCES.shape[0] - 1),
            value=min(5, ACTIVE_DISTANCES.shape[0] - 1),
        )

    available_policies = ["Random", "Greedy", "Fallback baseline"]

    if Q_TABLE is not None:
        available_policies.append("Q-learning")

    if SARSA_TABLE is not None:
        available_policies.append("SARSA")

    selected_policy = st.selectbox(
        "Policy",
        available_policies,
        key="mission_policy",
    )

    if st.button("Run mission simulation", key="run_mission"):
        result = run_mission(
            selected_policy,
            ACTIVE_DISTANCES,
            start=int(start_target),
            max_steps=max_steps,
        )

        m1, m2, m3 = st.columns(3)

        with m1:
            st.metric("Targets visited", result["targets_visited"])

        with m2:
            st.metric("Total simulated distance", f"{result['distance']:.2f}")

        with m3:
            st.metric("Cumulative reward", f"{result['reward']:.2f}")

        route_text = " → ".join(str(x) for x in result["route"])
        st.markdown("#### Route")
        st.code(route_text, language="text")

        if selected_policy in ("Random", "Greedy", "Fallback baseline"):
            st.caption("Result produced by a baseline policy, not a trained agent.")
        else:
            st.caption("Result produced using a saved tabular agent.")

        st.caption(f"Environment: {SIMULATION_MODE}")

    with st.expander("View simulation distance matrix"):
        st.dataframe(
            pd.DataFrame(ACTIVE_DISTANCES).round(2),
            width="stretch",
        )

# ---------------------------------------------------------
# POLICY BENCHMARK
# ---------------------------------------------------------

with tabs[5]:
    st.subheader("Policy benchmark")

    if DISTANCE_MATRIX is None:
        st.info(
            "No saved target-distance matrix was loaded. "
            "The benchmark below uses a synthetic environment for demonstration."
        )

    st.caption(
        "Lower simulated distance is better for this particular objective. "
        "Results do not establish real-world mission performance."
    )

    repeat_count = st.slider(
        "Runs per policy",
        min_value=5,
        max_value=100,
        value=20,
        step=5,
        key="benchmark_repeats",
    )

    if st.button("Run policy benchmark", key="run_benchmark"):
        with st.spinner("Evaluating policies..."):
            benchmark_df = benchmark_policies(
                ACTIVE_DISTANCES,
                repeats=repeat_count,
            )

        if benchmark_df.empty:
            st.info(
                "No benchmark results are available. "
                "Baseline policies can still be evaluated."
            )
        else:
            st.dataframe(
                benchmark_df.round(3),
                hide_index=True,
                width="stretch",
            )

            fig, ax = plt.subplots(figsize=(9, 4))
            ax.bar(
                benchmark_df["Policy"],
                benchmark_df["Mean distance"],
                yerr=benchmark_df["Std. deviation"],
                capsize=4,
            )
            ax.set_ylabel("Mean simulated distance")
            ax.set_title("Policy comparison")
            ax.tick_params(axis="x", rotation=20)
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

            st.download_button(
                "Download benchmark CSV",
                data=benchmark_df.to_csv(index=False).encode("utf-8"),
                file_name="orbital_policy_benchmark.csv",
                mime="text/csv",
            )

# ---------------------------------------------------------
# METHODS AND LIMITATIONS
# ---------------------------------------------------------

with tabs[6]:
    st.subheader("Methods, data, and limitations")

    with st.expander("NASA imagery", expanded=True):
        st.write(
            "NASA GIBS WMS supplies browse imagery. The application tries "
            "alternative layers and recent dates when an image cannot be fetched. "
            "NASA Worldview opens separately."
        )
        st.link_button("NASA Worldview ↗", NASA_WORLDVIEW)

    with st.expander("EuroSAT MSI and spectral analysis"):
        st.write(
            "A small dataset sample can be requested from the EuroSAT MSI "
            "Hugging Face dataset. Only numeric fields exposed by the sample "
            "are displayed by this version."
        )
        st.write(
            "NDVI, NDWI, and NDBI depend on the correct band mapping. "
            "Spectral indices are exploratory measurements, not ground-truth "
            "classifications."
        )
        st.link_button("EuroSAT MSI ↗", EUROSAT_URL)

    with st.expander("Clustering and reinforcement learning"):
        st.write(
            "Clusters represent feature-space groups, not geographic locations. "
            "Cloud probability and energy cost are not modeled as real telemetry "
            "in the fallback simulator."
        )
        st.write(
            "Saved tabular agents may encounter states that were not present "
            "during training. Benchmark results describe the selected simulation "
            "environment only."
        )

    with st.expander("Model artifact and diagnostics"):
        st.write(f"Artifact status: {artifact_result['status'].title()}")
        st.write(
            artifact_result["message"]
        )

        if artifact_result.get("path"):
            st.code(artifact_result["path"], language="text")

        st.write(
            "Pickle files should only be loaded from trusted sources. "
            "Compatibility depends on the Python and library versions used "
            "when the artifact was saved."
        )

        st.write(
            "If the artifact cannot be loaded, the app keeps working in "
            "demonstration mode. Missing saved models are not replaced with "
            "claims of trained-model performance."
        )

    with st.expander("Research limitations"):
        st.write(
            "This project is an educational earth-observation and reinforcement "
            "learning prototype. It is not operational satellite guidance."
        )
        st.write(
            "Synthetic targets and simulated distances must not be interpreted "
            "as real orbital positions, mission plans, or spacecraft telemetry."
        )

    st.markdown("#### Data sources")
    st.markdown(f"- [NASA Worldview / GIBS]({NASA_WORLDVIEW})")
    st.markdown(f"- [EuroSAT MSI]({EUROSAT_URL})")

# ---------------------------------------------------------
# FOOTER
# ---------------------------------------------------------

st.divider()

st.markdown(
    """
    <div class="orbital-muted" style="text-align:center; padding:10px;">
        ORBITAL · Earth-observation research prototype<br>
        Not affiliated with NASA · Not operational satellite guidance
    </div>
    """,
    unsafe_allow_html=True,
)
