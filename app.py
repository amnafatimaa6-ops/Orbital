
from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path
import io
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import streamlit as st
from PIL import Image

# ============================================================
# ORBITAL | Earth observation, multispectral analysis and RL
# Research prototype only. Does not control real satellites.
# ============================================================

st.set_page_config(
    page_title="ORBITAL | Earth Intelligence",
    page_icon="🌍",
    layout="wide",
)

ROOT = Path(__file__).resolve().parent
ARTIFACT_PATHS = [
    ROOT / "orbital_models.pkl",
    ROOT / "orbital_models" / "orbital_models.pkl",
]
NASA_WMS = "https://gibs.earthdata.nasa.gov/wms/epsg4326/best/wms.cgi"
WORLDVIEW_URL = "https://worldview.earthdata.nasa.gov/"
EUROSAT_URL = "https://huggingface.co/datasets/blanchon/EuroSAT_MSI"
EPS = 1e-8

LAYERS = {
    "VIIRS Suomi NPP": "VIIRS_SNPP_CorrectedReflectance_TrueColor",
    "MODIS Terra": "MODIS_Terra_CorrectedReflectance_TrueColor",
    "MODIS Aqua": "MODIS_Aqua_CorrectedReflectance_TrueColor",
}

REGIONS = {
    "Global": (-180, -90, 180, 90),
    "South Asia": (60, 5, 100, 38),
    "Europe": (-12, 34, 40, 72),
    "Africa": (-20, -36, 55, 38),
    "North America": (-170, 12, -50, 75),
    "South America": (-85, -57, -32, 14),
    "Australia": (110, -50, 180, -5),
}

POLICY_LABELS = {
    "random": "Random baseline",
    "greedy": "Greedy novelty",
    "q_learning": "Q-learning",
    "sarsa": "SARSA",
}

POLICY_COLORS = {
    "random": "#8caabd",
    "greedy": "#f2b66d",
    "q_learning": "#7ce0ca",
    "sarsa": "#8fa8ff",
}

BAND_NAMES = [
    "B1", "B2", "B3", "B4", "B5", "B6", "B7",
    "B8", "B8A", "B9", "B10", "B11", "B12",
]

plt.rcParams.update({
    "figure.facecolor": "#0b1928",
    "axes.facecolor": "#102235",
    "axes.edgecolor": "#294258",
    "axes.labelcolor": "#c5d6e3",
    "axes.titlecolor": "#eaf4ff",
    "xtick.color": "#c5d6e3",
    "ytick.color": "#c5d6e3",
    "text.color": "#c5d6e3",
    "grid.color": "#294258",
    "legend.facecolor": "#102235",
    "legend.edgecolor": "#294258",
})


st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Space+Grotesk:wght@400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Space Grotesk', sans-serif; }
.stApp {
    background: radial-gradient(ellipse at 15% 0%, #122b43 0%, transparent 38%),
                linear-gradient(180deg,#07111e 0%,#091522 100%);
}
section[data-testid="stSidebar"] {
    background: #0a1624;
    border-right: 1px solid #21364a;
}
.block-container { max-width: 1500px; padding-top: 1.5rem; }
.orbital-hero {
    padding: 30px;
    border: 1px solid #28445b;
    border-radius: 20px;
    background: linear-gradient(130deg,#10283d,#0b1928 65%,#122b37);
    margin-bottom: 20px;
}
.orbital-eyebrow {
    color: #7ce0ca;
    font-family: 'DM Mono', monospace;
    font-size: 12px;
    letter-spacing: 2px;
    text-transform: uppercase;
}
.orbital-title {
    font-size: clamp(42px, 6vw, 68px);
    line-height: 1;
    font-weight: 700;
    letter-spacing: -3px;
    margin: 12px 0;
    color: #f3f8ff;
}
.orbital-subtitle { color: #a8bfd1; line-height: 1.7; }
div[data-testid="stMetric"] {
    background: #102235;
    padding: 14px;
    border: 1px solid #223b50;
    border-radius: 12px;
}
div[data-testid="stMetricValue"] { color: #7ce0ca; }
hr { border-color: #223b50; }
</style>
""", unsafe_allow_html=True)


# ============================================================
# MODEL LOADING
# ============================================================

def first_existing(mapping, keys, default=None):
    if not isinstance(mapping, dict):
        return default

    for key in keys:
        if key in mapping and mapping[key] is not None:
            return mapping[key]

    return default


@st.cache_resource(show_spinner=False)
def load_artifact():
    for path in ARTIFACT_PATHS:
        if not path.is_file():
            continue

        try:
            with path.open("rb") as file:
                data = pickle.load(file)

            if not isinstance(data, dict):
                return {}, str(path), (
                    "The artifact loaded, but its top-level object "
                    "is not a dictionary."
                )

            return data, str(path), None

        except Exception as exc:
            return {}, str(path), f"{type(exc).__name__}: {exc}"

    return {}, None, "orbital_models.pkl was not found."


artifact, artifact_path, artifact_error = load_artifact()

q_table = first_existing(
    artifact, ["q_learning", "Q_v2", "Q_learning", "q_table"], {}
)
sarsa_table = first_existing(
    artifact, ["sarsa", "Q_sarsa", "SARSA", "sarsa_table"], {}
)
scaler = first_existing(artifact, ["scaler", "feature_scaler"])
cluster_model = first_existing(
    artifact, ["region_model", "kmeans", "cluster_model"]
)
distance_matrix = first_existing(
    artifact, ["distance_matrix", "distances"]
)

if distance_matrix is None and cluster_model is not None:
    if hasattr(cluster_model, "cluster_centers_"):
        centers = np.asarray(cluster_model.cluster_centers_, dtype=float)
        distance_matrix = np.linalg.norm(
            centers[:, None, :] - centers[None, :, :], axis=2
        )

if distance_matrix is not None:
    distance_matrix = np.asarray(distance_matrix, dtype=float)

    if (
        distance_matrix.ndim != 2
        or distance_matrix.shape[0] != distance_matrix.shape[1]
        or not np.isfinite(distance_matrix).all()
    ):
        distance_matrix = None

if distance_matrix is not None:
    maximum = float(distance_matrix.max())
    if maximum > 0:
        distance_matrix = distance_matrix / maximum

N_TARGETS = (
    len(distance_matrix) if distance_matrix is not None else 0
)

POLICIES = ["random", "greedy"]

if isinstance(q_table, dict) and q_table:
    POLICIES.append("q_learning")

if isinstance(sarsa_table, dict) and sarsa_table:
    POLICIES.append("sarsa")

RL_READY = distance_matrix is not None and N_TARGETS > 1


# ============================================================
# NASA GIBS
# ============================================================

@st.cache_data(ttl=3600, show_spinner=False, max_entries=24)
def fetch_nasa_image(layer_name, selected_date, region_name):
    west, south, east, north = REGIONS[region_name]
    requested_layer = LAYERS[layer_name]

    layer_candidates = [requested_layer] + [
        layer for layer in LAYERS.values()
        if layer != requested_layer
    ]

    start_date = date.fromisoformat(selected_date)
    errors = []

    width = 1400
    height = max(
        350,
        min(1100, int(width * (north - south) / (east - west))),
    )

    for day_offset in range(4):
        image_date = start_date - timedelta(days=day_offset)

        for layer in layer_candidates:
            params = {
                "SERVICE": "WMS",
                "REQUEST": "GetMap",
                "VERSION": "1.1.1",
                "LAYERS": layer,
                "STYLES": "",
                "SRS": "EPSG:4326",
                "BBOX": f"{west},{south},{east},{north}",
                "WIDTH": width,
                "HEIGHT": height,
                "FORMAT": "image/jpeg",
                "TRANSPARENT": "FALSE",
                "TIME": image_date.isoformat(),
            }

            try:
                response = requests.get(
                    NASA_WMS,
                    params=params,
                    timeout=(8, 25),
                    headers={"User-Agent": "ORBITAL-Earth-Research/2.0"},
                )
                response.raise_for_status()

                if not response.headers.get(
                    "Content-Type", ""
                ).lower().startswith("image/"):
                    raise ValueError("NASA returned a non-image response.")

                image = Image.open(io.BytesIO(response.content)).convert("RGB")

                if image.width < 100 or image.height < 100:
                    raise ValueError("NASA returned an unexpectedly small image.")

                fallback = (
                    layer != requested_layer or day_offset != 0
                )

                return image, layer, image_date.isoformat(), fallback

            except Exception as exc:
                errors.append(
                    f"{layer} {image_date}: {type(exc).__name__}: {exc}"
                )

    raise RuntimeError(
        "NASA imagery was unavailable after fallback attempts.\n"
        + "\n".join(errors[-4:])
    )


# ============================================================
# EUROSAT MSI AND SPECTRAL INDICES
# ============================================================

def to_hwc(image):
    array = np.squeeze(np.asarray(image))

    if array.ndim != 3:
        raise ValueError(
            f"Expected a 3D multispectral image; got {array.shape}."
        )

    # Convert channel-first arrays to channel-last when identifiable.
    if array.shape[0] <= 20 and array.shape[-1] > 20:
        array = np.moveaxis(array, 0, -1)

    if array.shape[-1] < 12:
        raise ValueError(
            f"Expected at least 12 bands; got {array.shape}."
        )

    return array.astype(np.float32)


def percentile_stretch(array):
    array = np.asarray(array, dtype=np.float32)
    output = np.zeros_like(array)

    if array.ndim == 2:
        low, high = np.nanpercentile(array, [2, 98])

        if high > low:
            return np.clip((array - low) / (high - low), 0, 1)

        return output

    for channel in range(array.shape[-1]):
        band = array[..., channel]
        low, high = np.nanpercentile(band, [2, 98])

        if np.isfinite(low) and np.isfinite(high) and high > low:
            output[..., channel] = np.clip(
                (band - low) / (high - low), 0, 1
            )

    return output


def composite(cube, bands):
    cube = to_hwc(cube)

    if max(bands) >= cube.shape[-1]:
        raise ValueError("The image lacks bands needed for this composite.")

    return percentile_stretch(cube[..., bands])


def spectral_indices(cube):
    cube = to_hwc(cube)

    # Assumed zero-based band positions used in this project.
    # Verify the dataset's channel order before scientific use.
    green = cube[..., 2]
    red = cube[..., 3]
    nir = cube[..., 7]
    swir = cube[..., 11]

    ndvi = (nir - red) / (nir + red + EPS)
    ndwi = (green - nir) / (green + nir + EPS)
    ndbi = (swir - nir) / (swir + nir + EPS)

    return {
        "NDVI": np.nan_to_num(ndvi, nan=0.0, posinf=0.0, neginf=0.0),
        "NDWI": np.nan_to_num(ndwi, nan=0.0, posinf=0.0, neginf=0.0),
        "NDBI": np.nan_to_num(ndbi, nan=0.0, posinf=0.0, neginf=0.0),
    }


def extract_features(cube):
    cube = to_hwc(cube)
    means = np.nanmean(cube[..., :13], axis=(0, 1))
    indices = spectral_indices(cube)

    features = list(means)

    for name in ("NDVI", "NDWI", "NDBI"):
        features.extend([
            float(np.nanmean(indices[name])),
            float(np.nanstd(indices[name])),
        ])

    return np.nan_to_num(
        np.asarray(features, dtype=np.float32)
    ), indices


@st.cache_data(ttl=86400, show_spinner=False, max_entries=8)
def load_eurosat_samples(sample_count, seed):
    from datasets import load_dataset

    dataset = load_dataset(
        "blanchon/EuroSAT_MSI",
        split="train",
        streaming=True,
    )

    try:
        class_names = dataset.features["label"].names
    except Exception:
        class_names = None

    dataset = dataset.shuffle(
        seed=int(seed),
        buffer_size=1000,
    )

    samples = []

    for row in dataset:
        if row.get("image") is None:
            continue

        samples.append({
            "image": np.asarray(row["image"]),
            "label": row.get("label"),
        })

        if len(samples) >= int(sample_count):
            break

    if not samples:
        raise RuntimeError("The dataset returned no samples.")

    return samples, class_names


def draw_index(array, name, cmap):
    fig, ax = plt.subplots(figsize=(4, 3.4))
    image = ax.imshow(array, cmap=cmap, vmin=-1, vmax=1)
    ax.set_title(name)
    ax.axis("off")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    return fig


# ============================================================
# REINFORCEMENT-LEARNING SIMULATION
# ============================================================

class OrbitalEnvironment:
    """Synthetic environment; it does not control real satellites."""

    def __init__(self, distances, budget=6, seed=42):
        self.distances = np.asarray(distances, dtype=float)
        self.n = len(self.distances)
        self.budget_limit = int(budget)
        self.rng = np.random.default_rng(seed)
        self.cloud_risk = np.linspace(0.10, 0.40, self.n)
        self.energy_cost = np.linspace(0.05, 0.20, self.n)
        self.reset()

    def reset(self):
        self.budget = self.budget_limit
        self.visited = set()
        self.last_target = 0
        self.weather = int(self.rng.integers(0, 2))
        return self.state()

    def state(self):
        return (
            self.last_target,
            tuple(sorted(self.visited)),
            self.budget,
            self.weather,
        )

    def step(self, action):
        action = int(action)

        if self.budget <= 0:
            raise ValueError("The observation budget is exhausted.")

        if not 0 <= action < self.n:
            raise ValueError("Invalid target action.")

        self.budget -= 1

        reward = -0.10 - self.energy_cost[action]

        if self.rng.random() < 0.25:
            self.weather = 1 - self.weather

        cloud_probability = self.cloud_risk[action]

        if self.weather:
            cloud_probability = min(0.90, cloud_probability + 0.25)

        success = self.rng.random() >= cloud_probability
        novelty = 0.0

        if not success:
            reward -= 0.25
            outcome = "cloud"
        elif action in self.visited:
            reward -= 0.50
            outcome = "repeated target"
        else:
            novelty = min(
                (self.distances[action, previous]
                 for previous in self.visited),
                default=1.0,
            )
            reward += float(novelty)
            self.visited.add(action)
            outcome = "new target"

        self.last_target = action

        info = {
            "success": success,
            "outcome": outcome,
            "novelty": float(novelty),
            "cloud_probability": float(cloud_probability),
            "weather": self.weather,
        }

        return self.state(), float(reward), self.budget == 0, info


def get_q_values(table, state, count):
    try:
        values = table.get(state)

        if values is None:
            return np.zeros(count)

        values = np.asarray(values, dtype=float).reshape(-1)

        if len(values) != count or not np.isfinite(values).all():
            return np.zeros(count)

        return values

    except Exception:
        return np.zeros(count)


def choose_action(policy, state, rng, distances, tables):
    count = len(distances)
    visited = set(state[1])

    if policy == "random":
        return int(rng.integers(count))

    if policy == "greedy":
        available = [i for i in range(count) if i not in visited]

        if not available:
            return int(rng.integers(count))

        if not visited:
            return int(rng.choice(available))

        novelty = {
            i: min(distances[i, j] for j in visited)
            for i in available
        }

        best = max(novelty.values())

        choices = [
            i for i in available
            if np.isclose(novelty[i], best)
        ]

        return int(rng.choice(choices))

    table = tables.get(policy, {})
    values = get_q_values(table, state, count)
    best_actions = np.flatnonzero(np.isclose(values, values.max()))

    return int(rng.choice(best_actions))


def diversity_score(visited, distances):
    visited = sorted(visited)
    pairs = [
        distances[a, b]
        for i, a in enumerate(visited)
        for b in visited[i + 1:]
    ]

    return float(np.mean(pairs)) if pairs else 0.0


def evaluate_policies(distances, tables, policies, episodes, seed, budget):
    rows = []
    raw_rewards = {}

    for policy in policies:
        rewards = []
        unique_targets = []
        diversities = []
        success_rates = []

        for episode in range(int(episodes)):
            episode_seed = int(seed) + episode * 13

            env = OrbitalEnvironment(
                distances,
                budget=budget,
                seed=episode_seed,
            )

            rng = np.random.default_rng(episode_seed + 1)
            state = env.reset()
            total_reward = 0.0
            successes = 0
            done = False

            while not done:
                action = choose_action(
                    policy, state, rng, distances, tables
                )

                state, reward, done, info = env.step(action)
                total_reward += reward
                successes += int(info["success"])

            rewards.append(total_reward)
            unique_targets.append(len(env.visited))
            diversities.append(
                diversity_score(env.visited, distances)
            )
            success_rates.append(successes / max(1, budget))

        rewards = np.asarray(rewards, dtype=float)
        standard_error = (
            rewards.std(ddof=1) / np.sqrt(len(rewards))
            if len(rewards) > 1 else 0.0
        )

        raw_rewards[policy] = rewards

        rows.append({
            "Policy": POLICY_LABELS[policy],
            "Policy key": policy,
            "Mean reward": float(rewards.mean()),
            "95% CI": float(1.96 * standard_error),
            "Reward std": float(rewards.std()),
            "Unique targets": float(np.mean(unique_targets)),
            "Spectral diversity": float(np.mean(diversities)),
            "Success rate": float(np.mean(success_rates)),
        })

    return pd.DataFrame(rows), raw_rewards


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.markdown("## 🌍 ORBITAL")
    st.caption("EARTH INTELLIGENCE / RESEARCH PROTOTYPE")
    st.divider()

    st.markdown("### Satellite imagery")

    max_date = date.today() - timedelta(days=1)

    selected_date = st.date_input(
        "Image date",
        value=max_date,
        min_value=date(2002, 1, 1),
        max_value=max_date,
    )

    selected_layer = st.selectbox("Satellite", list(LAYERS))
    selected_region = st.selectbox("Region", list(REGIONS))

    st.divider()
    st.markdown("### Saved models")

    if artifact_path and not artifact_error:
        st.success(f"Loaded {Path(artifact_path).name}")
    else:
        st.warning(artifact_error or "Model artifact unavailable.")

    left, right = st.columns(2)

    left.metric(
        "Q states",
        len(q_table) if isinstance(q_table, dict) else 0,
    )

    right.metric(
        "SARSA states",
        len(sarsa_table) if isinstance(sarsa_table, dict) else 0,
    )

    if artifact_error:
        with st.expander("Artifact diagnostics"):
            st.code(artifact_error)

    st.divider()
    st.markdown(f"[NASA Worldview ↗]({WORLDVIEW_URL})")
    st.markdown(f"[EuroSAT MSI ↗]({EUROSAT_URL})")
    st.caption("Not affiliated with NASA.")


# ============================================================
# HERO
# ============================================================

st.markdown("""
<div class="orbital-hero">
  <div class="orbital-eyebrow">
    Earth observation / Remote sensing / Reinforcement learning
  </div>
  <div class="orbital-title">ORBITAL</div>
  <div class="orbital-subtitle">
    Explore satellite imagery, inspect multispectral data, study
    spectral clusters, and evaluate tabular agents in a simulated
    observation environment.
  </div>
</div>
""", unsafe_allow_html=True)

tabs = st.tabs([
    "Overview",
    "Earth from Orbit",
    "Spectral Lab",
    "Target Explorer",
    "Mission Simulator",
    "Policy Benchmark",
    "Methods",
])


# ============================================================
# TAB 1 — OVERVIEW
# ============================================================

with tabs[0]:
    st.subheader("The research question")

    st.write(
        "Can an observation policy select spectrally distinct targets "
        "under a limited observation budget and simulated cloud risk? "
        "ORBITAL compares random selection, greedy novelty, and saved "
        "Q-learning/SARSA policies."
    )

    a, b, c, d = st.columns(4)

    a.metric("Spectral targets", N_TARGETS or "Unavailable")
    b.metric("Input bands", 13)
    c.metric("Default budget", "6 passes")
    d.metric("Weather states", 2)

    st.markdown("### Research pipeline")

    p1, p2, p3, p4 = st.columns(4)

    p1.markdown("**01 · Extract**\n\nBand means and spectral indices.")
    p2.markdown("**02 · Cluster**\n\nGroup patches in feature space.")
    p3.markdown("**03 · Simulate**\n\nBudget, novelty, and cloud risk.")
    p4.markdown("**04 · Evaluate**\n\nCompare policy performance.")

    if not RL_READY:
        st.warning(
            "The saved target-distance matrix is unavailable. "
            "The imagery and spectral tabs can still run independently."
        )


# ============================================================
# TAB 2 — EARTH IMAGERY
# ============================================================

with tabs[1]:
    st.subheader("Earth from orbit")

    st.caption(
        "NASA GIBS browse imagery. This is not live video or "
        "street-level imagery."
    )

    if st.button("Load / refresh NASA imagery", type="primary"):
        st.session_state.pop("nasa_image_result", None)
        st.session_state.pop("nasa_image_error", None)

    if (
        "nasa_image_result" not in st.session_state
        and "nasa_image_error" not in st.session_state
    ):
        with st.spinner("Requesting imagery from NASA GIBS..."):
            try:
                st.session_state["nasa_image_result"] = fetch_nasa_image(
                    selected_layer,
                    selected_date.isoformat(),
                    selected_region,
                )
            except Exception as exc:
                st.session_state["nasa_image_error"] = (
                    f"{type(exc).__name__}: {exc}"
                )

    result = st.session_state.get("nasa_image_result")

    if result:
        image, actual_layer, actual_date, fallback = result

        st.image(
            image,
            caption=(
                f"NASA GIBS · {actual_layer} · {actual_date} · "
                f"{selected_region}"
            ),
            width="stretch",
        )

        if fallback:
            st.info(
                "The requested product or date was unavailable. "
                "A fallback layer or date is displayed."
            )

        output = io.BytesIO()
        image.save(output, format="JPEG", quality=92)

        st.download_button(
            "Download image",
            data=output.getvalue(),
            file_name=f"orbital_nasa_{actual_date}.jpg",
            mime="image/jpeg",
        )

    elif st.session_state.get("nasa_image_error"):
        st.warning(
            "NASA imagery could not be loaded. "
            "The other dashboard tabs remain available."
        )

        with st.expander("Technical details"):
            st.code(st.session_state["nasa_image_error"])

    st.link_button("Open NASA Worldview ↗", WORLDVIEW_URL)


# ============================================================
# TAB 3 — SPECTRAL LAB
# ============================================================

with tabs[2]:
    st.subheader("Multispectral analysis lab")

    st.write(
        "Inspect EuroSAT MSI samples, build band composites, and "
        "calculate NDVI, NDWI, and NDBI. Results depend on correct "
        "band ordering and should be treated as exploratory."
    )

    left, right = st.columns(2)

    sample_count = left.slider(
        "Samples to load", 8, 40, 24, step=4
    )

    seed = right.number_input(
        "Shuffle seed", min_value=0, max_value=99999, value=7
    )

    try:
        with st.spinner("Loading EuroSAT MSI..."):
            samples, class_names = load_eurosat_samples(
                int(sample_count), int(seed)
            )

    except Exception as exc:
        samples, class_names = None, None

        st.warning(
            "EuroSAT could not be loaded. Check the network connection "
            "or Hugging Face dataset availability."
        )

        with st.expander("Dataset error"):
            st.code(f"{type(exc).__name__}: {exc}")

        st.link_button("Open EuroSAT dataset ↗", EUROSAT_URL)

    if samples:
        selected_index = st.selectbox(
            "Sample patch",
            range(len(samples)),
            format_func=lambda index: f"Patch {index + 1}",
        )

        sample = samples[selected_index]

        try:
            cube = to_hwc(sample["image"])
            features, indices = extract_features(cube)

            label = sample["label"]

            if (
                isinstance(label, (int, np.integer))
                and class_names
                and int(label) < len(class_names)
            ):
                label_text = str(class_names[int(label)])
            else:
                label_text = str(label)

            metrics = st.columns(5)

            metrics[0].metric("Label", label_text)
            metrics[1].metric(
                "Mean NDVI", f"{indices['NDVI'].mean():.3f}"
            )
            metrics[2].metric(
                "Mean NDWI", f"{indices['NDWI'].mean():.3f}"
            )
            metrics[3].metric(
                "Mean NDBI", f"{indices['NDBI'].mean():.3f}"
            )

            if scaler is not None and cluster_model is not None:
                try:
                    scaled = scaler.transform(features.reshape(1, -1))
                    cluster = int(cluster_model.predict(scaled)[0])
                    metrics[4].metric("Cluster", f"T{cluster}")
                except Exception:
                    metrics[4].metric("Cluster", "Unavailable")
            else:
                metrics[4].metric("Cluster", "No model")

            st.markdown("### Spectral composites")

            col1, col2, col3 = st.columns(3)

            with col1:
                st.image(
                    composite(cube, [3, 2, 1]),
                    caption="True-colour approximation (R/G/B)",
                    width="stretch",
                )

            with col2:
                st.image(
                    composite(cube, [7, 3, 2]),
                    caption="False colour (NIR/R/G)",
                    width="stretch",
                )

            with col3:
                st.image(
                    composite(cube, [11, 7, 3]),
                    caption="SWIR/NIR/Red",
                    width="stretch",
                )

            st.markdown("### Spectral index maps")

            index_cols = st.columns(3)

            for column, name, cmap in zip(
                index_cols,
                ["NDVI", "NDWI", "NDBI"],
                ["RdYlGn", "BrBG", "PuOr"],
            ):
                with column:
                    fig = draw_index(indices[name], name, cmap)
                    st.pyplot(fig, width="stretch")
                    plt.close(fig)

            st.markdown("### Spectral signature")

            fig, ax = plt.subplots(figsize=(10, 3.5))

            ax.plot(
                range(1, min(14, len(features) + 1)),
                features[:13],
                marker="o",
                linewidth=2,
                color="#7ce0ca",
            )

            ax.set_xticks(range(1, 14))
            ax.set_xticklabels(BAND_NAMES)
            ax.set_xlabel("Band index")
            ax.set_ylabel("Mean band value")
            ax.grid(alpha=0.3)

            fig.tight_layout()
            st.pyplot(fig, width="stretch")
            plt.close(fig)

            feature_table = pd.DataFrame({
                "Feature": [
                    *BAND_NAMES,
                    "NDVI_mean", "NDVI_std",
                    "NDWI_mean", "NDWI_std",
                    "NDBI_mean", "NDBI_std",
                ][:len(features)],
                "Value": features,
            })

            with st.expander("Inspect extracted features"):
                st.dataframe(
                    feature_table.round(6),
                    width="stretch",
                    hide_index=True,
                )

            st.download_button(
                "Download features CSV",
                data=feature_table.to_csv(index=False).encode("utf-8"),
                file_name=f"orbital_features_{selected_index + 1}.csv",
                mime="text/csv",
            )

        except Exception as exc:
            st.error("This sample could not be processed.")

            with st.expander("Processing details"):
                st.code(f"{type(exc).__name__}: {exc}")


# ============================================================
# TAB 4 — TARGET EXPLORER
# ============================================================

with tabs[3]:
    st.subheader("Spectral target explorer")

    st.caption(
        "Target IDs represent clusters in feature space, "
        "not geographic coordinates."
    )

    if (
        cluster_model is not None
        and hasattr(cluster_model, "cluster_centers_")
    ):
        centers = np.asarray(cluster_model.cluster_centers_, dtype=float)

        if scaler is not None:
            try:
                original_centers = scaler.inverse_transform(centers)
            except Exception:
                original_centers = centers
        else:
            original_centers = centers

        if original_centers.shape[1] >= 13:
            count = min(13, original_centers.shape[1])

            fig, ax = plt.subplots(figsize=(10, 4))

            for index, center in enumerate(original_centers):
                ax.plot(
                    range(1, count + 1),
                    center[:count],
                    marker="o",
                    label=f"T{index}",
                )

            ax.set_title("Cluster-centre spectral signatures")
            ax.set_xlabel("Band")
            ax.set_ylabel("Mean feature value")
            ax.grid(alpha=0.3)
            ax.legend(ncol=4)

            fig.tight_layout()
            st.pyplot(fig, width="stretch")
            plt.close(fig)

            rows = []

            for index, center in enumerate(original_centers):
                rows.append({
                    "Target": f"T{index}",
                    **{
                        BAND_NAMES[i]: float(center[i])
                        for i in range(min(13, len(center)))
                    },
                })

            st.dataframe(
                pd.DataFrame(rows).round(4),
                width="stretch",
                hide_index=True,
            )

    else:
        st.info("No compatible clustering model was found in the artifact.")

    if distance_matrix is not None:
        fig, ax = plt.subplots(figsize=(6, 5))

        heatmap = ax.imshow(distance_matrix, cmap="viridis")
        ax.set_title("Pairwise target distances")
        ax.set_xlabel("Target")
        ax.set_ylabel("Target")

        fig.colorbar(heatmap, ax=ax)
        fig.tight_layout()

        st.pyplot(fig, width="stretch")
        plt.close(fig)


# ============================================================
# TAB 5 — MISSION SIMULATOR
# ============================================================

with tabs[4]:
    st.subheader("Mission simulator")

    if not RL_READY:
        st.info(
            "A valid saved target-distance matrix is needed "
            "to run the simulation."
        )

    else:
        st.warning(
            "This is a synthetic simulation. It does not issue satellite "
            "commands or model actual weather, orbit, or energy telemetry."
        )

        c1, c2 = st.columns(2)

        policy = c1.selectbox(
            "Policy",
            POLICIES,
            format_func=lambda value: POLICY_LABELS[value],
            key="mission_policy",
        )

        mission_seed = int(c2.number_input(
            "Mission seed",
            min_value=0,
            max_value=99999,
            value=42,
            key="mission_seed",
        ))

        table_map = {
            "q_learning": q_table if isinstance(q_table, dict) else {},
            "sarsa": sarsa_table if isinstance(sarsa_table, dict) else {},
        }

        mission_key = f"{policy}-{mission_seed}-{N_TARGETS}"

        if (
            st.button("Reset mission")
            or st.session_state.get("mission_key") != mission_key
        ):
            env = OrbitalEnvironment(
                distance_matrix, budget=6, seed=mission_seed
            )

            st.session_state["mission"] = {
                "env": env,
                "state": env.reset(),
                "rng": np.random.default_rng(mission_seed + 1),
                "log": [],
                "total_reward": 0.0,
                "done": False,
            }

            st.session_state["mission_key"] = mission_key

        mission = st.session_state["mission"]
        env = mission["env"]

        def run_one_pass():
            state = mission["state"]

            action = choose_action(
                policy,
                state,
                mission["rng"],
                distance_matrix,
                table_map,
            )

            next_state, reward, done, info = env.step(action)

            mission["log"].append({
                "Pass": len(mission["log"]) + 1,
                "Target": f"T{action}",
                "Weather": "Cloudy" if info["weather"] else "Clear",
                "Cloud probability": round(
                    info["cloud_probability"], 3
                ),
                "Outcome": info["outcome"],
                "Novelty": round(info["novelty"], 3),
                "Reward": round(reward, 3),
            })

            mission["state"] = next_state
            mission["done"] = done
            mission["total_reward"] += reward

        step_col, run_col = st.columns(2)

        step_clicked = step_col.button(
            "▶ Next pass",
            disabled=mission["done"],
        )

        run_clicked = run_col.button(
            "⏭ Run to end",
            disabled=mission["done"],
        )

        if step_clicked and not mission["done"]:
            run_one_pass()

        if run_clicked and not mission["done"]:
            while not mission["done"]:
                run_one_pass()

        metrics = st.columns(4)

        metrics[0].metric("Passes remaining", env.budget)
        metrics[1].metric(
            "Unique targets", f"{len(env.visited)} / {N_TARGETS}"
        )
        metrics[2].metric(
            "Total reward", f"{mission['total_reward']:.3f}"
        )
        metrics[3].metric(
            "Weather", "Cloudy" if env.weather else "Clear"
        )

        if mission["log"]:
            st.dataframe(
                pd.DataFrame(mission["log"]),
                width="stretch",
                hide_index=True,
            )

        if mission["done"]:
            st.success("Simulation completed.")


# ============================================================
# TAB 6 — POLICY BENCHMARK
# ============================================================

with tabs[5]:
    st.subheader("Policy benchmark")

    if not RL_READY:
        st.info(
            "A valid saved target-distance matrix is needed "
            "for policy benchmarking."
        )

    else:
        st.write(
            "Evaluate policies over repeated synthetic missions. "
            "Confidence intervals describe simulation variability, "
            "not real-world satellite performance."
        )

        c1, c2, c3 = st.columns(3)

        episodes = c1.select_slider(
            "Episodes per policy",
            options=[100, 250, 500, 1000, 2000],
            value=500,
        )

        eval_seed = int(c2.number_input(
            "Evaluation seed",
            min_value=0,
            max_value=99999,
            value=2026,
            key="benchmark_seed",
        ))

        budget = c3.slider("Observation budget", 3, 10, 6)

        if budget != 6:
            st.info(
                "The saved Q-learning/SARSA policies may have been "
                "trained with a six-pass budget. Other budgets are "
                "generalisation tests."
            )

        if st.button("Run policy benchmark", type="primary"):
            with st.spinner("Evaluating policies..."):
                results, raw_rewards = evaluate_policies(
                    distance_matrix,
                    {
                        "q_learning": (
                            q_table if isinstance(q_table, dict) else {}
                        ),
                        "sarsa": (
                            sarsa_table if isinstance(sarsa_table, dict) else {}
                        ),
                    },
                    POLICIES,
                    int(episodes),
                    eval_seed,
                    int(budget),
                )

            st.session_state["benchmark_results"] = results
            st.session_state["benchmark_rewards"] = raw_rewards

        results = st.session_state.get("benchmark_results")
        raw_rewards = st.session_state.get("benchmark_rewards")

        if results is not None:
            labels = results["Policy"].tolist()
            colors = [
                POLICY_COLORS[key] for key in results["Policy key"]
            ]

            fig, axes = plt.subplots(1, 3, figsize=(14, 4))

            axes[0].bar(
                labels,
                results["Mean reward"],
                yerr=results["95% CI"],
                color=colors,
                capsize=4,
            )
            axes[0].set_title("Mean reward ± 95% CI")

            axes[1].bar(
                labels,
                results["Unique targets"],
                color=colors,
            )
            axes[1].set_title("Mean unique targets")

            axes[2].bar(
                labels,
                results["Spectral diversity"],
                color=colors,
            )
            axes[2].set_title("Spectral diversity")

            for ax in axes:
                ax.grid(axis="y", alpha=0.3)
                ax.tick_params(axis="x", rotation=15)

            fig.tight_layout()
            st.pyplot(fig, width="stretch")
            plt.close(fig)

            st.dataframe(
                results.drop(columns=["Policy key"]).round(4),
                width="stretch",
                hide_index=True,
            )

            if isinstance(raw_rewards, dict):
                fig, ax = plt.subplots(figsize=(9, 3.5))

                for key, values in raw_rewards.items():
                    ax.hist(
                        values,
                        bins=30,
                        alpha=0.45,
                        label=POLICY_LABELS[key],
                        density=True,
                    )

                ax.set_title("Distribution of episode rewards")
                ax.set_xlabel("Episode reward")
                ax.set_ylabel("Density")
                ax.legend()
                ax.grid(alpha=0.3)

                fig.tight_layout()
                st.pyplot(fig, width="stretch")
                plt.close(fig)

            csv = results.drop(
                columns=["Policy key"]
            ).to_csv(index=False).encode("utf-8")

            st.download_button(
                "Download benchmark results",
                data=csv,
                file_name="orbital_benchmark.csv",
                mime="text/csv",
            )

        else:
            st.info(
                "Choose the evaluation settings and click "
                "**Run policy benchmark**."
            )


# ============================================================
# TAB 7 — METHODS
# ============================================================

with tabs[6]:
    st.subheader("Methods, data, and limitations")

    with st.expander("NASA imagery", expanded=True):
        st.write(
            "NASA GIBS WMS provides browse imagery. The application "
            "tries alternative layers and recent dates if the requested "
            "image cannot be fetched. NASA Worldview opens separately."
        )

    with st.expander("EuroSAT MSI and spectral indices"):
        st.markdown("""
- Samples are streamed from the `blanchon/EuroSAT_MSI` dataset.
- The project extracts 13 band means plus mean and standard deviation
  for NDVI, NDWI, and NDBI.
- Index calculations depend on the correct channel order.
- Composite band positions should be verified against the dataset metadata.
- Spectral indices are exploratory measurements, not ground-truth classifications.
""")

    with st.expander("Clustering and reinforcement learning"):
        st.markdown("""
- Clusters represent feature-space groups, not geographic coordinates.
- Cloud probability and energy cost are simulated.
- Saved tabular agents may encounter states not present during training.
- Benchmark results describe the simulator only.
- A single run does not establish statistical superiority.
""")

    with st.expander("Model artifact"):
        st.write(f"Artifact: `{artifact_path or 'not found'}`")
        st.write(
            f"Q-learning entries: "
            f"{len(q_table) if isinstance(q_table, dict) else 0}"
        )
        st.write(
            f"SARSA entries: "
            f"{len(sarsa_table) if isinstance(sarsa_table, dict) else 0}"
        )

        if artifact_error:
            st.error(artifact_error)

        st.caption(
            "Pickle artifacts should only be loaded from trusted sources. "
            "Use a compatible scikit-learn version when loading saved estimators."
        )

    with st.expander("Data sources"):
        st.markdown(
            f"- [NASA Worldview / GIBS]({WORLDVIEW_URL})\n"
            f"- [EuroSAT MSI on Hugging Face]({EUROSAT_URL})"
        )


st.divider()

st.caption(
    "ORBITAL · Earth-observation research prototype · "
    "Not affiliated with NASA · Not operational satellite guidance."
)
