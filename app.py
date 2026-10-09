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
# ORBITAL v2
# Reinforcement learning for information-efficient Earth observation
# Research prototype - not operational satellite guidance
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
EUROSAT_URL = "https://huggingface.co/datasets/blanchon/EuroSAT_MSI"
EPS = 1e-8

NASA_LAYERS = {
    "VIIRS - Suomi NPP True Color": "VIIRS_SNPP_CorrectedReflectance_TrueColor",
    "MODIS - Terra True Color": "MODIS_Terra_CorrectedReflectance_TrueColor",
    "MODIS - Aqua True Color": "MODIS_Aqua_CorrectedReflectance_TrueColor",
}

# (west, south, east, north)
REGIONS = {
    "Global": (-180, -90, 180, 90),
    "Europe": (-12, 34, 40, 72),
    "South Asia": (60, 5, 100, 38),
    "Africa": (-20, -36, 55, 38),
    "North America": (-170, 12, -50, 75),
    "South America": (-85, -57, -32, 14),
    "Australia & Oceania": (110, -50, 180, -5),
}

# Palette used in every chart so the dashboard feels consistent
BG = "#0b1928"
PANEL = "#102235"
GRID = "#223b50"
TEXT = "#c5d6e3"
ACCENT = "#7ce0ca"
COLORS = {
    "random": "#8caabd",
    "greedy": "#f2b66d",
    "q_learning": "#7ce0ca",
    "sarsa": "#8fa8ff",
}
POLICY_LABELS = {
    "random": "Random",
    "greedy": "Greedy novelty",
    "q_learning": "Q-learning",
    "sarsa": "SARSA",
}

plt.rcParams.update({
    "figure.facecolor": BG,
    "axes.facecolor": PANEL,
    "axes.edgecolor": GRID,
    "axes.labelcolor": TEXT,
    "axes.titlecolor": "#eaf4ff",
    "xtick.color": TEXT,
    "ytick.color": TEXT,
    "grid.color": GRID,
    "text.color": TEXT,
    "legend.facecolor": PANEL,
    "legend.edgecolor": GRID,
    "font.size": 10,
})

# ============================================================
# STYLING
# ============================================================

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Space+Grotesk:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] { font-family: 'Space Grotesk', sans-serif; }

    .stApp {
        background:
            radial-gradient(ellipse at 15% 0%, #122b43 0%, transparent 38%),
            linear-gradient(180deg, #07111e 0%, #091522 100%);
        color: #e7f1fa;
    }
    section[data-testid="stSidebar"] {
        background: #0a1624; border-right: 1px solid #21364a;
    }
    .orbital-hero {
        padding: 28px 30px; border: 1px solid #28445b; border-radius: 20px;
        background: linear-gradient(130deg, #10283d, #0b1928 65%, #122b37);
        margin-bottom: 18px;
    }
    .orbital-eyebrow {
        color: #78d9c3; font-family: 'DM Mono', monospace; font-size: 12px;
        letter-spacing: 2px; text-transform: uppercase;
    }
    .orbital-title {
        font-size: clamp(38px, 6vw, 64px); line-height: 1; font-weight: 700;
        letter-spacing: -3px; margin: 12px 0; color: #f3f8ff;
    }
    .orbital-subtitle { color: #a8bfd1; font-size: 16px; max-width: 800px; line-height: 1.6; }
    .orbital-callout {
        border-left: 3px solid #78d9c3; background: rgba(120,217,195,0.07);
        padding: 12px 16px; border-radius: 0 10px 10px 0; color: #c5d6e3;
        margin: 10px 0 16px 0;
    }
    .orbital-callout.warn { border-left-color: #f2b66d; background: rgba(242,182,109,0.07); }
    h1, h2, h3, h4 { color: #eaf4ff !important; }
    p, li, label { color: #c5d6e3; }
    div[data-testid="stMetric"] {
        background: #102235; padding: 14px; border: 1px solid #223b50; border-radius: 12px;
    }
    div[data-testid="stMetricValue"] { color: #7ce0ca; }
    .stButton > button, .stLinkButton > a { border-radius: 10px; font-weight: 600; }
    a { color: #79dbc8 !important; }
    button[data-baseweb="tab"] { font-weight: 600; }
    hr { border-color: #223b50; }
    </style>
    """,
    unsafe_allow_html=True,
)


def callout(text, warn=False):
    cls = "orbital-callout warn" if warn else "orbital-callout"
    st.markdown(f'<div class="{cls}">{text}</div>', unsafe_allow_html=True)


def show_image(image, caption=None):
    """Works on both newer and older Streamlit versions."""
    try:
        st.image(image, caption=caption, width="stretch")
    except TypeError:
        st.image(image, caption=caption, use_container_width=True)


def show_df(df, **kwargs):
    try:
        st.dataframe(df, width="stretch", hide_index=True, **kwargs)
    except TypeError:
        st.dataframe(df, use_container_width=True, hide_index=True, **kwargs)


# ============================================================
# MODEL ARTIFACT
# ============================================================

@st.cache_resource(show_spinner=False)
def load_model_artifact():
    candidates = [
        BASE_DIR / "orbital_models.pkl",
        BASE_DIR / "orbital_models" / "orbital_models.pkl",
        BASE_DIR / "orbital_models.pickle",
        BASE_DIR / "orbital_models" / "orbital_models.pickle",
    ]
    for path in candidates:
        if path.exists():
            try:
                with path.open("rb") as file:
                    artifact = pickle.load(file)
                if isinstance(artifact, dict):
                    return artifact, str(path), None
                return {}, str(path), "Artifact exists but is not a dictionary."
            except Exception as exc:
                return {}, str(path), f"{type(exc).__name__}: {exc}"
    return {}, None, "No model artifact found in the expected locations."


def get_field(artifact, names, default=None):
    for name in names:
        if name in artifact and artifact[name] is not None:
            return artifact[name]
    return default


artifact, artifact_path, artifact_error = load_model_artifact()

q_table = get_field(artifact, ["q_learning", "Q_v2", "Q_learning", "q_table"], {})
sarsa_table = get_field(artifact, ["sarsa", "Q_sarsa", "SARSA", "sarsa_table"], {})
scaler = get_field(artifact, ["scaler", "feature_scaler"])
cluster_model = get_field(artifact, ["region_model", "kmeans", "cluster_model"])
distance_matrix = get_field(artifact, ["distance_matrix", "distances"])
# Optional: save these from the notebook to get training curves here
q_train_rewards = get_field(artifact, ["q_learning_rewards", "episode_rewards_v2"])
sarsa_train_rewards = get_field(artifact, ["sarsa_rewards"])

# If the distance matrix was not saved, rebuild it from cluster centres
if distance_matrix is None and cluster_model is not None and hasattr(cluster_model, "cluster_centers_"):
    c = np.asarray(cluster_model.cluster_centers_)
    d = np.linalg.norm(c[:, None, :] - c[None, :, :], axis=2)
    distance_matrix = d / d.max() if d.max() > 0 else d

distance_matrix = None if distance_matrix is None else np.asarray(distance_matrix, dtype=float)
RL_READY = distance_matrix is not None and distance_matrix.ndim == 2 and len(q_table) > 0
N_TARGETS = len(distance_matrix) if distance_matrix is not None else 0

POLICIES = ["random", "greedy"]
if len(q_table):
    POLICIES.append("q_learning")
if len(sarsa_table):
    POLICIES.append("sarsa")

# ============================================================
# SIMULATION ENVIRONMENT (mirrors OrbitalEnvV2 from the notebook)
# ============================================================

class OrbitalEnvV2:
    def __init__(self, distance_matrix, budget=6, observation_cost=0.10, seed=42):
        self.distances = np.asarray(distance_matrix)
        self.n_targets = len(self.distances)
        self.initial_budget = budget
        self.observation_cost = observation_cost
        self.rng = np.random.default_rng(seed)
        self.cloud_risk = np.linspace(0.10, 0.40, self.n_targets)
        self.energy_cost = np.linspace(0.05, 0.20, self.n_targets)
        self.reset()

    def reset(self):
        self.budget = self.initial_budget
        self.visited = set()
        self.last_target = 0
        self.weather = int(self.rng.integers(0, 2))
        return self.state()

    def state(self):
        return (self.last_target, tuple(sorted(self.visited)), self.budget, self.weather)

    def step(self, action):
        action = int(action)
        self.budget -= 1
        reward = -self.observation_cost - self.energy_cost[action]

        if self.rng.random() < 0.25:
            self.weather = 1 - self.weather

        cloud_p = self.cloud_risk[action]
        if self.weather == 1:
            cloud_p = min(0.90, cloud_p + 0.25)

        success = self.rng.random() >= cloud_p
        novelty = 0.0
        outcome = "cloud"
        if not success:
            reward -= 0.25
        elif action in self.visited:
            reward -= 0.50
            outcome = "redundant"
        else:
            novelty = (
                min(self.distances[action, p] for p in self.visited)
                if self.visited else 1.0
            )
            reward += float(novelty)
            self.visited.add(action)
            outcome = "new"

        self.last_target = action
        done = self.budget == 0
        info = {
            "success": bool(success),
            "outcome": outcome,
            "novelty": float(novelty),
            "cloud_probability": float(cloud_p),
            "weather": self.weather,
        }
        return self.state(), float(reward), done, info


def q_values(table, state, n):
    values = table.get(state)
    return np.zeros(n) if values is None else np.asarray(values, dtype=float)


def choose_action(policy, state, rng, n, dist, tables):
    if policy == "random":
        return int(rng.integers(n))

    if policy == "greedy":
        visited = set(state[1])
        available = [i for i in range(n) if i not in visited]
        if not available:
            return int(rng.integers(n))
        if not visited:
            return int(rng.choice(available))
        novelty = {i: min(dist[i, j] for j in visited) for i in available}
        top = max(novelty.values())
        return int(rng.choice([i for i, v in novelty.items() if np.isclose(v, top)]))

    values = q_values(tables[policy], state, n)
    best = np.flatnonzero(np.isclose(values, values.max()))
    return int(rng.choice(best))


def pairwise_diversity(chosen, dist):
    chosen = sorted(chosen)
    if len(chosen) < 2:
        return 0.0
    return float(np.mean([dist[a, b] for i, a in enumerate(chosen) for b in chosen[i + 1:]]))


@st.cache_data(show_spinner=False)
def run_benchmark(_dist, _tables, policies, n_episodes, seed, budget, fingerprint):
    """Evaluate every policy on identical environment seeds."""
    n = len(_dist)
    rows, raw = [], {}
    for policy in policies:
        env = OrbitalEnvV2(_dist, budget=budget, seed=seed)
        rng = np.random.default_rng(seed + 1)
        returns, uniques, divers, successes = [], [], [], []
        for _ in range(n_episodes):
            state = env.reset()
            total, done, ok = 0.0, False, 0
            while not done:
                action = choose_action(policy, state, rng, n, _dist, _tables)
                state, reward, done, info = env.step(action)
                total += reward
                ok += int(info["success"])
            returns.append(total)
            uniques.append(len(env.visited))
            divers.append(pairwise_diversity(env.visited, _dist))
            successes.append(ok / budget)
        returns = np.asarray(returns)
        raw[policy] = returns
        rows.append({
            "Policy": POLICY_LABELS[policy],
            "key": policy,
            "Mean reward": returns.mean(),
            "95% CI": 1.96 * returns.std(ddof=1) / np.sqrt(n_episodes),
            "Reward std": returns.std(ddof=1),
            "Unique targets": float(np.mean(uniques)),
            "Spectral diversity": float(np.mean(divers)),
            "Obs. success rate": float(np.mean(successes)),
        })
    return pd.DataFrame(rows), raw


# ============================================================
# NASA GIBS
# ============================================================

def validate_image_response(response):
    response.raise_for_status()
    content_type = response.headers.get("Content-Type", "").lower()
    if not content_type.startswith("image/"):
        raise ValueError(f"Expected an image, received Content-Type: {content_type or 'unknown'}")
    image = Image.open(io.BytesIO(response.content))
    image.load()
    if image.width < 100 or image.height < 100:
        raise ValueError(f"Unexpectedly small image: {image.width} x {image.height}")
    return image.convert("RGB")


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_nasa_image(layer_name, selected_date, region_name):
    preferred = NASA_LAYERS[layer_name]
    others = [v for v in NASA_LAYERS.values() if v != preferred]
    day0 = date.fromisoformat(str(selected_date))
    west, south, east, north = REGIONS[region_name]
    width = 1600
    height = max(300, min(1600, round(width * (north - south) / (east - west))))

    attempts = []
    for offset in range(0, 4):
        day = day0 - timedelta(days=offset)
        attempts.append((preferred, day))
        attempts.extend((alt, day) for alt in others)

    errors = []
    for layer, day in attempts:
        if day < date(2002, 1, 1):
            continue
        params = {
            "SERVICE": "WMS", "REQUEST": "GetMap", "VERSION": "1.1.1",
            "LAYERS": layer, "STYLES": "", "SRS": "EPSG:4326",
            "BBOX": f"{west},{south},{east},{north}",
            "WIDTH": str(width), "HEIGHT": str(height),
            "FORMAT": "image/jpeg", "TRANSPARENT": "FALSE", "TIME": day.isoformat(),
        }
        try:
            response = requests.get(
                NASA_WMS, params=params, timeout=30,
                headers={"User-Agent": "ORBITAL-Earth-Observation-Research/2.0"},
            )
            image = validate_image_response(response)
            return image, layer, day.isoformat(), (layer != preferred or day != day0)
        except Exception as exc:
            errors.append(f"{layer} {day}: {type(exc).__name__}: {exc}")

    raise RuntimeError(
        "NASA GIBS did not return a valid image after several attempts.\n"
        + "\n".join(errors[-5:])
    )


# ============================================================
# MULTISPECTRAL UTILITIES
# ============================================================

FEATURE_NAMES = (
    ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B9", "B10", "B11", "B12"]
    + ["NDVI_mean", "NDVI_std", "NDWI_mean", "NDWI_std", "NDBI_mean", "NDBI_std"]
)
EUROSAT_CLASSES = [
    "AnnualCrop", "Forest", "HerbaceousVegetation", "Highway", "Industrial",
    "Pasture", "PermanentCrop", "Residential", "River", "SeaLake",
]


def image_hwc(value):
    arr = np.squeeze(np.asarray(value))
    if arr.ndim != 3:
        raise ValueError(f"Expected a 3D multispectral image, received shape {arr.shape}")
    if arr.shape[0] <= 20 and arr.shape[-1] > 20:
        arr = np.moveaxis(arr, 0, -1)
    if arr.shape[-1] < 12:
        raise ValueError(f"Need at least 12 bands, received shape {arr.shape}")
    return arr.astype(np.float32)


def stretch(array):
    array = np.asarray(array, dtype=np.float32)
    out = np.zeros_like(array)
    if array.ndim == 2:
        lo, hi = np.nanpercentile(array, [2, 98])
        return np.clip((array - lo) / (hi - lo), 0, 1) if hi > lo else out
    for b in range(array.shape[-1]):
        ch = array[..., b]
        lo, hi = np.nanpercentile(ch, [2, 98])
        if hi > lo:
            out[..., b] = np.clip((ch - lo) / (hi - lo), 0, 1)
    return out


def make_composite(cube, band_indices):
    return stretch(cube[..., list(band_indices)])


def calculate_indices(cube):
    green, red, nir, swir = cube[..., 2], cube[..., 3], cube[..., 7], cube[..., 11]
    return {
        "NDVI": (nir - red) / (nir + red + EPS),
        "NDWI": (green - nir) / (green + nir + EPS),
        "NDBI": (swir - nir) / (swir + nir + EPS),
    }


def calculate_features(cube):
    feats = list(np.nanmean(cube[..., :13], axis=(0, 1)))
    indices = calculate_indices(cube)
    for name in ["NDVI", "NDWI", "NDBI"]:
        feats.extend([float(np.nanmean(indices[name])), float(np.nanstd(indices[name]))])
    return np.asarray(feats, dtype=np.float32), indices


def index_figure(index, title):
    fig, ax = plt.subplots(figsize=(4.2, 3.6))
    im = ax.imshow(index, cmap="RdYlGn", vmin=-1, vmax=1)
    ax.set_title(title)
    ax.set_axis_off()
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    return fig


@st.cache_data(ttl=86400, show_spinner=False)
def load_eurosat_samples(limit=24, seed=7):
    from datasets import load_dataset

    ds = load_dataset("blanchon/EuroSAT_MSI", split="train", streaming=True)
    try:
        class_names = ds.features["label"].names
    except Exception:
        class_names = None
    # Shuffle so we see a mix of land-cover classes, not just the first class
    ds = ds.shuffle(seed=seed, buffer_size=1000)
    samples = []
    for sample in ds:
        samples.append({
            "image": np.asarray(sample["image"]),
            "label": sample.get("label"),
        })
        if len(samples) >= limit:
            break
    if not samples:
        raise RuntimeError("The dataset stream returned no samples.")
    return samples, class_names


def heuristic_hint(ndvi, ndwi, ndbi):
    if ndwi > 0:
        return "Water-like"
    if ndvi > 0.6:
        return "Dense vegetation"
    if ndvi > 0.4:
        return "Vegetation / crops"
    if ndbi > -0.2:
        return "Built-up or bare"
    return "Mixed / transitional"


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.markdown("## 🌍 ORBITAL")
    st.caption("EARTH INTELLIGENCE / RESEARCH")
    st.divider()

    st.markdown("### Imagery settings")
    latest_date = date.today() - timedelta(days=1)
    selected_date = st.date_input(
        "Imagery date", value=latest_date, min_value=date(2002, 1, 1),
        max_value=latest_date, help="NASA GIBS may lag by a day or more.",
    )
    selected_layer = st.selectbox("Satellite layer", list(NASA_LAYERS))
    selected_region = st.selectbox("Region", list(REGIONS))

    st.divider()
    st.markdown("### Research artifact")
    if artifact_path and not artifact_error:
        st.success(f"Loaded `{Path(artifact_path).name}`")
    else:
        st.error(artifact_error or "No artifact")
    c1, c2 = st.columns(2)
    c1.metric("Q states", len(q_table))
    c2.metric("SARSA states", len(sarsa_table))

    st.divider()
    st.markdown(f"[NASA Worldview ↗]({NASA_WORLDVIEW})")
    st.markdown(f"[EuroSAT MSI ↗]({EUROSAT_URL})")
    st.caption("Research prototype - not operational satellite guidance.")

# ============================================================
# HERO
# ============================================================

st.markdown(
    """
    <div class="orbital-hero">
        <div class="orbital-eyebrow">Earth observation / Reinforcement learning / Spectral intelligence</div>
        <div class="orbital-title">ORBITAL</div>
        <div class="orbital-subtitle">
            Reinforcement learning for information-efficient Earth observation.
            A satellite has a limited observation budget and clouds get in the way:
            which spectrally distinct targets should it look at? Explore real imagery,
            multispectral signatures, and watch trained agents plan a mission.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

tab_overview, tab_earth, tab_spectral, tab_targets, tab_mission, tab_bench, tab_methods = st.tabs([
    "Overview", "Earth from orbit", "Spectral lab", "Target explorer",
    "Mission simulator", "Policy benchmark", "Methods",
])

# ============================================================
# TAB: OVERVIEW
# ============================================================

with tab_overview:
    st.subheader("The problem")
    st.write(
        "Satellites cannot observe everything. Each pass costs energy, clouds can ruin it, "
        "and looking at the same kind of surface twice wastes budget. ORBITAL clusters "
        "multispectral patches into **spectral targets**, then trains agents to pick "
        "a sequence of targets that maximises *new information* under a fixed budget."
    )

    o1, o2, o3, o4 = st.columns(4)
    o1.metric("Spectral targets", N_TARGETS or "n/a")
    o2.metric("Features / patch", len(FEATURE_NAMES))
    o3.metric("Observation budget", "6 passes")
    o4.metric("Weather regimes", 2)

    st.markdown("#### How the pipeline fits together")
    p1, p2, p3, p4 = st.columns(4)
    p1.markdown("**1 · Extract**  \n13 Sentinel-2 band means + NDVI / NDWI / NDBI statistics per patch")
    p2.markdown("**2 · Cluster**  \nMiniBatch k-means groups patches into 8 spectral targets")
    p3.markdown("**3 · Simulate**  \nBudget, energy cost, target-specific cloud risk, weather regime")
    p4.markdown("**4 · Learn**  \nTabular Q-learning and SARSA compared against greedy and random")

    if RL_READY:
        st.markdown("#### Headline result (live, 1,000 episodes)")
        with st.spinner("Running quick benchmark..."):
            quick, _ = run_benchmark(
                distance_matrix, {"q_learning": q_table, "sarsa": sarsa_table},
                tuple(POLICIES), 1000, 2026, 6, f"{len(q_table)}-{len(sarsa_table)}",
            )
        best = quick.sort_values("Mean reward", ascending=False).iloc[0]
        cols = st.columns(len(quick))
        for col, (_, r) in zip(cols, quick.iterrows()):
            col.metric(r["Policy"], f"{r['Mean reward']:.2f}", f"±{r['95% CI']:.2f} (95% CI)", delta_color="off")
        callout(
            f"Highest mean reward: <b>{best['Policy']}</b>. Open the <i>Policy benchmark</i> tab "
            "to check whether differences are larger than the confidence intervals - "
            "in a small tabular setting, learned and greedy policies can be statistically close."
        )
    else:
        callout("RL artifacts were not found, so live simulation tabs are disabled.", warn=True)

# ============================================================
# TAB: EARTH FROM ORBIT
# ============================================================

with tab_earth:
    st.subheader("Earth from orbit")
    st.caption("NASA GIBS browse imagery. This is a map image for a chosen day, not a live video feed.")

    fetch_clicked = st.button("↻ Load / refresh imagery", type="primary")
    if fetch_clicked:
        fetch_nasa_image.clear()

    if fetch_clicked or "nasa_result" not in st.session_state:
        with st.spinner("Requesting imagery from NASA GIBS..."):
            try:
                st.session_state["nasa_result"] = fetch_nasa_image(
                    selected_layer, selected_date.isoformat(), selected_region
                )
                st.session_state["nasa_error"] = None
            except Exception as exc:
                st.session_state["nasa_result"] = None
                st.session_state["nasa_error"] = str(exc)

    result = st.session_state.get("nasa_result")
    if result:
        img, layer, day, fallback = result
        show_image(img, f"NASA GIBS · {layer} · {day} · {selected_region}")
        if fallback:
            st.info(f"Requested layer/date unavailable; showing {layer} for {day}.")
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        st.download_button("Download image", buf.getvalue(), f"orbital_{day}.jpg", "image/jpeg")
    else:
        st.warning("NASA imagery could not be loaded. The other tabs still work.")
        with st.expander("Technical details"):
            st.code(st.session_state.get("nasa_error", "No image loaded."))

    st.link_button("Open interactive NASA Worldview ↗", NASA_WORLDVIEW)

# ============================================================
# TAB: SPECTRAL LAB
# ============================================================

with tab_spectral:
    st.subheader("Spectral lab")
    st.write(
        "Explore Sentinel-2 patches from EuroSAT MSI. Different surfaces reflect "
        "different wavelengths - that is the signal the RL agent reasons over."
    )

    sc1, sc2 = st.columns([1, 1])
    n_samples = sc1.slider("Patches to load", 8, 40, 24, step=4)
    shuffle_seed = sc2.number_input("Shuffle seed", 0, 9999, 7)

    try:
        with st.spinner("Loading EuroSAT MSI samples (first load can take a minute)..."):
            samples, class_names = load_eurosat_samples(int(n_samples), int(shuffle_seed))
    except Exception as exc:
        samples, class_names = None, None
        st.error("The multispectral dataset could not be loaded.")
        with st.expander("Error details"):
            st.code(f"{type(exc).__name__}: {exc}")
        st.link_button("Open dataset page ↗", EUROSAT_URL)

    if samples:
        idx = st.selectbox("Patch", range(len(samples)), format_func=lambda i: f"Patch {i + 1}")
        sample = samples[idx]
        try:
            cube = image_hwc(sample["image"])
            features, indices = calculate_features(cube)

            label = sample["label"]
            if class_names and isinstance(label, (int, np.integer)):
                label_text = class_names[int(label)]
            elif isinstance(label, (int, np.integer)) and int(label) < len(EUROSAT_CLASSES):
                label_text = f"{int(label)} (likely {EUROSAT_CLASSES[int(label)]})"
            else:
                label_text = str(label)

            m = st.columns(5)
            m[0].metric("Dataset label", label_text)
            m[1].metric("Mean NDVI", f"{indices['NDVI'].mean():.3f}")
            m[2].metric("Mean NDWI", f"{indices['NDWI'].mean():.3f}")
            m[3].metric("Mean NDBI", f"{indices['NDBI'].mean():.3f}")

            if scaler is not None and cluster_model is not None:
                try:
                    scaled = np.nan_to_num(scaler.transform(features.reshape(1, -1)))
                    target_id = int(cluster_model.predict(scaled)[0])
                    m[4].metric("Assigned target", f"T{target_id}")
                except Exception:
                    pass

            st.markdown("#### Composites")
            c1, c2, c3 = st.columns(3)
            with c1:
                show_image(make_composite(cube, [3, 2, 1]), "True-colour approximation (R, G, B)")
            with c2:
                show_image(make_composite(cube, [7, 3, 2]), "False colour (NIR, R, G) - vegetation glows red")
            with c3:
                show_image(make_composite(cube, [11, 7, 3]), "SWIR / NIR / R - separates bare, built, wet")

            st.markdown("#### Spectral indices")
            i1, i2, i3 = st.columns(3)
            for col, name in zip([i1, i2, i3], ["NDVI", "NDWI", "NDBI"]):
                with col:
                    fig = index_figure(indices[name], name)
                    st.pyplot(fig)
                    plt.close(fig)

            st.markdown("#### Spectral signature")
            fig, ax = plt.subplots(figsize=(10, 3.4))
            ax.plot(range(13), features[:13], marker="o", color=ACCENT, linewidth=2)
            ax.set_xticks(range(13))
            ax.set_xticklabels(FEATURE_NAMES[:13], rotation=0)
            ax.set_ylabel("Mean value (digital number)")
            ax.set_title("Mean spectrum across the 13 Sentinel-2 bands")
            ax.grid(alpha=0.3)
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

            table = pd.DataFrame({"Feature": FEATURE_NAMES[: len(features)], "Value": features})
            with st.expander("All 19 extracted features"):
                show_df(table)
            st.download_button(
                "Download features (CSV)", table.to_csv(index=False).encode(),
                f"orbital_features_patch_{idx + 1}.csv", "text/csv",
            )
        except Exception as exc:
            st.error("This patch could not be processed.")
            with st.expander("Details"):
                st.code(f"{type(exc).__name__}: {exc}")

# ============================================================
# TAB: TARGET EXPLORER
# ============================================================

with tab_targets:
    st.subheader("Spectral target explorer")
    st.write(
        "The eight targets are unsupervised k-means clusters. Interpretation labels below "
        "are simple threshold hints on the cluster-mean indices, not verified land cover."
    )

    if cluster_model is not None and hasattr(cluster_model, "cluster_centers_"):
        centers = np.asarray(cluster_model.cluster_centers_)
        raw_centers = None
        if scaler is not None:
            try:
                raw_centers = scaler.inverse_transform(centers)
            except Exception:
                raw_centers = None

        if raw_centers is not None and raw_centers.shape[1] >= 19:
            profile = pd.DataFrame({
                "Target": [f"T{i}" for i in range(len(centers))],
                "NDVI": raw_centers[:, 13],
                "NDWI": raw_centers[:, 15],
                "NDBI": raw_centers[:, 17],
            })
            profile["Hint"] = [
                heuristic_hint(r.NDVI, r.NDWI, r.NDBI) for r in profile.itertuples()
            ]
            if distance_matrix is not None:
                profile["Mean distance to others"] = [
                    distance_matrix[i][np.arange(N_TARGETS) != i].mean()
                    for i in range(len(centers))
                ]
            st.markdown("#### Target profiles (in original index units)")
            show_df(profile.round(3))

            fig, ax = plt.subplots(figsize=(10, 3.8))
            x = np.arange(len(profile))
            w = 0.26
            ax.bar(x - w, profile["NDVI"], w, label="NDVI", color="#6fd08c")
            ax.bar(x, profile["NDWI"], w, label="NDWI", color="#6fb7ff")
            ax.bar(x + w, profile["NDBI"], w, label="NDBI", color="#f2b66d")
            ax.axhline(0, color=TEXT, linewidth=0.7)
            ax.set_xticks(x)
            ax.set_xticklabels(profile["Target"])
            ax.set_title("Spectral indices by target")
            ax.legend(ncol=3)
            ax.grid(axis="y", alpha=0.3)
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

        left, right = st.columns(2)
        with left:
            st.markdown("#### Target map (PCA of cluster centres)")
            try:
                from sklearn.decomposition import PCA

                coords = PCA(n_components=2, random_state=0).fit_transform(centers)
                fig, ax = plt.subplots(figsize=(5.4, 4.4))
                ax.scatter(coords[:, 0], coords[:, 1], s=260, color=ACCENT, alpha=0.85, edgecolor=BG)
                for i, (px, py) in enumerate(coords):
                    ax.text(px, py, f"T{i}", ha="center", va="center", color=BG, fontweight="bold")
                ax.set_xlabel("PC 1")
                ax.set_ylabel("PC 2")
                ax.grid(alpha=0.3)
                fig.tight_layout()
                st.pyplot(fig)
                plt.close(fig)
            except Exception as exc:
                st.caption(f"PCA unavailable: {exc}")
        with right:
            st.markdown("#### Dissimilarity matrix")
            if distance_matrix is not None:
                fig, ax = plt.subplots(figsize=(5.4, 4.4))
                im = ax.imshow(distance_matrix, cmap="viridis", vmin=0, vmax=1)
                ax.set_xticks(range(N_TARGETS))
                ax.set_yticks(range(N_TARGETS))
                ax.set_xticklabels([f"T{i}" for i in range(N_TARGETS)])
                ax.set_yticklabels([f"T{i}" for i in range(N_TARGETS)])
                for i in range(N_TARGETS):
                    for j in range(N_TARGETS):
                        ax.text(j, i, f"{distance_matrix[i, j]:.2f}", ha="center", va="center",
                                fontsize=7, color="white" if distance_matrix[i, j] < 0.5 else "black")
                fig.colorbar(im, ax=ax, fraction=0.046, label="Normalised distance")
                fig.tight_layout()
                st.pyplot(fig)
                plt.close(fig)
                st.caption("Novelty reward = distance from the nearest already-observed target.")
    else:
        st.info("No compatible cluster model found in the artifact (`region_model`).")

# ============================================================
# TAB: MISSION SIMULATOR
# ============================================================

with tab_mission:
    st.subheader("Mission simulator")
    if not RL_READY:
        st.info("Needs the saved Q-table and distance matrix.")
    else:
        st.write(
            "Fly one simulated mission step by step. The chart shows what the agent believes "
            "each target is worth right now; the log shows what actually happened."
        )
        tables = {"q_learning": q_table, "sarsa": sarsa_table}

        mc1, mc2, mc3, mc4 = st.columns([1.4, 1, 1, 1])
        policy = mc1.selectbox(
            "Agent", POLICIES, format_func=lambda k: POLICY_LABELS[k],
            index=POLICIES.index("q_learning") if "q_learning" in POLICIES else 0,
        )
        seed = mc2.number_input("Mission seed", 0, 99999, 42)
        new_clicked = mc3.button("⟲ New mission", type="primary")
        mc4.write("")

        if new_clicked or "mission" not in st.session_state or st.session_state.get("mission_seed") != seed:
            env = OrbitalEnvV2(distance_matrix, seed=int(seed))
            st.session_state["mission"] = {
                "env": env,
                "state": env.reset(),
                "rng": np.random.default_rng(int(seed) + 1),
                "log": [],
                "done": False,
                "total": 0.0,
            }
            st.session_state["mission_seed"] = seed

        mission = st.session_state["mission"]
        env = mission["env"]

        b1, b2, _ = st.columns([1, 1, 3])
        step_clicked = b1.button("▶ Next pass", disabled=mission["done"])
        run_clicked = b2.button("⏭ Run to end", disabled=mission["done"])

        def do_step():
            state = mission["state"]
            qv = q_values(tables[policy], state, N_TARGETS) if policy in tables else None
            action = choose_action(policy, state, mission["rng"], N_TARGETS, distance_matrix, tables)
            next_state, reward, done, info = env.step(action)
            mission["log"].append({
                "Pass": len(mission["log"]) + 1,
                "Target": f"T{action}",
                "Weather": "Cloudy" if info["weather"] else "Clear",
                "Cloud prob.": round(info["cloud_probability"], 2),
                "Outcome": info["outcome"],
                "Novelty": round(info["novelty"], 2),
                "Reward": round(reward, 3),
                "_qv": qv, "_action": action,
            })
            mission["state"] = next_state
            mission["total"] += reward
            mission["done"] = done

        if step_clicked and not mission["done"]:
            do_step()
        if run_clicked:
            while not mission["done"]:
                do_step()

        s = st.columns(4)
        s[0].metric("Passes left", env.budget)
        s[1].metric("Targets captured", f"{len(env.visited)} / {N_TARGETS}")
        s[2].metric("Total reward", f"{mission['total']:.2f}")
        s[3].metric("Weather now", "Cloudy" if env.weather else "Clear")

        left, right = st.columns([1.1, 1])
        with left:
            st.markdown("#### Agent's value estimate for the next pass")
            current_qv = (
                q_values(tables[policy], mission["state"], N_TARGETS)
                if policy in tables else None
            )
            if current_qv is None:
                st.caption("Greedy and random agents do not use a value table.")
            else:
                known = mission["state"] in tables[policy]
                fig, ax = plt.subplots(figsize=(6, 3.6))
                colors = [
                    "#4a5d6e" if i in env.visited else COLORS.get(policy, ACCENT)
                    for i in range(N_TARGETS)
                ]
                ax.bar([f"T{i}" for i in range(N_TARGETS)], current_qv, color=colors)
                ax.axhline(0, color=TEXT, linewidth=0.6)
                ax.set_ylabel("Q-value")
                ax.grid(axis="y", alpha=0.3)
                fig.tight_layout()
                st.pyplot(fig)
                plt.close(fig)
                st.caption(
                    "Grey bars are already captured. "
                    + ("" if known else "This exact state was never visited in training, so values are all zero and the agent picks at random.")
                )
        with right:
            st.markdown("#### Captured targets")
            fig, ax = plt.subplots(figsize=(5, 3.6))
            ax.set_xlim(-0.5, N_TARGETS - 0.5)
            ax.set_ylim(-0.6, 0.6)
            for i in range(N_TARGETS):
                got = i in env.visited
                ax.scatter(i, 0, s=520, color=ACCENT if got else "#1b3247",
                           edgecolor=ACCENT, linewidth=1.4)
                ax.text(i, 0, f"T{i}", ha="center", va="center",
                        color=BG if got else TEXT, fontsize=9, fontweight="bold")
            ax.set_axis_off()
            st.pyplot(fig)
            plt.close(fig)
            st.caption("Filled = successfully observed (cloud-free, new).")

        if mission["log"]:
            st.markdown("#### Mission log")
            log_df = pd.DataFrame([{k: v for k, v in r.items() if not k.startswith("_")} for r in mission["log"]])
            show_df(log_df)
        if mission["done"]:
            callout(
                f"Mission complete: {len(env.visited)} unique targets, total reward {mission['total']:.2f}. "
                "A single mission is noisy - use the benchmark tab for statistics."
            )

# ============================================================
# TAB: POLICY BENCHMARK
# ============================================================

with tab_bench:
    st.subheader("Policy benchmark")
    if not RL_READY:
        st.info("Needs the saved Q-table and distance matrix.")
    else:
        st.write(
            "Every policy runs on the same simulated environment seed, so differences come "
            "from the policy rather than from luck of the draw."
        )
        bc1, bc2, bc3 = st.columns(3)
        n_eps = bc1.select_slider("Episodes per policy", [200, 500, 1000, 2000, 5000], value=2000)
        bench_seed = bc2.number_input("Evaluation seed", 0, 99999, 2026, key="bench_seed")
        budget = bc3.slider("Observation budget", 3, 10, 6)

        if budget != 6:
            callout(
                "The agents were trained with a budget of 6. At other budgets the state space "
                "differs, so learned policies fall back to random for unseen states. "
                "That is a useful generalisation test - not a bug.", warn=True,
            )

        tables = {"q_learning": q_table, "sarsa": sarsa_table}
        with st.spinner("Simulating..."):
            df, raw = run_benchmark(
                distance_matrix, tables, tuple(POLICIES), int(n_eps), int(bench_seed),
                int(budget), f"{len(q_table)}-{len(sarsa_table)}",
            )

        fig, axes = plt.subplots(1, 3, figsize=(14, 4))
        names = df["Policy"].tolist()
        cols = [COLORS[k] for k in df["key"]]

        axes[0].bar(names, df["Mean reward"], yerr=df["95% CI"], color=cols, capsize=5)
        axes[0].set_title("Mean reward (±95% CI)")
        axes[1].bar(names, df["Unique targets"], color=cols)
        axes[1].set_title("Unique targets captured")
        axes[2].bar(names, df["Spectral diversity"], color=cols)
        axes[2].set_title("Pairwise spectral diversity")
        for ax in axes:
            ax.grid(axis="y", alpha=0.3)
            ax.tick_params(axis="x", rotation=15)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        display = df.drop(columns=["key"]).round(3)
        show_df(display)

        # Honest statistical read-out
        ranked = df.sort_values("Mean reward", ascending=False).reset_index(drop=True)
        top, second = ranked.iloc[0], ranked.iloc[1]
        gap = top["Mean reward"] - second["Mean reward"]
        margin = top["95% CI"] + second["95% CI"]
        if gap > margin:
            callout(
                f"<b>{top['Policy']}</b> leads <b>{second['Policy']}</b> by {gap:.3f} reward, "
                "larger than the combined confidence margin."
            )
        else:
            callout(
                f"<b>{top['Policy']}</b> and <b>{second['Policy']}</b> differ by only {gap:.3f} reward "
                f"(combined 95% margin {margin:.3f}), so this benchmark cannot separate them. "
                "Compare their coverage and diversity columns for the trade-off.", warn=True,
            )

        st.markdown("#### Reward distributions")
        fig, ax = plt.subplots(figsize=(10, 3.8))
        for key in POLICIES:
            ax.hist(raw[key], bins=40, alpha=0.5, label=POLICY_LABELS[key],
                    color=COLORS[key], density=True)
        ax.set_xlabel("Episode reward")
        ax.set_ylabel("Density")
        ax.legend()
        ax.grid(alpha=0.3)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        for label, series in [("Q-learning", q_train_rewards), ("SARSA", sarsa_train_rewards)]:
            if series is not None and len(series) > 400:
                fig, ax = plt.subplots(figsize=(10, 3))
                ma = np.convolve(series, np.ones(200) / 200, mode="valid")
                ax.plot(ma, color=COLORS["q_learning" if label == "Q-learning" else "sarsa"])
                ax.set_title(f"{label} training curve (200-episode moving average)")
                ax.set_xlabel("Training episode")
                ax.grid(alpha=0.3)
                fig.tight_layout()
                st.pyplot(fig)
                plt.close(fig)
        if q_train_rewards is None:
            st.caption(
                "Tip: add `q_learning_rewards` and `sarsa_rewards` to the saved artifact "
                "to show training curves here."
            )

# ============================================================
# TAB: METHODS
# ============================================================

with tab_methods:
    st.subheader("Methods and provenance")

    with st.expander("Reward function", expanded=True):
        st.markdown(
            """
            Per pass: `reward = -observation_cost - energy_cost[target]`, then:

            - **Cloud-blocked** (probability depends on target and weather): `-0.25`
            - **Redundant** (already captured): `-0.50`
            - **New target**: `+ novelty`, where novelty is the distance to the nearest
              already-captured target in standardised spectral feature space (first target = 1.0)

            Cloud risk rises with target index (0.10 → 0.40) and by +0.25 in the cloudy weather
            regime, which flips with probability 0.25 per pass.
            """
        )
    with st.expander("Spectral features"):
        st.markdown(
            """
            Per patch: mean of each of the 13 Sentinel-2 bands plus mean and standard deviation of
            NDVI (vegetation), NDWI (green vs NIR), and NDBI (SWIR vs NIR). Features are standardised
            before k-means. Band order is assumed to be B1-B12 including B8A.
            """
        )
    with st.expander("Limitations"):
        st.markdown(
            """
            - Clusters are unsupervised spectral groups, not verified land cover.
            - The environment is a simulation: cloud and energy models are synthetic.
            - Tabular RL over (last target, visited set, budget, weather) does not generalise to new
              budgets or target counts.
            - Results come from 600 EuroSAT patches, a European dataset.
            - Nothing here controls or advises a real satellite.
            """
        )
    with st.expander("Data sources"):
        st.markdown(
            f"- NASA GIBS imagery via WMS ([Worldview]({NASA_WORLDVIEW}))\n"
            f"- [EuroSAT MSI]({EUROSAT_URL}) multispectral patches"
        )

st.divider()
st.caption("ORBITAL · Research prototype · Not affiliated with NASA. External data belongs to its providers.")
