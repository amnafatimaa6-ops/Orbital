
import os
import pickle
import time
import warnings

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# =========================================================
# ORBITAL | Reinforcement Learning for Earth Observation
# =========================================================

st.set_page_config(
    page_title="ORBITAL | Earth Intelligence",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ------------------------- DESIGN -------------------------

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Manrope:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Manrope', sans-serif;
}
.stApp {
    background:
        radial-gradient(ellipse at 85% 0%, rgba(32, 92, 122, .18), transparent 36%),
        #08111c;
    color: #e7eef7;
}
[data-testid="stSidebar"] {
    background: #0d1927;
    border-right: 1px solid #223345;
}
[data-testid="stHeader"] {
    background: rgba(8,17,28,.9);
}
.block-container {
    max-width: 1450px;
    padding-top: 2rem;
    padding-bottom: 4rem;
}
h1, h2, h3 {
    letter-spacing: -0.045em;
}
h1 { font-weight: 800; }
h2 { font-weight: 750; }
h3 { font-weight: 700; }
.eyebrow {
    color: #65d8c5;
    font-family: 'DM Mono', monospace;
    font-size: 11px;
    letter-spacing: .16em;
    text-transform: uppercase;
}
.hero {
    background: linear-gradient(120deg, #11283a 0%, #102031 52%, #12333c 100%);
    border: 1px solid #2b4b5b;
    border-radius: 22px;
    padding: 30px 32px;
    margin: 10px 0 22px 0;
}
.hero-title {
    font-size: clamp(32px, 4vw, 50px);
    line-height: 1.05;
    font-weight: 800;
    letter-spacing: -.055em;
    margin: 8px 0 12px 0;
}
.hero-sub {
    color: #b7c9d8;
    font-size: 15px;
    line-height: 1.7;
    max-width: 760px;
}
.status-pill {
    display: inline-block;
    border: 1px solid #316c68;
    background: #123a3b;
    color: #7ce7d2;
    border-radius: 999px;
    padding: 6px 11px;
    font-family: 'DM Mono', monospace;
    font-size: 10px;
    letter-spacing: .06em;
}
.metric-card {
    background: linear-gradient(145deg, #122235, #0d1a28);
    border: 1px solid #26394d;
    border-radius: 16px;
    padding: 18px 19px;
    min-height: 116px;
}
.metric-label {
    color: #9db0c3;
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: .11em;
    margin-bottom: 11px;
}
.metric-value {
    color: #f0f6fc;
    font-size: 28px;
    font-weight: 800;
    line-height: 1.1;
}
.metric-foot {
    color: #65d8c5;
    font-size: 11px;
    margin-top: 9px;
}
.section-head {
    border-bottom: 1px solid #24384a;
    padding-bottom: 11px;
    margin: 34px 0 18px 0;
}
.panel {
    background: #0d1a28;
    border: 1px solid #24384a;
    border-radius: 16px;
    padding: 18px;
}
.small-muted {
    color: #9db0c3;
    font-size: 12px;
}
.mono {
    font-family: 'DM Mono', monospace;
}
div[data-testid="stMetric"] {
    background: #0d1a28;
    border: 1px solid #26394d;
    padding: 15px 17px;
    border-radius: 14px;
}
div[data-testid="stMetricLabel"] {
    color: #a7b8c9;
}
div[data-testid="stMetricValue"] {
    color: #f1f6fb;
}
.stButton button {
    border-radius: 10px;
    min-height: 42px;
    font-weight: 700;
}
hr {
    border-color: #24384a;
}
</style>
""", unsafe_allow_html=True)


# ------------------------- HELPERS -------------------------

def metric_card(label, value, foot=""):
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-foot">{foot}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def heading(number, title, subtitle):
    st.markdown(
        f"""
        <div class="section-head">
            <div class="eyebrow">{number} / RESEARCH MODULE</div>
            <h2 style="margin:5px 0 4px 0">{title}</h2>
            <div class="small-muted">{subtitle}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def normalize_band(band):
    band = np.asarray(band, dtype=np.float32)
    lo, hi = np.nanpercentile(band, [2, 98])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return np.zeros_like(band)
    return np.clip((band - lo) / (hi - lo), 0, 1)


def to_hwc(image):
    arr = np.asarray(image)

    if arr.ndim == 2:
        arr = np.repeat(arr[..., None], 3, axis=2)

    if arr.ndim == 3 and arr.shape[0] in (3, 4, 13) and arr.shape[-1] not in (3, 4, 13):
        arr = np.moveaxis(arr, 0, -1)

    if arr.ndim != 3:
        raise ValueError("Expected a 2D or 3D satellite image.")

    return arr.astype(np.float32)


def spectral_features(image):
    """Extract 13 band means plus mean/std for three spectral indices."""
    arr = to_hwc(image)

    if arr.shape[-1] < 12:
        raise ValueError(
            f"Expected at least 12 spectral bands, got {arr.shape[-1]}."
        )

    # Match the notebook's band conventions.
    green = arr[:, :, 2]
    red = arr[:, :, 3]
    nir = arr[:, :, 7]
    swir = arr[:, :, 11]
    eps = 1e-8

    ndvi = (nir - red) / (nir + red + eps)
    ndwi = (green - nir) / (green + nir + eps)
    ndbi = (swir - nir) / (swir + nir + eps)

    band_means = np.nanmean(arr, axis=(0, 1)).tolist()
    extra = [
        np.nanmean(ndvi), np.nanstd(ndvi),
        np.nanmean(ndwi), np.nanstd(ndwi),
        np.nanmean(ndbi), np.nanstd(ndbi),
    ]

    return np.asarray(band_means[:13] + extra, dtype=np.float64)


def index_maps(image):
    arr = to_hwc(image)
    eps = 1e-8
    green, red, nir, swir = arr[:, :, 2], arr[:, :, 3], arr[:, :, 7], arr[:, :, 11]

    return {
        "NDVI · vegetation": (nir - red) / (nir + red + eps),
        "NDWI · water signal": (green - nir) / (green + nir + eps),
        "NDBI · built-up signal": (swir - nir) / (swir + nir + eps),
    }


def composite(image, kind="Natural colour"):
    arr = to_hwc(image)

    if kind == "Natural colour":
        # Natural-colour-like composite using visible bands.
        indices = [2, 1, 0]
    elif kind == "False colour":
        indices = [7, 3, 2]
    else:
        indices = [11, 7, 3]

    if max(indices) >= arr.shape[-1]:
        indices = [0, 1, 2]

    channels = [normalize_band(arr[:, :, i]) for i in indices]
    return np.stack(channels, axis=-1)


# ------------------------- MODEL LOADING -------------------------

@st.cache_resource(show_spinner=False)
def load_artifacts():
    path = "orbital_models.pkl"
    if not os.path.exists(path):
        return None, f"Model file not found: {path}"

    try:
        with open(path, "rb") as f:
            bundle = pickle.load(f)

        if not isinstance(bundle, dict):
            return None, "The pickle file does not contain a model dictionary."

        return bundle, None
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"


def first_existing(mapping, keys):
    for key in keys:
        if key in mapping:
            return mapping[key]
    return None


# ------------------------- DATA LOADING -------------------------

@st.cache_data(show_spinner=False, ttl=3600)
def load_satellite_samples(limit=24):
    """
    Load a small sample of EuroSAT MSI images.
    Cached so repeated Streamlit reruns don't download them again.
    """
    from datasets import load_dataset

    ds = load_dataset(
        "blanchon/EuroSAT_MSI",
        split="train",
        streaming=True,
    )

    samples = []
    for i, row in enumerate(ds):
        image = row.get("image")
        if image is None:
            continue

        samples.append({
            "sample_id": i,
            "image": np.asarray(image),
            "label": str(row.get("label", "Unknown")),
            "filename": str(row.get("filename", f"sample_{i}")),
        })

        if len(samples) >= limit:
            break

    if not samples:
        raise RuntimeError("No images were returned by the dataset.")

    return samples


# ------------------------- ENVIRONMENT -------------------------

N_TARGETS = 8
BUDGET = 6


def choose_q_action(q_table, state, valid_actions, rng):
    """
    Try common Q-table state formats. If the reconstructed state isn't
    present, use a transparent novelty-based fallback rather than crashing.
    """
    if q_table:
        candidate_keys = [
            state,
            str(state),
            tuple(state) if isinstance(state, list) else None,
        ]

        for key in candidate_keys:
            if key is None:
                continue
            try:
                if key in q_table:
                    values = np.asarray(q_table[key], dtype=float).ravel()
                    if len(values) >= N_TARGETS:
                        masked = values[:N_TARGETS].copy()
                        for action in range(N_TARGETS):
                            if action not in valid_actions:
                                masked[action] = -np.inf
                        if np.isfinite(masked).any():
                            return int(np.argmax(masked)), False
            except Exception:
                pass

    return None, True


def simulate_mission(policy_name, seed, q_table, distance_matrix, centers):
    """
    Lightweight demonstration simulator for the dashboard.
    This is not a physical orbital dynamics model.
    """
    rng = np.random.default_rng(seed)
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.shape != (N_TARGETS, N_TARGETS):
        distances = np.ones((N_TARGETS, N_TARGETS)) - np.eye(N_TARGETS)

    visited = []
    actions_log = []
    total_reward = 0.0
    budget_left = BUDGET
    fallback_count = 0
    weather = 0

    for step in range(BUDGET):
        weather = int(rng.random() < 0.25) if step else 0
        available = list(range(N_TARGETS))

        if policy_name == "Random":
            action = int(rng.choice(available))

        elif policy_name == "Greedy novelty":
            if not visited:
                action = int(rng.integers(N_TARGETS))
            else:
                novelty = np.min(distances[:, visited], axis=1)
                novelty[visited] = -np.inf
                action = int(np.argmax(novelty))

        else:
            # Use a stored Q table where the reconstructed state matches.
            state = (budget_left, weather, tuple(sorted(set(visited))))
            action, fallback = choose_q_action(
                q_table, state, available, rng
            )
            fallback_count += int(fallback)

            if action is None:
                # Explicit fallback for state formats that don't match.
                if not visited:
                    action = int(rng.integers(N_TARGETS))
                else:
                    novelty = np.min(distances[:, visited], axis=1)
                    novelty[visited] = -np.inf
                    action = int(np.argmax(novelty))

        cloud_risk = 0.10 + 0.30 * (action / max(N_TARGETS - 1, 1))
        cloud_risk = min(0.85, cloud_risk + 0.12 * weather)
        energy_cost = 0.05 + 0.15 * (action / max(N_TARGETS - 1, 1))

        reward = 0.0
        if rng.random() < cloud_risk:
            reward -= 0.25
            outcome = "Cloud-obstructed"
        else:
            outcome = "Observation acquired"
            if action not in visited:
                reward += 1.0 if not visited else float(
                    np.min(distances[action, visited])
                )
            else:
                reward -= 0.50

        reward -= energy_cost * 0.20
        visited.append(action)
        total_reward += reward
        budget_left -= 1

        actions_log.append({
            "Step": step + 1,
            "Target": action,
            "Weather regime": "Variable" if weather else "Stable",
            "Outcome": outcome,
            "Energy cost": round(energy_cost, 3),
            "Reward": round(reward, 3),
            "Budget remaining": budget_left,
        })

    unique_targets = len(set(visited))
    diversity = unique_targets / N_TARGETS

    return {
        "policy": policy_name,
        "reward": float(total_reward),
        "unique_targets": unique_targets,
        "diversity": float(diversity),
        "actions": visited,
        "log": pd.DataFrame(actions_log),
        "fallbacks": fallback_count,
    }


def evaluate_policies(episodes, seed, q_learning, sarsa, distance_matrix, centers):
    rows = []
    policies = [
        ("Random", None),
        ("Greedy novelty", None),
        ("Q-learning", q_learning),
        ("SARSA", sarsa),
    ]

    for name, q_table in policies:
        rewards, unique, diversity = [], [], []

        for ep in range(episodes):
            result = simulate_mission(
                name,
                seed + ep,
                q_table,
                distance_matrix,
                centers,
            )
            rewards.append(result["reward"])
            unique.append(result["unique_targets"])
            diversity.append(result["diversity"])

        rows.append({
            "Policy": name,
            "Mean reward": np.mean(rewards),
            "Reward SD": np.std(rewards),
            "Mean unique targets": np.mean(unique),
            "Mean diversity": np.mean(diversity),
        })

    return pd.DataFrame(rows)


# ------------------------- LOAD MODELS -------------------------

bundle, load_error = load_artifacts()

if bundle is not None:
    q_learning = first_existing(
        bundle, ["q_learning", "Q_v2", "Q", "q_table"]
    )
    sarsa = first_existing(
        bundle, ["sarsa", "Q_sarsa", "SARSA", "sarsa_table"]
    )
    scaler = bundle.get("scaler")
    region_model = bundle.get("region_model")
    target_features = bundle.get("target_features")
    distance_matrix = bundle.get("distance_matrix")

    if distance_matrix is None:
        distance_matrix = np.ones((N_TARGETS, N_TARGETS)) - np.eye(N_TARGETS)

    if target_features is None:
        target_features = np.zeros((N_TARGETS, 19))

    centers = np.asarray(target_features)
else:
    q_learning, sarsa, scaler, region_model = None, None, None, None
    centers = np.zeros((N_TARGETS, 19))
    distance_matrix = np.ones((N_TARGETS, N_TARGETS)) - np.eye(N_TARGETS)


# ------------------------- SIDEBAR -------------------------

with st.sidebar:
    st.markdown('<div class="eyebrow">ORBITAL / CONTROL ROOM</div>', unsafe_allow_html=True)
    st.title("Mission setup")
    st.caption("Configure a reproducible observation run.")

    policy_name = st.selectbox(
        "Mission policy",
        ["Q-learning", "SARSA", "Greedy novelty", "Random"],
    )
    mission_seed = st.number_input(
        "Random seed", min_value=0, max_value=999999, value=42, step=1
    )
    st.divider()

    st.markdown("**Evaluation settings**")
    episodes = st.select_slider(
        "Episodes per policy",
        options=[25, 50, 100, 200],
        value=50,
    )

    run_mission = st.button(
        "▶  Run observation mission",
        type="primary",
        use_container_width=True,
    )
    run_comparison = st.button(
        "⇄  Benchmark all policies",
        use_container_width=True,
    )

    st.divider()
    st.markdown("**Artifact status**")
    if bundle is not None:
        st.success("Model artifact loaded")
        st.caption(f"Q-learning states: {len(q_learning) if q_learning else 0}")
        st.caption(f"SARSA states: {len(sarsa) if sarsa else 0}")
    else:
        st.error("Model artifact unavailable")
        st.caption(load_error)

    st.markdown("---")
    st.caption("Research prototype · Not operational satellite guidance")


# ------------------------- HERO -------------------------

st.markdown("""
<div class="hero">
    <div class="eyebrow">EARTH INTELLIGENCE · REINFORCEMENT LEARNING</div>
    <div class="hero-title">Observe smarter.<br>Discover more.</div>
    <div class="hero-sub">
        ORBITAL explores how a reinforcement-learning agent can allocate a
        limited observation budget across spectrally distinct Earth targets.
        Inspect multispectral signals, explore discovered target groups,
        and compare observation strategies in a controlled simulation.
    </div>
    <div style="margin-top:20px">
        <span class="status-pill">● RESEARCH PROTOTYPE</span>
    </div>
</div>
""", unsafe_allow_html=True)

c1, c2, c3, c4 = st.columns(4)
with c1:
    metric_card("Spectral targets", "08", "Unsupervised groups")
with c2:
    metric_card("Input features", "19", "Band statistics + indices")
with c3:
    metric_card("Mission budget", str(BUDGET), "Simulated observations")
with c4:
    metric_card("Learning methods", "02", "Q-learning + SARSA")


# ------------------------- SATELLITE INTELLIGENCE -------------------------

heading(
    "01",
    "Satellite intelligence",
    "Explore a multispectral sample and inspect its spectral signatures."
)

sample_records = []
dataset_error = None

with st.spinner("Loading multispectral samples…"):
    try:
        sample_records = load_satellite_samples(24)
    except Exception as exc:
        dataset_error = f"{type(exc).__name__}: {exc}"

if sample_records:
    sample_labels = [
        f"{s['sample_id']:03d} · {s['filename']}"
        for s in sample_records
    ]

    left, right = st.columns([1.2, 0.8], gap="large")

    with left:
        chosen_label = st.selectbox("Image sample", sample_labels)
        sample_idx = sample_labels.index(chosen_label)
        sample = sample_records[sample_idx]
        img = sample["image"]

        composite_mode = st.radio(
            "Display composite",
            ["Natural colour", "False colour", "SWIR / NIR / Red"],
            horizontal=True,
        )

        try:
            shown_image = composite(img, composite_mode)
            st.image(
                shown_image,
                caption=f"{sample['filename']} · {composite_mode}",
                use_container_width=True,
            )
        except Exception as exc:
            st.warning(f"Could not render this image: {exc}")

        st.caption(
            "Composite channels are visualizations of selected spectral bands, "
            "not necessarily a true-colour photograph."
        )

    with right:
        try:
            features = spectral_features(img)
            if scaler is not None and region_model is not None:
                scaled = scaler.transform(features.reshape(1, -1))
                target_id = int(region_model.predict(scaled)[0])
            else:
                target_id = -1

            st.markdown("#### Spectral interpretation")
            if target_id >= 0:
                st.markdown(
                    f'<div class="panel"><div class="eyebrow">ASSIGNED SPECTRAL GROUP</div>'
                    f'<div style="font-size:42px;font-weight:800;margin-top:8px">'
                    f'Target {target_id}</div>'
                    f'<div class="small-muted">Cluster assignment from the saved model</div></div>',
                    unsafe_allow_html=True,
                )
            else:
                st.info("Model unavailable; target assignment cannot be calculated.")

            maps = index_maps(img)
            cols = st.columns(3)
            for col, (name, values) in zip(cols, maps.items()):
                with col:
                    st.metric(name.split(" · ")[0], f"{np.nanmean(values):.3f}")

            with st.expander("Inspect spectral index maps"):
                map_cols = st.columns(3)
                for col, (name, values) in zip(map_cols, maps.items()):
                    with col:
                        fig, ax = plt.subplots(figsize=(4, 3))
                        im = ax.imshow(values, cmap="RdYlGn" if "NDVI" in name else "BrBG")
                        ax.set_title(name, fontsize=9)
                        ax.axis("off")
                        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                        st.pyplot(fig, use_container_width=True)
                        plt.close(fig)

            with st.expander("View the 19 extracted features"):
                feat_names = (
                    [f"Band {i+1} mean" for i in range(13)]
                    + ["NDVI mean", "NDVI std", "NDWI mean", "NDWI std",
                       "NDBI mean", "NDBI std"]
                )
                st.dataframe(
                    pd.DataFrame({"Feature": feat_names, "Value": features}),
                    use_container_width=True,
                    hide_index=True,
                )

        except Exception as exc:
            st.error(f"Feature extraction failed: {exc}")

else:
    st.warning(
        "The image dataset could not be loaded. Model and dashboard sections "
        "remain available. Check the app logs and dataset connectivity."
    )
    if dataset_error:
        with st.expander("Dataset loading details"):
            st.code(dataset_error)


# ------------------------- TARGET DISCOVERY -------------------------

heading(
    "02",
    "Spectral target discovery",
    "Inspect the learned feature profiles and pairwise separation between groups."
)

if bundle is not None and scaler is not None and centers.ndim == 2:
    left, right = st.columns([1, 1], gap="large")

    with left:
        target_choice = st.selectbox(
            "Select spectral target",
            list(range(min(N_TARGETS, len(centers)))),
            format_func=lambda x: f"Target {x}",
        )

        try:
            center = np.asarray(centers[target_choice]).reshape(1, -1)
            # The notebook stores standardized cluster centers.
            if center.shape[1] == getattr(scaler, "n_features_in_", center.shape[1]):
                center_original = scaler.inverse_transform(center)[0]
            else:
                center_original = center[0]

            fig, ax = plt.subplots(figsize=(8, 3.2))
            ax.plot(
                range(1, len(center_original[:13]) + 1),
                center_original[:13],
                marker="o",
                linewidth=2,
            )
            ax.set_xlabel("Spectral band")
            ax.set_ylabel("Mean reflectance / stored feature scale")
            ax.set_title(f"Target {target_choice} · band profile")
            ax.grid(alpha=0.2)
            fig.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)
        except Exception as exc:
            st.warning(f"Could not display centroid profile: {exc}")

    with right:
        dist = np.asarray(distance_matrix, dtype=float)
        fig, ax = plt.subplots(figsize=(6, 4))
        im = ax.imshow(dist, cmap="viridis")
        ax.set_title("Pairwise target distance")
        ax.set_xlabel("Target")
        ax.set_ylabel("Target")
        ax.set_xticks(range(dist.shape[1]))
        ax.set_yticks(range(dist.shape[0]))
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

        st.caption(
            "Distance values follow the notebook's saved representation. "
            "They describe feature-space separation, not geographic distance."
        )
else:
    st.info("Load the model artifact to inspect spectral target profiles.")


# ------------------------- MISSION CONTROL -------------------------

heading(
    "03",
    "Mission control",
    "Run a six-observation mission and inspect the agent's decisions."
)

if "mission_result" not in st.session_state:
    st.session_state.mission_result = None

if run_mission:
    table = (
        q_learning if policy_name == "Q-learning"
        else sarsa if policy_name == "SARSA"
        else None
    )

    with st.spinner("Simulating mission…"):
        st.session_state.mission_result = simulate_mission(
            policy_name,
            int(mission_seed),
            table,
            distance_matrix,
            centers,
        )

mission_result = st.session_state.mission_result

if mission_result is not None:
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Mission reward", f"{mission_result['reward']:.3f}")
    m2.metric("Unique targets", f"{mission_result['unique_targets']} / {N_TARGETS}")
    m3.metric("Target coverage", f"{mission_result['diversity']:.0%}")
    m4.metric("Fallback decisions", mission_result["fallbacks"])

    left, right = st.columns([1, 1], gap="large")

    with left:
        st.markdown("#### Observation sequence")
        seq = mission_result["actions"]
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.plot(range(1, len(seq) + 1), seq, marker="o", linewidth=2)
        ax.set_xticks(range(1, len(seq) + 1))
        ax.set_yticks(range(N_TARGETS))
        ax.set_xlabel("Observation step")
        ax.set_ylabel("Spectral target")
        ax.grid(alpha=0.25)
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    with right:
        st.markdown("#### Mission action log")
        st.dataframe(
            mission_result["log"],
            use_container_width=True,
            hide_index=True,
        )

    st.download_button(
        "Download mission log (CSV)",
        mission_result["log"].to_csv(index=False).encode("utf-8"),
        file_name="orbital_mission_log.csv",
        mime="text/csv",
    )
else:
    st.markdown(
        """
        <div class="panel">
            <div class="eyebrow">READY FOR SIMULATION</div>
            <h3 style="margin:8px 0">Your next mission starts here.</h3>
            <div class="small-muted">
                Choose a policy and seed in the sidebar, then run the mission.
                The result includes reward, target coverage, observation sequence,
                and a step-by-step action log.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ------------------------- POLICY BENCHMARK -------------------------

heading(
    "04",
    "Policy benchmark",
    "Compare baseline strategies and learned policies over repeated simulated missions."
)

if "comparison_results" not in st.session_state:
    st.session_state.comparison_results = None

if run_comparison:
    with st.spinner(f"Evaluating four policies across {episodes} episodes each…"):
        st.session_state.comparison_results = evaluate_policies(
            int(episodes),
            int(mission_seed),
            q_learning,
            sarsa,
            distance_matrix,
            centers,
        )

comparison = st.session_state.comparison_results

if comparison is not None:
    best_idx = comparison["Mean reward"].idxmax()
    best_policy = comparison.loc[best_idx, "Policy"]

    st.markdown(
        f"""
        <div class="panel" style="margin-bottom:16px">
            <div class="eyebrow">HIGHEST MEAN REWARD IN THIS RUN</div>
            <div style="font-size:26px;font-weight:800;margin-top:5px">{best_policy}</div>
            <div class="small-muted">
                This identifies the best mean reward in the current simulation only;
                it does not establish statistical significance or real-world superiority.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.dataframe(
        comparison.style.format({
            "Mean reward": "{:.3f}",
            "Reward SD": "{:.3f}",
            "Mean unique targets": "{:.2f}",
            "Mean diversity": "{:.1%}",
        }),
        use_container_width=True,
        hide_index=True,
    )

    chart_metric = st.selectbox(
        "Metric to visualize",
        ["Mean reward", "Mean unique targets", "Mean diversity", "Reward SD"],
    )

    fig, ax = plt.subplots(figsize=(9, 3.8))
    ordered = comparison.sort_values(chart_metric, ascending=False)
    ax.bar(ordered["Policy"], ordered[chart_metric])
    ax.set_ylabel(chart_metric)
    ax.set_title(f"Policy comparison · {episodes} episodes per policy")
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    st.download_button(
        "Download benchmark results (CSV)",
        comparison.to_csv(index=False).encode("utf-8"),
        file_name="orbital_policy_benchmark.csv",
        mime="text/csv",
    )
else:
    st.markdown(
        """
        <div class="panel">
            <h3 style="margin-top:0">Four strategies. One consistent test.</h3>
            <div class="small-muted">
                Random provides a baseline; Greedy novelty prioritizes unexplored
                targets; Q-learning and SARSA use the saved tables where compatible
                states are available. Use the sidebar to start the benchmark.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ------------------------- METHODS & LIMITATIONS -------------------------

heading(
    "05",
    "Research notes",
    "What this prototype tests—and what it does not claim."
)

with st.expander("Methods, assumptions, and limitations", expanded=False):
    st.markdown("""
    **Data and representation**
    - Multispectral samples are sourced from `blanchon/EuroSAT_MSI`.
    - The feature representation contains 13 band means and mean/standard
      deviation for NDVI, NDWI, and NDBI.
    - Spectral targets are unsupervised clusters. They should not automatically
      be interpreted as verified land-cover categories or geographic regions.

    **Reinforcement learning**
    - Q-learning and SARSA tables are loaded from the saved artifact.
    - The app attempts to match common state formats. When the reconstructed
      mission state does not match a saved table, the fallback is novelty-based.
      The fallback counter makes these decisions visible.
    - The dashboard's simplified mission simulator is not guaranteed to reproduce
      the exact environment used during notebook training.

    **Simulation assumptions**
    - Observation budget: six steps.
    - Cloud risk, energy costs, weather changes, novelty rewards, and penalties
      are illustrative simulation parameters, not measured satellite telemetry.
    - The app does not model orbital mechanics, real satellite scheduling,
      downlink constraints, or operational safety.

    **Evaluation**
    - Benchmark metrics are computed from newly simulated episodes.
    - Random seeds support repeatability for a fixed software environment.
    - Differences in mean reward alone do not demonstrate statistical significance.
    """)

st.markdown("---")
st.markdown(
    """
    <div style="text-align:center;padding:10px 0 24px 0">
        <div class="eyebrow">ORBITAL · EARTH INTELLIGENCE</div>
        <div class="small-muted" style="margin-top:8px">
            A research prototype for information-efficient observation planning.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)
