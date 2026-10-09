
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
    page_title="ORBITAL | Earth Observation Lab",
    page_icon="🛰️",
    layout="wide",
)

BAND_NAMES = [
    "B1 Coastal", "B2 Blue", "B3 Green", "B4 Red",
    "B5 Red Edge 1", "B6 Red Edge 2", "B7 Red Edge 3",
    "B8 Near-IR", "B8A Narrow Near-IR", "B9 Water Vapour",
    "B10 Cirrus", "B11 SWIR 1", "B12 SWIR 2",
]
FEATURE_NAMES = BAND_NAMES + [
    "NDVI mean", "NDVI std",
    "NDWI mean", "NDWI std",
    "NDBI mean", "NDBI std",
]

st.title("🛰️ ORBITAL")
st.subheader("Earth Observation Intelligence Lab")
st.markdown(
    "Can a reinforcement learning agent choose a diverse set of "
    "satellite observation targets when observations are limited?"
)

st.markdown(
    "**Research pipeline:** real multispectral imagery → spectral features "
    "→ eight unsupervised target clusters → simulated resource-constrained "
    "observation planning."
)

st.info(
    "The imagery comes from EuroSAT. The observation environment, weather "
    "changes, cloud probabilities, and energy costs are simulated assumptions. "
    "The system is a research prototype, not a physical satellite simulator."
)


# =========================================================
# 1. LOAD THE SAVED MODELS
# =========================================================

@st.cache_resource
def load_artifacts():
    model_path = ROOT / "orbital_models.pkl"
    metadata_path = ROOT / "metadata.json"

    if not model_path.exists():
        raise FileNotFoundError("orbital_models.pkl is missing.")

    # Only load pickle files from your own trusted repository.
    with open(model_path, "rb") as f:
        models = pickle.load(f)

    metadata = {}
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text())

    return models, metadata


try:
    models, metadata = load_artifacts()
    Q = models["q_learning"]
    SARSA = models["sarsa"]
    scaler = models["scaler"]
    cluster_model = models["region_model"]
    centers = np.asarray(models["target_features"])
    distances = np.asarray(models["distance_matrix"])

except Exception as exc:
    st.error(f"Could not load saved models: {exc}")
    st.stop()

N_TARGETS = len(distances)
BUDGET = int(metadata.get("rl_budget", 6))


# =========================================================
# 2. EXPLAINABLE REMOTE-SENSING FEATURES
# =========================================================

def spectral_features(image):
    image = np.asarray(image, dtype=np.float32)

    if image.shape != (64, 64, 13):
        raise ValueError(f"Unexpected image shape: {image.shape}")

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


def stretch(band):
    band = np.asarray(band, dtype=float)
    lo, hi = np.percentile(band, [2, 98])

    if hi <= lo:
        return np.zeros_like(band)

    return np.clip((band - lo) / (hi - lo), 0, 1)


def composite(image, bands):
    return np.dstack([stretch(image[:, :, b]) for b in bands])


def calculate_index(image, a, b):
    a = image[:, :, a].astype(float)
    b = image[:, :, b].astype(float)
    return (a - b) / (a + b + 1e-8)


def interpret_index(name, value):
    if name == "NDVI":
        if value > 0.4:
            return "Higher vegetation signal"
        if value > 0.15:
            return "Moderate vegetation signal"
        return "Low vegetation signal or non-vegetated surface"

    if name == "NDWI":
        if value > 0.1:
            return "Higher green-versus-NIR water-related signal"
        return "Green-versus-NIR signal is not strongly positive"

    if name == "NDBI":
        if value > 0.1:
            return "SWIR is relatively higher than near-infrared"
        return "SWIR is not strongly elevated relative to near-infrared"

    return "No interpretation available"


# =========================================================
# 3. RETRIEVE REAL MULTISPECTRAL IMAGES
# =========================================================

@st.cache_data(ttl=3600, show_spinner=False)
def load_eurosat_samples():
    api = "https://datasets-server.huggingface.co/rows"
    dataset_name = "blanchon/EuroSAT_MSI"

    # Small, distributed sample; we do not download the full dataset.
    offsets = [0, 100, 500, 1000, 1600, 3200, 4800, 6400, 8000, 9600, 11200]
    samples = []
    errors = []

    for offset in offsets:
        for attempt in range(2):
            try:
                response = requests.get(
                    api,
                    params={
                        "dataset": dataset_name,
                        "config": "default",
                        "split": "train",
                        "offset": offset,
                        "length": 1,
                    },
                    timeout=25,
                )
                response.raise_for_status()

                for item in response.json().get("rows", []):
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
                errors.append(str(exc))
                time.sleep(1 + attempt)

    if samples:
        return samples

    # Fallback: use the Hugging Face datasets library.
    try:
        from datasets import load_dataset

        dataset = load_dataset(
            dataset_name,
            split="train",
            streaming=True,
        )

        for i, row in enumerate(dataset):
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
                    "filename": str(row.get("filename", f"patch_{i}")),
                })

            if len(samples) >= 12:
                break

        if samples:
            return samples

    except Exception as exc:
        errors.append(str(exc))

    raise RuntimeError(
        "EuroSAT could not be retrieved. Try again later. "
        + " | ".join(errors[-2:])
    )


# =========================================================
# 4. RECREATE THE TRAINING SIMULATION FOR EVALUATION
# =========================================================

class OrbitalEnvironment:
    """Reproduction of the notebook's simulated OrbitalEnvV2."""

    def __init__(self, distance_matrix, budget=6, seed=42):
        self.distances = np.asarray(distance_matrix)
        self.n_targets = len(self.distances)
        self.initial_budget = budget
        self.observation_cost = 0.10
        self.rng = np.random.default_rng(seed)
        self.cloud_risk = np.linspace(0.10, 0.40, self.n_targets)
        self.energy_cost = np.linspace(0.05, 0.20, self.n_targets)

    def reset(self):
        self.budget = self.initial_budget
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
        self.budget -= 1
        reward = -self.observation_cost - self.energy_cost[action]

        if self.rng.random() < 0.25:
            self.weather = 1 - self.weather

        cloud_probability = self.cloud_risk[action]
        if self.weather == 1:
            cloud_probability = min(0.90, cloud_probability + 0.25)

        success = self.rng.random() >= cloud_probability
        novelty = 0.0

        if not success:
            reward -= 0.25

        elif action in self.visited:
            reward -= 0.50

        else:
            novelty = (
                min(self.distances[action, prev] for prev in self.visited)
                if self.visited else 1.0
            )
            reward += float(novelty)
            self.visited.add(action)

        self.last_target = action

        info = {
            "success": success,
            "cloud_probability": cloud_probability,
            "energy_cost": self.energy_cost[action],
            "novelty": novelty,
            "weather": self.weather,
            "unique_targets": len(self.visited),
        }

        return self.state(), reward, info


def greedy_action(env):
    unvisited = [
        i for i in range(env.n_targets)
        if i not in env.visited
    ]

    if not unvisited:
        return int(env.rng.integers(env.n_targets))

    if not env.visited:
        return int(env.rng.integers(env.n_targets))

    return max(
        unvisited,
        key=lambda i: min(
            env.distances[i, previous]
            for previous in env.visited
        ),
    )


def choose_action(policy, state, env):
    if policy == "Random":
        return int(env.rng.integers(env.n_targets))

    if policy == "Greedy novelty":
        return greedy_action(env)

    table = Q if policy == "Q-learning" else SARSA
    values = table.get(state)

    if values is not None:
        return int(np.argmax(values))

    # States unseen during training use an explicit fallback.
    return greedy_action(env)


def run_episode(policy, seed):
    env = OrbitalEnvironment(distances, budget=BUDGET, seed=seed)
    state = env.reset()
    total_reward = 0.0
    rows = []

    while env.budget > 0:
        action = choose_action(policy, state, env)
        next_state, reward, info = env.step(action)
        total_reward += reward

        rows.append({
            "Step": len(rows) + 1,
            "Target selected": action,
            "Observation successful": info["success"],
            "Simulated cloud risk": round(info["cloud_probability"], 3),
            "Energy cost": round(info["energy_cost"], 3),
            "Spectral novelty": round(info["novelty"], 3),
            "Reward": round(reward, 3),
            "Unique targets": info["unique_targets"],
        })

        state = next_state

    visited = sorted(env.visited)

    if len(visited) >= 2:
        pairs = [
            distances[visited[i], visited[j]]
            for i in range(len(visited))
            for j in range(i + 1, len(visited))
        ]
        diversity = float(np.mean(pairs))
    else:
        diversity = 0.0

    return {
        "reward": total_reward,
        "unique_targets": len(visited),
        "diversity": diversity,
        "visited": visited,
        "rows": rows,
    }


@st.cache_data(show_spinner=False)
def evaluate_policies(n_episodes):
    records = []

    policies = ["Random", "Greedy novelty", "Q-learning", "SARSA"]

    for policy in policies:
        for episode in range(n_episodes):
            result = run_episode(
                policy,
                seed=10000 + episode,
            )

            records.append({
                "Policy": policy,
                "Episode": episode + 1,
                "Reward": result["reward"],
                "Unique targets": result["unique_targets"],
                "Pairwise diversity": result["diversity"],
            })

    return pd.DataFrame(records)


# =========================================================
# 5. TOP-LEVEL NAVIGATION
# =========================================================

overview, imagery, clusters, planning, evaluation, methods = st.tabs([
    "Overview",
    "Satellite explorer",
    "Spectral targets",
    "Observation planner",
    "Policy evaluation",
    "Methodology",
])


# =========================================================
# TAB 1: OVERVIEW
# =========================================================

with overview:
    st.header("Mission overview")

    a, b, c, d = st.columns(4)
    a.metric("Spectral targets", N_TARGETS)
    b.metric("Features per patch", int(scaler.n_features_in_))
    c.metric("Q-learning states", len(Q))
    d.metric("SARSA states", len(SARSA))

    st.markdown("### How ORBITAL works")

    st.markdown("""
    **01 — Observe:** obtain a multispectral satellite image patch.

    **02 — Characterize:** calculate band statistics and NDVI, NDWI and NDBI.

    **03 — Discover:** use MiniBatchKMeans to organize patches into eight
    spectral clusters.

    **04 — Compare:** calculate distances between the learned target centroids.

    **05 — Plan:** use tabular reinforcement learning to choose targets under
    a limited observation budget.

    **06 — Evaluate:** compare Q-learning and SARSA against random and greedy
    baselines in the simulated environment.
    """)

    st.markdown("### What does the agent optimize?")

    st.write(
        "The reward combines novelty from visiting a spectrally different "
        "target with penalties for simulated cloud obstruction, energy and "
        "observation costs, and repeated observations."
    )

    st.markdown("### Saved model status")
    st.success("Saved clustering model, scaler, Q-learning table and SARSA table loaded.")

    st.markdown("### Research limitations")
    st.write(
        "The target clusters are spectral groups, not verified geographic "
        "regions. The environment does not model orbital mechanics, actual "
        "satellite telemetry, real forecasts, or operational mission constraints."
    )


# =========================================================
# TAB 2: SATELLITE EXPLORER
# =========================================================

with imagery:
    st.header("Explore real satellite imagery")

    st.write(
        "A multispectral image records reflected energy at different "
        "wavelengths. Different band combinations highlight different surface properties."
    )

    if st.button("Load EuroSAT samples", type="primary"):
        try:
            with st.spinner("Retrieving image patches..."):
                st.session_state["samples"] = load_eurosat_samples()
        except Exception as exc:
            st.error(str(exc))

    samples = st.session_state.get("samples", [])

    if samples:
        idx = st.selectbox(
            "Choose a patch",
            range(len(samples)),
            format_func=lambda i: (
                f"Patch {i + 1}: {samples[i]['filename']}"
            ),
        )

        item = samples[idx]
        image = item["image"]

        try:
            features = spectral_features(image).reshape(1, -1)
            standardized = scaler.transform(features)
            target = int(cluster_model.predict(standardized)[0])
            centroid = centers[target]

            st.success(f"Model-assigned spectral target: {target}")

            left, right = st.columns(2)

            with left:
                st.markdown("#### Natural-colour approximation")
                st.image(
                    composite(image, [3, 2, 1]),
                    use_container_width=True,
                )
                st.caption(
                    "Red, green and blue bands. Display contrast is stretched "
                    "for visualization."
                )

            with right:
                st.markdown("#### False-colour vegetation view")
                st.image(
                    composite(image, [7, 3, 2]),
                    use_container_width=True,
                )
                st.caption(
                    "Near-infrared, red and green assigned to display RGB."
                )

            st.markdown("### Spectral indices")

            index_specs = [
                ("NDVI", 7, 3, "Vegetation-related spectral contrast"),
                ("NDWI", 2, 7, "Green versus near-infrared contrast"),
                ("NDBI", 11, 7, "SWIR versus near-infrared contrast"),
            ]

            fig, axes = plt.subplots(1, 3, figsize=(13, 4))

            for ax, (name, band_a, band_b, description) in zip(
                axes, index_specs
            ):
                values = calculate_index(image, band_a, band_b)
                plot = ax.imshow(
                    values,
                    cmap="RdYlGn",
                    vmin=-1,
                    vmax=1,
                )
                ax.set_title(name)
                ax.axis("off")
                fig.colorbar(plot, ax=ax, fraction=0.046)

            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

            raw = features[0]
            stats = {
                "NDVI": raw[13],
                "NDWI": raw[15],
                "NDBI": raw[17],
            }

            st.markdown("### What do the index values suggest?")

            cols = st.columns(3)

            for col, (name, value) in zip(cols, stats.items()):
                col.metric(name, f"{value:.3f}")
                col.write(interpret_index(name, float(value)))

            st.caption(
                "These are approximate interpretations of index values, "
                "not ground-truth land-cover classifications. Index thresholds "
                "depend on the dataset and surface conditions."
            )

            with st.expander("Inspect all 19 extracted features"):
                st.dataframe(
                    pd.DataFrame({
                        "Feature": FEATURE_NAMES,
                        "Value": raw,
                    }),
                    use_container_width=True,
                )

            st.write("Dataset label:", item["label"])
            st.write("Filename:", item["filename"])

        except Exception as exc:
            st.error(f"Image analysis failed: {exc}")

    else:
        st.info("Load a small set of real EuroSAT patches to begin exploring.")


# =========================================================
# TAB 3: SPECTRAL TARGETS
# =========================================================

with clusters:
    st.header("Eight learned spectral targets")

    st.write(
        "MiniBatchKMeans groups image patches using the 19-dimensional "
        "feature representation. The distance matrix describes separation "
        "between cluster centroids in standardized feature space."
    )

    target = st.selectbox(
        "Inspect target cluster",
        list(range(N_TARGETS)),
        key="cluster_target",
    )

    centroid_scaled = centers[target]
    centroid_original = scaler.inverse_transform(
        centroid_scaled.reshape(1, -1)
    )[0]

    st.subheader(f"Target {target} profile")

    st.write(
        "The plot below shows the cluster centroid after reversing feature "
        "standardization. The first 13 values are dataset band means; the "
        "last six are index means and standard deviations."
    )

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.bar(FEATURE_NAMES, centroid_original)
    ax.set_ylabel("Approximate original feature scale")
    ax.set_title(f"Spectral feature profile — Target {target}")
    ax.tick_params(axis="x", rotation=65)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    profile = pd.DataFrame({
        "Index statistic": [
            "NDVI mean", "NDVI standard deviation",
            "NDWI mean", "NDWI standard deviation",
            "NDBI mean", "NDBI standard deviation",
        ],
        "Value": centroid_original[13:19],
    })

    st.markdown("### Index summary")
    st.dataframe(profile.round(4), use_container_width=True)

    st.markdown("### How distinct is this target?")

    other_distances = distances[target].copy()
    other_distances[target] = np.nan

    st.metric(
        "Mean distance to other target centroids",
        f"{np.nanmean(other_distances):.3f}",
    )

    st.write(
        "A larger mean distance indicates greater separation from the other "
        "learned centroids under the standardized feature representation. "
        "It does not establish geographic uniqueness."
    )

    st.markdown("### Target distance matrix")

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(distances, cmap="viridis")
    ax.set_xticks(range(N_TARGETS))
    ax.set_yticks(range(N_TARGETS))
    ax.set_xticklabels(range(N_TARGETS))
    ax.set_yticklabels(range(N_TARGETS))
    ax.set_xlabel("Target")
    ax.set_ylabel("Target")
    ax.set_title("Normalized pairwise spectral distances")
    fig.colorbar(im, ax=ax, label="Normalized distance")
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


# =========================================================
# TAB 4: INTERACTIVE OBSERVATION PLANNER
# =========================================================

with planning:
    st.header("Simulated observation planning")

    st.write(
        "Choose an agent and run a six-action episode. Each action attempts "
        "to observe one of the eight spectral targets. Clouds, weather, and "
        "energy costs are simulated."
    )

    policy = st.selectbox(
        "Planning policy",
        ["Q-learning", "SARSA", "Greedy novelty", "Random"],
        key="planner_policy",
    )

    seed = st.number_input(
        "Random seed",
        min_value=0,
        max_value=999999,
        value=42,
        step=1,
    )

    if st.button("Run simulated mission", type="primary"):
        result = run_episode(policy, int(seed))
        st.session_state["last_episode"] = {
            "policy": policy,
            **result,
        }

    episode = st.session_state.get("last_episode")

    if episode:
        m1, m2, m3 = st.columns(3)
        m1.metric("Total reward", f"{episode['reward']:.3f}")
        m2.metric(
            "Unique targets observed",
            f"{episode['unique_targets']} / {N_TARGETS}",
        )
        m3.metric("Mean pairwise diversity", f"{episode['diversity']:.3f}")

        st.markdown("### Action-by-action explanation")

        frame = pd.DataFrame(episode["rows"])
        st.dataframe(frame, use_container_width=True)

        st.markdown("### Target selection sequence")

        chosen = [
            row["Target selected"]
            for row in episode["rows"]
        ]

        st.write(" → ".join(f"Target {x}" for x in chosen))

        st.markdown("### Targets successfully observed")

        observed = episode["visited"]
        if observed:
            st.write(", ".join(f"Target {x}" for x in observed))
        else:
            st.write("No successful observations in this episode.")

        st.caption(
            "This is one simulated episode. If the Q-table does not contain "
            "a encountered state, the planner uses a greedy-novelty fallback. "
            "The fallback is shown to avoid pretending every state was learned."
        )


# =========================================================
# TAB 5: REPEATABLE POLICY EVALUATION
# =========================================================

with evaluation:
    st.header("Compare observation policies")

    st.write(
        "Evaluate the saved Q-learning and SARSA tables against random "
        "selection and a greedy novelty baseline in the same simulated "
        "environment. These are fresh evaluation runs, not the original "
        "training metrics."
    )

    episode_count = st.select_slider(
        "Evaluation episodes per policy",
        options=[25, 50, 100, 200],
        value=100,
    )

    if st.button("Evaluate all four policies"):
        with st.spinner("Running simulated evaluation episodes..."):
            result_df = evaluate_policies(int(episode_count))
            st.session_state["evaluation_results"] = result_df

    result_df = st.session_state.get("evaluation_results")

    if result_df is not None:
        summary = result_df.groupby("Policy").agg(
            Mean_reward=("Reward", "mean"),
            Reward_std=("Reward", "std"),
            Mean_unique_targets=("Unique targets", "mean"),
            Mean_pairwise_diversity=("Pairwise diversity", "mean"),
        ).reset_index()

        st.markdown("### Evaluation summary")
        st.dataframe(
            summary.round(3),
            use_container_width=True,
        )

        st.markdown("### Mean episode reward")

        fig, ax = plt.subplots(figsize=(9, 4))
        reward_means = result_df.groupby("Policy")["Reward"].mean()
        reward_means.plot(kind="bar", ax=ax)
        ax.set_ylabel("Mean total episode reward")
        ax.set_xlabel("")
        ax.set_title("Policy reward comparison")
        ax.tick_params(axis="x", rotation=20)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        st.markdown("### Target coverage")

        fig, ax = plt.subplots(figsize=(9, 4))
        coverage = result_df.groupby("Policy")["Unique targets"].mean()
        coverage.plot(kind="bar", ax=ax)
        ax.set_ylabel("Mean unique targets per episode")
        ax.set_xlabel("")
        ax.set_title("Observation coverage comparison")
        ax.tick_params(axis="x", rotation=20)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        st.markdown("### Spectral diversity")

        fig, ax = plt.subplots(figsize=(9, 4))
        diversity = result_df.groupby("Policy")["Pairwise diversity"].mean()
        diversity.plot(kind="bar", ax=ax)
        ax.set_ylabel("Mean pairwise centroid distance")
        ax.set_xlabel("")
        ax.set_title("Spectral diversity comparison")
        ax.tick_params(axis="x", rotation=20)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        st.download_button(
            "Download evaluation results CSV",
            result_df.to_csv(index=False).encode("utf-8"),
            file_name="orbital_fresh_policy_evaluation.csv",
            mime="text/csv",
        )

        st.caption(
            "Interpret results across multiple episodes. Random seeds and "
            "the environment are controlled for reproducibility, but a single "
            "training seed and one evaluation configuration do not establish "
            "statistical superiority."
        )

    else:
        st.info(
            "Run the evaluation to generate new policy comparisons. "
            "The results will be calculated from the saved Q-tables and "
            "the reconstructed simulation."
        )


# =========================================================
# TAB 6: METHODS AND REPRODUCIBILITY
# =========================================================

with methods:
    st.header("Methodology and reproducibility")

    st.markdown("### Data and features")

    st.write(
        "The experiment used 600 EuroSAT multispectral patches during feature "
        "extraction and clustering. Each patch has 64 × 64 pixels and 13 bands. "
        "The feature vector consists of 13 band means plus the mean and standard "
        "deviation of NDVI, NDWI, and NDBI."
    )

    st.markdown("### Unsupervised target discovery")

    st.write(
        "StandardScaler normalizes the features before MiniBatchKMeans "
        "learns eight spectral centroids. Distances between the centroids "
        "provide a proxy for spectral dissimilarity."
    )

    st.markdown("### Reinforcement learning")

    st.write(
        "The tabular state includes the previous target, the set of previously "
        "visited targets, remaining observation budget, and simulated weather "
        "regime. The reward encourages novel observations while penalizing "
        "cloud-obstructed observations, repeated targets, and resource costs."
    )

    st.markdown("### Saved artifacts")

    artifact_rows = []

    for name, value in models.items():
        if name in ("q_learning", "sarsa"):
            description = f"{len(value)} saved states"
        elif hasattr(value, "shape"):
            description = f"Shape: {value.shape}"
        else:
            description = type(value).__name__

        artifact_rows.append({
            "Artifact": name,
            "Status": "Loaded",
            "Details": description,
        })

    st.dataframe(
        pd.DataFrame(artifact_rows),
        use_container_width=True,
    )

    st.markdown("### Limitations")

    st.markdown("""
    - The target clusters are not ground-truth land-cover labels.
    - Cloud probabilities and energy costs are simulated.
    - No orbital mechanics or real spacecraft telemetry is modeled.
    - Unseen states in the saved Q-tables use a fallback action.
    - Results depend on training, environment assumptions, and random seeds.
    - Performance comparisons should be repeated across independent training
      seeds before drawing strong conclusions.
    """)

    st.markdown("### Metadata")
    st.json(metadata)


st.divider()
st.caption(
    "ORBITAL | Multispectral remote sensing + unsupervised learning + "
    "tabular reinforcement learning"
)
