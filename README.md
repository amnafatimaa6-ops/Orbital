# ORBITAL

**Can a reinforcement learning agent learn to pick more useful satellite observation targets when it only gets a few shots and the weather might ruin them?**

ORBITAL is a small undergraduate research project that combines Sentinel-2 satellite imagery with reinforcement learning. I use real multispectral image patches to build a set of "targets", then train agents (Q-learning and SARSA) in a simulated environment to choose which targets to observe with a limited budget, and compare them against random and greedy strategies.

Everything lives in one notebook (`Notebook (2).ipynb`), and the results below are copied from its outputs.

> **Important:** the targets are *spectral clusters*, not real places on Earth, and the clouds, weather and costs are *simulated*. This is a learning/research prototype, not a real satellite scheduler.

---

## The idea in plain words

A satellite can't photograph everything. Each pass it has a limited number of observations, and clouds can ruin some of them. So which targets should it pick? A good choice is usually targets that look *different* from what it has already seen, so it doesn't waste shots on near-duplicates.

I turned that into a small game:

1. Take real satellite image patches and group them by how they look spectrally (8 groups = 8 "targets").
2. Measure how different each pair of targets is.
3. Let an agent pick targets one at a time (6 picks per episode). It gets rewarded for picking something new and penalized for cloud failures, repeats and costs.
4. See whether a learned policy beats random picking and a simple "pick the most different one" rule.

---

## Dataset

[EuroSAT MSI](https://huggingface.co/datasets/blanchon/EuroSAT_MSI), loaded through Hugging Face `datasets`.

| Property | Value |
|---|---|
| Patches available | 16,200 |
| Patches used | 600 (random sample, seed 42) |
| Patch size | 64 × 64 pixels |
| Spectral bands | 13 (Sentinel-2) |

The dataset's land-cover labels are **not** used. The clustering is fully unsupervised.

---

## What the notebook does

### 1. Feature extraction (19 features per patch)
- Mean value of each of the 13 bands
- Mean and standard deviation of **NDVI** (vegetation), **NDWI** (green vs NIR, used here as a water-related index) and **NDBI** (built-up/SWIR vs NIR), computed per pixel and then summarized per patch

Features are standardized with `StandardScaler`.

### 2. PCA for visualization
The 19 features are reduced to 2 components for a scatter plot. These two components capture **72.91%** of the variance.

### 3. Building the targets
`MiniBatchKMeans` groups the 600 patches into **8 clusters** (cluster sizes: 99, 107, 57, 82, 71, 71, 49, 64). Each cluster is a "target". Distances between cluster centers are normalized to 0–1 and stored as a **dissimilarity matrix**, which is used as a stand-in for "how new is this target compared to what I've already seen".

Cluster summaries show the targets make physical sense. For example, Target 2 is the only cluster with a positive mean NDWI (0.446), which is consistent with water-like patches, and Targets 4 and 5 have the highest NDVI (0.641 and 0.699), consistent with vegetation.

### 4. Environments

**v1 (simple):** budget of 6 observations, fixed 20% cloud-failure chance, observation cost 0.10. Reward = novelty (minimum distance to already-observed targets) − costs. A failed observation costs an extra 0.25 and a repeated target costs an extra 0.50.

**v2 (more realistic-ish, still synthetic):**
- cloud risk depends on the target (10% to 40%)
- each target has an energy cost (0.05 to 0.20)
- two weather regimes (good/bad) that flip with 25% chance each step; bad weather adds 0.25 to cloud probability
- weather is part of the agent's state

### 5. Agents
- **Q-learning** (tabular) and **SARSA** (tabular)
- State = last target, set of visited targets, remaining budget (+ weather in v2)
- 6,000 training episodes, learning rate 0.15, discount 0.95, epsilon-greedy with epsilon decaying from 1.0 to 0.05

### 6. Baselines
- **Random:** picks any target
- **Greedy:** picks the unvisited target farthest (in spectral distance) from the ones already visited. It ignores weather and costs.

---

## Results

### v1 environment (2,000 evaluation episodes)

| Policy | Mean reward | Reward std | Mean unique targets |
|---|---|---|---|
| Random | 0.802 | 0.733 | 3.792 |
| Greedy | **2.300** | 0.660 | 4.839 |
| Q-learning | 2.266 | 0.773 | 4.839 |

In the simple environment the greedy rule is actually slightly better than Q-learning (difference: −0.033). That makes sense, since greedy is almost exactly what the reward asks for. I kept this result because it's honest.

### v2 environment (2,000 evaluation episodes)

| Policy | Mean reward | Reward std | Mean unique targets |
|---|---|---|---|
| Random | -0.328 | 0.751 | 3.084 |
| Greedy | 0.524 | 1.002 | 3.553 |
| Q-learning | **0.536** | **0.683** | **3.929** |

When clouds, weather and energy costs are added, Q-learning edges ahead of greedy on mean reward, with lower variance and more unique targets. The reward gap versus greedy (0.536 vs 0.524) is very small, so I would not claim Q-learning is clearly better, only that it is competitive and more consistent here.

### Spectral diversity in v2 (1,000 episodes)

| Policy | Mean unique targets | Mean pairwise diversity |
|---|---|---|
| Random | 3.086 | 0.480 |
| Greedy | 3.579 | **0.661** |
| Q-learning | **3.950** | 0.514 |

Q-learning covers more distinct targets, but greedy picks sets that are *more different from each other*. So "more targets" and "more diverse targets" are not the same thing.

### Q-learning vs SARSA in v2 (2,000 held-out episodes)

| Agent | Mean reward | Reward std | Mean unique targets |
|---|---|---|---|
| Q-learning | **0.550** | 0.660 | **3.952** |
| SARSA | 0.421 | 0.610 | 3.760 |

### Training curves
- v1 Q-learning: mean reward rose from 0.926 (first 500 episodes) to 2.045 (last 500)
- v2 Q-learning: from -0.275 to 0.387
- v2 SARSA: from -0.297 to 0.395

### Other plots the notebook produces
PCA scatter plot, true-colour / false-colour / SWIR composites of a sample patch, NDVI/NDWI/NDBI maps, mean-index bar chart per cluster, sample images from Targets 2 and 5, dissimilarity heatmap, target-selection frequency chart (Random vs Q-learning), a 4-panel results summary (`orbital_results.png`) and a Q-learning vs SARSA comparison (`orbital_rl_comparison.png`).

---

## Limitations

- **Simulated conditions.** Clouds, weather, energy costs and rewards are made up. No real weather data, orbits, or satellite schedules are used.
- **Targets aren't places.** They are clusters in feature space, with no coordinates attached.
- **Novelty ≠ scientific value.** Spectral distance is a simple proxy; a "different-looking" cluster isn't automatically more important.
- **Small sample.** Only 600 of 16,200 patches, with one random seed for sampling and clustering.
- **Single runs, no statistics.** Each agent was trained once, so there are no confidence intervals or repeated seeds. Small differences (like Q-learning vs greedy in v2) could be noise.
- **Different goals.** Greedy only maximizes spectral distance, while the RL agents optimize a reward that also includes failures and costs, so the comparison isn't perfectly apples-to-apples. Greedy also ignores weather.
- **Tabular RL doesn't scale.** The state includes the set of visited targets, so the table grows quickly (about 3,100 states already with 8 targets).
- **Unseen states during evaluation.** Q-tables are `defaultdict`s, so a state never seen in training starts with all-zero values and the agent effectively picks at random there.
- **Same simulator for training and testing.** Evaluation uses different random seeds, but the same environment code, so this doesn't show the policy works in any other setting.
- **No downstream check.** I did not test whether the selected observations help a real task such as land-cover classification.

## Possible future work

- Keep coordinates/metadata so targets can be real locations
- Use real cloud-cover data and more realistic weather models
- Compare with stronger baselines (e.g. farthest-point sampling)
- Run many training seeds and report confidence intervals
- Try multi-objective rewards (diversity vs coverage vs cost)
- Use Double Q-learning or function approximation (e.g. a small neural network) so more targets are possible
- Give the greedy baseline the same costs/weather information for a fairer comparison
- Test whether the chosen observations improve a downstream task

---

## Repository contents

| File | Description |
|---|---|
| `Notebook (2).ipynb` | Full workflow: data, features, clustering, environments, training, evaluation, plots |
| `orbital_models.pkl` | Saved Q-learning and SARSA tables, scaler, clustering model, distance matrix |
| `metadata.json` | Experiment settings and feature names |
| `requirements.txt` | Python dependencies |
| `LICENSE` | MIT |

## How to run

Open the notebook and run the cells in order (the first cell installs `datasets`). Fixed seeds are used for sampling, clustering and training, though results may still vary slightly with library versions. Only load `.pkl` files from sources you trust.

## Tech

Python, NumPy, pandas, Matplotlib, scikit-learn, Hugging Face `datasets`, Sentinel-2 / EuroSAT MSI.

---

**Author:** Amna Fatima
