# ORBITAL

### Reinforcement Learning for Information-Efficient Earth Observation

ORBITAL is an experimental research project exploring how reinforcement learning can help prioritize Earth-observation targets under limited observation budgets, uncertain weather conditions, and varying spectral information value.

The project combines multispectral remote-sensing data from Sentinel-2 imagery with spectral feature engineering, unsupervised clustering, and tabular reinforcement learning. It investigates whether an intelligent observation policy can select a more informative and diverse set of targets than a random strategy.

The work was developed as a reproducible Python research notebook, with experiments covering spectral analysis, target selection, reward design, and policy evaluation.

## Research Objectives

The central question behind ORBITAL is:

**Can a reinforcement learning agent improve observation efficiency by prioritizing spectrally diverse targets when observation opportunities are limited and conditions are uncertain?**

To investigate this question, the project focuses on five objectives:

* Extract meaningful spectral features from multispectral satellite imagery.
* Group observations into spectral clusters to construct candidate observation targets.
* Design a simulated environment with limited observation budgets, cloud-related failures, and observation costs.
* Train and compare Q-learning and SARSA agents.
* Evaluate learned policies against random selection and a greedy diversity-based baseline.

## Dataset

ORBITAL uses the [EuroSAT MSI dataset](https://huggingface.co/datasets/blanchon/EuroSAT_MSI), accessed through the Hugging Face `datasets` library.

The dataset contains multispectral Sentinel-2 image patches representing different land-cover classes.

| Property                     | Value             |
| ---------------------------- | ----------------- |
| Dataset                      | EuroSAT MSI       |
| Total available observations | 16,200            |
| Sample used in experiments   | 600 image patches |
| Image dimensions             | 64 × 64 pixels    |
| Spectral bands               | 13                |
| Extracted features           | 19                |
| Candidate spectral targets   | 8                 |

A fixed random seed is used when sampling observations to support reproducibility.

## Methodology

### 1. Multispectral feature engineering

Each image patch is represented by 19 numerical features:

* Mean reflectance across the 13 spectral bands.
* Mean and standard deviation of NDVI.
* Mean and standard deviation of NDWI.
* Mean and standard deviation of NDBI.

The indices are calculated pixel-wise and then summarized across each image patch.

The extracted features are standardized before clustering and dimensionality reduction.

### 2. Spectral analysis and dimensionality reduction

Principal Component Analysis (PCA) reduces the 19-dimensional standardized feature space to two components for visualization.

In the reported experiment, the first two principal components represent **72.91% of the variance** in the standardized feature matrix.

This projection provides a compact view of spectral variation across the sampled observations.

### 3. Unsupervised target discovery

MiniBatch K-Means groups the 600 observations into eight spectral clusters.

The cluster centroids represent characteristic spectral profiles. Pairwise Euclidean distances between these centroids are normalized to construct a target-dissimilarity matrix.

This matrix is used as a proxy for spectral novelty when selecting observations.

Importantly, these targets are **spectral clusters, not geographic locations or independently verified satellite acquisition sites**.

### 4. Simulated Earth-observation environment

A custom environment models sequential target selection under a limited observation budget.

The environment incorporates:

* A finite number of observation opportunities.
* Target-dependent cloud risks.
* Stochastic weather regimes.
* Observation and energy costs.
* Penalties for failed observations and repeated selections.
* Rewards for successfully observing spectrally novel targets.

The initial environment uses a fixed cloud-failure probability. A second version introduces target-dependent cloud risks and two simulated weather regimes.

These environmental conditions are synthetic and are intended to test decision-making behavior, not to reproduce actual satellite operations or forecast real cloud cover.

### 5. Reinforcement learning

Two tabular reinforcement learning algorithms are implemented:

**Q-learning** learns action values using the maximum estimated value of the next state.

**SARSA** learns action values using the next action selected by its current epsilon-greedy policy.

Both agents use a discrete state representation containing the last selected target, the set of visited targets, the remaining observation budget, and, in the second environment, the simulated weather regime.

The main reported training configuration uses 6,000 episodes, a learning rate of 0.15, a discount factor of 0.95, and an epsilon-greedy exploration schedule.

### 6. Policy evaluation

The experiments compare three selection strategies:

* **Random:** selects targets without an information-based strategy.
* **Greedy:** prioritizes the target with the greatest minimum spectral distance from previously visited targets.
* **Q-learning:** selects targets using learned action values.

The evaluation considers mean episode reward, reward variability, unique targets observed, and pairwise spectral diversity.

A separate experiment compares Q-learning with SARSA.

## Results

### Policy comparison in the second simulated environment

The following values are from the reported 2,000-episode evaluation of the second environment.

| Policy     | Mean reward | Reward standard deviation | Mean unique targets |
| ---------- | ----------: | ------------------------: | ------------------: |
| Random     |      -0.328 |                     0.751 |               3.084 |
| Greedy     |       0.524 |                     1.002 |               3.553 |
| Q-learning |       0.536 |                     0.683 |               3.929 |

In this experiment, Q-learning achieves the highest mean reward and observes more unique targets on average than either baseline.

The mean reward difference between Q-learning and the greedy baseline is relatively small. This suggests that the learned policy offers a modest improvement in this particular simulated setting, rather than establishing a general advantage over heuristic methods.

### Spectral diversity

A separate evaluation reports the following mean pairwise spectral-diversity scores:

| Policy     | Mean unique targets | Mean pairwise diversity |
| ---------- | ------------------: | ----------------------: |
| Random     |               3.086 |                   0.480 |
| Greedy     |               3.579 |                   0.661 |
| Q-learning |               3.950 |                   0.514 |

These results reveal an important trade-off. Q-learning observes more unique targets on average, while the greedy baseline achieves greater mean pairwise spectral diversity.

Consequently, observing more unique targets does not necessarily mean selecting the most spectrally diverse collection. The results motivate further work on multi-objective reward design.

### Q-learning versus SARSA

The reported held-out comparison gives the following results:

| Algorithm  | Mean reward | Reward standard deviation | Mean unique targets |
| ---------- | ----------: | ------------------------: | ------------------: |
| Q-learning |       0.550 |                     0.660 |               3.952 |
| SARSA      |       0.421 |                     0.610 |               3.760 |

Under this evaluation configuration, Q-learning achieves higher mean reward and greater observation coverage than SARSA. These results describe the tested environment and configuration; further repeated experiments are required to establish how consistently the difference holds.

### Spectral analysis outputs

The notebook also generates:

* A two-component PCA projection of the spectral feature space.
* True-colour and false-colour image composites.
* NDVI, NDWI, and NDBI maps for a representative image patch.
* Cluster-level spectral-index summaries.
* A normalized spectral-dissimilarity matrix.
* Target-selection frequency plots.
* Training curves and reinforcement learning comparisons.

Together, these outputs connect the remote-sensing feature pipeline with the decision-making experiments.

## Limitations

ORBITAL is a research prototype, and several limitations should be considered when interpreting its results.

**Synthetic operating conditions**

Cloud probabilities, weather transitions, energy costs, and observation rewards are simulated. The environment does not use real-time weather, satellite telemetry, orbital constraints, or operational acquisition data.

**Limited sample size**

The main experiments use 600 patches from a dataset containing 16,200 observations. Results may change with different samples, land-cover distributions, and clustering configurations.

**Spectral clusters are not physical locations**

The eight targets represent clusters in feature space. They do not correspond to actual coordinates, satellite ground tracks, or independently validated regions of interest.

**Simplified information-value objective**

Spectral dissimilarity is used as a proxy for novelty. Greater spectral distance does not automatically imply greater scientific importance, environmental change, or information gain.

**Tabular reinforcement learning**

The state includes the set of visited targets, causing the state space to grow rapidly. Tabular Q-learning and SARSA may become inefficient as the number of targets or environmental variables increases.

**Evaluation uncertainty**

The reported experiments use particular random seeds and configurations. More independent training runs, confidence intervals, and statistical comparisons are needed to assess the robustness of the results.

**Potential reward and policy mismatch**

The greedy baseline explicitly maximizes minimum spectral distance, while the reinforcement learning agents optimize a reward that also includes observation failures and costs. The different objectives help explain why the policies may perform differently on reward and diversity metrics.

## Future Improvements

The next stages of ORBITAL could address these limitations in several ways.

1. **Real geographic targets:** Retain coordinates and metadata for candidate observations so that target selection can incorporate location, accessibility, and spatial coverage.

2. **Realistic environmental data:** Introduce cloud-cover observations, acquisition constraints, and more realistic weather transition models.

3. **Improved reward engineering:** Investigate explicit trade-offs among information gain, spectral diversity, observation success, energy expenditure, and scientific relevance.

4. **Stronger baselines:** Compare against random selection, greedy novelty, farthest-point sampling, and other established optimization strategies under matched objectives and evaluation conditions.

5. **Robust evaluation:** Run multiple independent training seeds, use held-out samples, report confidence intervals, and test sensitivity to cloud risk, observation budgets, and reward weights.

6. **Scalable reinforcement learning:** Explore Double Q-learning, function approximation, or other methods that can handle larger target sets and richer state representations.

7. **Uncertainty-aware target selection:** Estimate uncertainty in spectral features and target value to distinguish genuinely informative observations from noisy or ambiguous ones.

8. **Scientific validation:** Evaluate whether the selected observations improve a downstream task, such as land-cover classification, change detection, or identification of unusual spectral patterns.

9. **Reproducible research packaging:** Publish experiment configurations, trained model artifacts, dependency versions, and a clear notebook workflow for reproducing the results.

## Technology Stack

* Python
* NumPy and pandas
* Matplotlib
* scikit-learn
* Hugging Face `datasets`
* Sentinel-2 multispectral imagery
* Principal Component Analysis
* MiniBatch K-Means
* Tabular Q-learning and SARSA

## Reproducibility

The research is organized as a notebook-based workflow, from dataset loading and feature extraction through clustering, environment construction, agent training, evaluation, and visualization.

To reproduce the experiments, run the notebook cells in order. The notebook uses fixed random seeds in its principal sampling, clustering, and training configurations, although results can still depend on software versions and execution details.

The trained agents and preprocessing artifacts can be saved for subsequent evaluation. When sharing serialized models, use compatible dependency versions and include the experiment metadata required to interpret the artifacts.

## Conclusion

ORBITAL explores the intersection of multispectral remote sensing and reinforcement learning through a controlled simulation of limited-budget observation planning.

The experiments show that Q-learning can outperform random selection and achieve slightly higher mean reward than a greedy novelty-based strategy in one simulated environment. At the same time, the spectral-diversity results demonstrate that no single metric captures every aspect of observation quality.

The most important next step is to move beyond synthetic assumptions and evaluate the approach against stronger baselines, realistic acquisition constraints, and downstream scientific objectives.

ORBITAL is an ongoing research prototype, intended as a foundation for further experimentation in intelligent Earth observation rather than a validated operational satellite-planning system.

---

**Author:** Amna Fatima

**Project:** ORBITAL — Reinforcement Learning for Information-Efficient Earth Observation
