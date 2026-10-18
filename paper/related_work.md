# Related work and novelty check (T0.3, preliminary)

Status: **preliminary, from three web searches on 2026-10-18 (titles and abstracts only, papers not read).**
The owner should read the "close" papers, verify every citation (title, venue, year) and extend the search
(Google Scholar, Semantic Scholar) before any claim goes into a report.

## Close papers found by search

| Paper | What it does | How GreenQueue differs |
|---|---|---|
| Li, Liu, Ding, *Uncertainty-Aware Decarbonization for Datacenters* (arXiv 2407.02390) | Conformal prediction for carbon-intensity forecasts; evaluated on temporal and spatial load shifting; argues more uncertainty means more conservative shifting. | **Closest match.** GreenQueue's carbon-side conformal intervals are therefore NOT a novel claim. Differences to confirm after reading: (a) conformal bounds also on job duration/energy, (b) deadline-violation rate tracked against a target alpha under finite GPU capacity, (c) adaptive (ACI, delayed feedback) intervals under shift, (d) trace replay with an oracle bound. |
| LACS, *Learning-Augmented Algorithms for Carbon-Aware Resource Scaling with Uncertain Demand* (arXiv 2404.15211) | Unknown job length; learning-augmented online algorithm; injects forecast error. | Uncertainty handled by online-algorithm robustness, not calibrated intervals. |
| CarbonScaler (arXiv 2302.08681) | Re-plans the remaining schedule when progress or carbon deviates. | Re-planning (like our MPC) but no calibrated bounds. |
| LiteCast (arXiv 2511.06187) | Forecaster optimised for ranking of low/high-carbon periods; slack-window jobs. | Different forecaster objective; no interval guarantee. |
| Sohrabbeig, Ardakanian, Musilek (PESGM 2026 preprint) | Decision-focused carbon forecasting for placement. | Predict-then-optimise vs end-to-end; spatial. |
| CUFF (IBM Research) | Uncertainty-driven GPU demand forecasting for green AI clusters. | Demand, not job energy; conformal use not confirmed. |

No paper was found combining **conformal prediction with GPU job energy/duration estimation** (one search). The
energy-prediction papers found give point estimates (analytical A100 training model, NeuralPower, EnergAIzer).

## Consequence for the claims in implementation.md section 2

- C3 (conformal scheduler) must be narrowed to what is different from Li et al. (duration/energy bounds, alpha-tracking
  under capacity, adaptive intervals, honest negative results). Do not claim "first conformal carbon-aware scheduler".
- C2 (gray-box energy model with calibrated intervals) looks open, but the search was shallow.
- C1 (open benchmark) and C4 (average vs marginal) are not contradicted by anything found.

## Background references recalled by the assistant (NOT verified; check each before citing)

- Wiesner et al., "Let's Wait Awhile: How Temporal Workload Shifting Can Reduce Carbon Emissions in the Cloud", 2021.
- Radovanovic et al., "Carbon-Aware Computing for Datacenters", 2022.
- Sukprasert et al., "On the Limitations of Carbon-Aware Temporal and Spatial Workload Shifting in the Cloud", 2024.
- Romano, Patterson, Candes, "Conformalized Quantile Regression", NeurIPS 2019.
- Gibbs and Candes, "Adaptive Conformal Inference Under Distribution Shift", NeurIPS 2021.
- Angelopoulos and Bates, "A Gentle Introduction to Conformal Prediction and Distribution-Free Uncertainty Quantification".
- Weng et al., "MLaaS in the Wild: Workload Analysis and Scheduling in Large-Scale Heterogeneous GPU Clusters", NSDI 2022
  (the Alibaba trace used here; the authors ask users to cite it).
- You et al., "Zeus: Understanding and Optimizing GPU Energy Consumption of DNN Training", NSDI 2023.
