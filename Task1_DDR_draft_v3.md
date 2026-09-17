---
output:
  word_document: default
  pdf_document: default
  html_document: default
---
# CS5500 — Task 1 Dissertation Definition Report

**Student name:** Nachiket Magadum
**Student registration number:** _[INSERT YOUR BRUNEL STUDENT ID]_
**Student email:** _[INSERT YOUR @brunel.ac.uk EMAIL]_
**Dissertation Title:** Transformer-Based Anomaly Detection for Multivariate Time-Series: A Lightweight Variant with Transfer Learning and a Motorsport-Telemetry Case Study
**Supervisor name:** Professor Xiaohui Liu

---

## Abstract (~200 words)

Industrial systems across manufacturing, aviation, energy, and motorsport generate multivariate time-series telemetry from dozens of heterogeneous sensors. Detecting anomalies in such streams is critical for predictive maintenance, operational safety, and decision support, yet the rarity, subtlety, and cross-channel structure of anomalies makes the problem fundamentally difficult. Classical detectors such as Isolation Forest treat channels independently and miss the time-based and inter-channel patterns that distinguish a genuine fault from normal variation. Transformer-based approaches, including the recent Fusionformer architecture proposed by Wang et al. (2025), have shown strong results on industrial benchmarks through fusion attention and adversarial training, but their generalisation across datasets and their suitability for resource-constrained deployment remain underexplored.

This dissertation proposes a Fusionformer-inspired transformer for multivariate anomaly detection together with EdgeFusion, a lightweight variant using linear attention and knowledge distillation that targets comparable detection performance with substantially reduced computational footprint. The methodology is developed and evaluated on the Skoltech Anomaly Benchmark (SKAB) and the NASA MSL/SMAP datasets, with cross-dataset transfer learning analysis using established transferability estimation methods. A final chapter presents a cross-domain application to publicly available Formula 1 telemetry through the FastF1 library. The expected contribution is a critical evaluation of accuracy-efficiency-transferability trade-offs in transformer-based anomaly detection.

---

## 1. Background (~500 words)

Modern engineered systems are densely instrumented. A typical industrial asset streams tens to hundreds of channels of sensor data, including vibration, temperature, pressure, current, torque, and rotational speed, at frequencies from a few hertz to tens of kilohertz. High-performance systems such as Formula 1 cars share this same multi-sensor structure, with continuous telemetry of speed, throttle, brake, gear, engine RPM, tyre temperatures, and many other signals. In both cases the central operational question is the same: is the current behaviour normal, or does it signal an emerging fault?

This question defines the problem of multivariate time-series anomaly detection (Chandola, Banerjee and Kumar, 2009; Pang et al., 2021). It is harder than ordinary classification for several reasons. First, anomalies are rare and often not labelled in practice, complicating supervised learning approaches. Second, anomalies in multivariate streams are often defined by changes in the relationships between channels rather than by single-channel threshold violations. For example, bearing degradation may show subtle joint changes in vibration spectra and temperature long before any single sensor sets off an alarm. Third, the data is non-stationary, drift is common, and a model trained in one operating regime may transfer poorly to another.

Classical unsupervised detectors such as Isolation Forest (Liu, Ting and Zhou, 2008) remain widely used because they are computationally cheap and require no labels. My own undergraduate project, an Internet-of-Things crash-detection system using an ADXL345 accelerometer and an Isolation Forest classifier (Magadum et al., 2024), followed exactly this pattern and was recognised as the best final-year project of its cohort. The work also exposed the limits of channel-independent classical models: subtle multi-channel patterns and short-duration events were difficult to distinguish from noise.

Recent years have seen a rapid shift toward deep learning approaches, especially transformer-based detectors that exploit self-attention to model temporal and cross-channel dependencies jointly (Wen et al., 2023). The Anomaly Transformer (Xu et al., 2022) introduced anomaly-attention for distinguishing anomalous points; TranAD (Tuli et al., 2022) added adversarial training; PatchTST (Nie et al., 2023) demonstrated the value of patching for efficient sequence modelling. Fusionformer (Wang et al., 2025), the work of the proposed supervisor, advances this line further with fusion attention and adversarial training, achieving state-of-the-art results on industrial benchmarks.

Despite these advances, three practical gaps motivate further work. First, leading transformer-based detectors carry significant computational cost, and their suitability for edge or real-time deployment is rarely characterised; this matters because predictive maintenance in industrial and motorsport settings is increasingly expected to operate close to the sensors themselves rather than in the cloud. Second, models are typically trained and evaluated on a single dataset and their cross-dataset transferability is rarely studied directly, despite this being central to real deployments where labelled fault data per machine is scarce (Xue et al., 2024). Third, the field's evaluation methodology has itself been criticised, with point-adjusted F1 scores in particular shown to overstate progress (Wu and Keogh, 2021). This dissertation addresses all three gaps through a lightweight architectural variant, an explicit transfer learning study, and the consistent use of stricter evaluation metrics.

---

## 2. Aims and Objectives (~200 words)

### Aim
To develop and critically evaluate a transformer-based anomaly detection system for multivariate time-series, including a lightweight variant suitable for edge deployment and an explicit transfer learning study, and to assess its generalisation from industrial sensor streams to motorsport telemetry as a contrasting application domain.

### Objectives
1. Conduct a critical literature review of multivariate time-series anomaly detection, focusing on transformer-based methods, efficient attention mechanisms, transfer learning, and evaluation methodology, identifying the specific research gaps addressed in this dissertation.
2. Design and implement a Fusionformer-inspired transformer-based anomaly detection model (Wang et al., 2025) in PyTorch, including the full data pipeline, training loop, and experimental tracking infrastructure.
3. Design and implement EdgeFusion, a lightweight variant of the model using linear attention and knowledge distillation, with explicit benchmarking of parameter count, FLOPs, and inference latency against the full model.
4. Reproduce Isolation Forest, LSTM-autoencoder, and a standard transformer baseline on the SKAB benchmark, comparing the proposed models against these baselines under both standard and stricter evaluation metrics.
5. Investigate cross-dataset transfer learning to NASA MSL/SMAP, characterising when transfer succeeds or fails using transferability estimation methods (Xue et al., 2024).
6. As a cross-domain demonstration, apply the methodology to Formula 1 telemetry obtained via the FastF1 library, with critical discussion of strengths and limitations in this new setting.

---

## 3. Approach (~500 words)

The dissertation follows a quantitative experimental methodology grounded in the empirical norms of the deep learning literature. The work proceeds through five connected stages, each producing artefacts that feed both the experiments and the dissertation write-up.

**Data acquisition and preparation.** Three publicly available datasets are used. The Skoltech Anomaly Benchmark (SKAB; Katser and Maksimov, 2020) provides multivariate readings from industrial valves with hand-labelled anomalies. The NASA MSL and SMAP datasets (Hundman et al., 2018) provide spacecraft telemetry with channel-level anomaly labels and are the standard secondary benchmark for transfer learning studies. Formula 1 telemetry is accessed through the open-source FastF1 Python library, providing multivariate driver and car-state recordings from real race weekends. For each dataset, the data is cleaned, segmented using sliding windows, normalised per channel, and split with strict temporal validation discipline to avoid the leakage problems documented by Wu and Keogh (2021). All datasets are public and the project does not involve human participants; an ethics self-certification declaration will be filed with Brunel's Research Ethics process in Week 1 of the work.

**Baseline establishment.** Three baselines are reproduced on SKAB: an Isolation Forest, an LSTM-autoencoder, and a standard transformer-based anomaly detector. Each is evaluated using point-level precision, recall, and F1, together with PR-AUC and event-level F1, to surface the gap between point-adjusted scoring and stricter evaluation.

**Primary model development.** The primary model follows the Fusionformer design principles of fusion attention and adversarial training. The implementation will be carried out from scratch in PyTorch 2.x based on the published architecture, with hyperparameter selection through Optuna Bayesian optimisation on a validation split and a held-out test set untouched during model selection. Ablations isolate the contribution of fusion attention, the adversarial component, and the positional encoding choice. Experiment tracking is done using Weights & Biases throughout.

**EdgeFusion lightweight variant.** A second model, provisionally named EdgeFusion, is developed as a lightweight architectural variant. EdgeFusion replaces full quadratic self-attention with linear attention (Choromanski et al., 2021) and uses knowledge distillation (Hinton, Vinyals and Dean, 2015) from the trained primary model as teacher. The aim is to retain a substantial fraction of detection performance while substantially reducing parameter count, floating-point operations, and inference latency. EdgeFusion is benchmarked head-to-head against the primary model on identical data splits, with efficiency metrics measured on standard hardware including a Raspberry Pi 4 to simulate edge deployment.

**Transfer learning experiments.** Both models are transferred from SKAB to NASA MSL/SMAP under three regimes: zero-shot inference, linear probing of frozen features, and full fine-tuning on a small target sample. Transferability of feature representations is estimated using methods reviewed by Xue et al. (2024), including the H-score and LogME. Where time allows, adversarial domain adaptation is added as a fourth condition.

**Motorsport demonstration.** Finally, the methodology is applied to FastF1 telemetry for selected race weekends. Because the F1 data lacks ground-truth anomaly labels, evaluation is qualitative and cross-referenced with documented race incidents, mechanical retirements, and tyre-compound changes. This chapter is positioned as a cross-domain demonstration contingent on the success of the industrial experiments, and is subject to a scope-control checkpoint on 4 August 2026.

---

## 4. Plan (~200 words + Gantt chart)

The dissertation runs from 20 June 2026 (post-DDR) to 29 September 2026 (final submission), with a personal soft target of mid-September to allow polishing margin. Work is divided into four phases. Phase 1 (Weeks 1–4) establishes the development environment, completes the literature review, prepares the datasets, and trains the classical baselines. Phase 2 (Weeks 5–8) develops, tunes, and ablates the primary Fusionformer-inspired model, builds the EdgeFusion variant, and conducts the first transfer experiments to NASA MSL/SMAP. Phase 3 (Weeks 9–11) addresses the Formula 1 demonstration chapter, conditional on the success of Phase 2; should the industrial methodology not be producing publishable results by 4 August 2026, the F1 chapter is dropped and additional time is reinvested in deeper industrial ablations and EdgeFusion efficiency studies. Phase 4 (Weeks 12–14) is dedicated to writing, revising chapters with supervisor feedback, and producing final figures and references. My part-time work commitments (typically Thursday to Sunday) and a short period of unavailability between 25 and 29 June 2026 are accommodated in the schedule. The part-time schedule is flexible and can be rearranged around supervisor meetings as required. A continuous research log is maintained from Week 1 and fortnightly supervisor meetings are held throughout.

### Gantt chart (Weeks 1–14, starting 20 June 2026)

| Task | W1 | W2 | W3 | W4 | W5 | W6 | W7 | W8 | W9 | W10 | W11 | W12 | W13 | W14 |
|------|----|----|----|----|----|----|----|----|----|-----|-----|-----|-----|-----|
| Environment & repo setup, ethics application | ■ | | | | | | | | | | | | | |
| Literature review | ■ | ■ | ■ | ■ | ■ | ■ | | | | | | ■ | ■ | |
| Dataset acquisition (SKAB, NASA) | ■ | ■ | | | | | | | | | | | | |
| Personal travel (25-29 June) — light reading only | ▒ | | | | | | | | | | | | | |
| Baselines (IF, LSTM-AE, std Transformer) | | | ■ | ■ | | | | | | | | | | |
| Build primary Fusionformer-style model | | | | ■ | ■ | ■ | | | | | | | | |
| Hyperparameter tuning & ablations (primary) | | | | | | ■ | ■ | | | | | | | |
| Build EdgeFusion (linear attention + distillation) | | | | | | | ■ | ■ | | | | | | |
| Edge benchmarks (FLOPs, latency, Raspberry Pi) | | | | | | | | ■ | | | | | | |
| Transfer to NASA (zero-shot, linear probe, FT) | | | | | | | ■ | ■ | | | | | | |
| Transferability estimation (H-score, LogME) | | | | | | | | ■ | | | | | | |
| **Kill-switch checkpoint (4 Aug 2026)** | | | | | | | | ★ | | | | | | |
| F1 telemetry exploration (FastF1) | | | | | | | | | ■ | | | | | |
| F1 cross-domain experiments | | | | | | | | | | ■ | ■ | | | |
| Methodology chapter draft | | | | | ■ | ■ | | | | | | | | |
| Results chapters draft | | | | | | | | ■ | ■ | ■ | ■ | | | |
| Discussion & conclusion draft | | | | | | | | | | | | ■ | | |
| Full draft for supervisor review | | | | | | | | | | | | ★ | | |
| Apply feedback, revision | | | | | | | | | | | | | ■ | |
| Final polish, figures, references | | | | | | | | | | | | | ■ | ■ |
| **Final submission (29 September 2026)** | | | | | | | | | | | | | | ★ |

★ = milestone   ▒ = travel / partial availability

---

## References (Harvard style — alphabetical by first-author surname)

Chandola, V., Banerjee, A. and Kumar, V. (2009) Anomaly Detection: A Survey. *ACM Computing Surveys*, 41 (3), 1–58. doi: 10.1145/1541880.1541882.

Choromanski, K., Likhosherstov, V., Dohan, D., Song, X., Gane, A., Sarlós, T., Hawkins, P., Davis, J., Mohiuddin, A., Kaiser, Ł., Belanger, D., Colwell, L. and Weller, A. (2021) Rethinking Attention with Performers. In: *Proceedings of the International Conference on Learning Representations (ICLR)*. Online. arXiv: 2009.14794.

Hinton, G., Vinyals, O. and Dean, J. (2015) *Distilling the Knowledge in a Neural Network*. arXiv preprint arXiv:1503.02531.

Hundman, K., Constantinou, V., Laporte, C., Colwell, I. and Soderstrom, T. (2018) Detecting Spacecraft Anomalies Using LSTMs and Nonparametric Dynamic Thresholding. In: *Proceedings of the 24th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining*, 387–395. London, UK. doi: 10.1145/3219819.3219845. Dataset and code available: https://github.com/khundman/telemanom [Accessed: 16-06-2026].

Katser, I. and Maksimov, V. (2020) *Skoltech Anomaly Benchmark (SKAB)*. Skolkovo Institute of Science and Technology. Available: https://github.com/waico/SKAB [Accessed: 16-06-2026].

Liu, F.T., Ting, K.M. and Zhou, Z.-H. (2008) Isolation Forest. In: *Proceedings of the 8th IEEE International Conference on Data Mining*, 413–422. Pisa, Italy. doi: 10.1109/ICDM.2008.17.

Magadum, N., Vardhan, S.A., Duganavar, B.R. and Chinchanikar, C. (2024) *Accident Detection and Alert System Using IoT*. Final-Year Project Report, KLS Gogte Institute of Technology, Belagavi.

Nie, Y., Nguyen, N.H., Sinthong, P. and Kalagnanam, J. (2023) A Time Series is Worth 64 Words: Long-Term Forecasting with Transformers. In: *Proceedings of the International Conference on Learning Representations (ICLR)*. Kigali, Rwanda. arXiv: 2211.14730.

Oehrly, P. (2024) *FastF1: Python package for accessing Formula 1 timing and telemetry data*. Available: https://github.com/theOehrly/Fast-F1 [Accessed: 16-06-2026].

Pan, S.J. and Yang, Q. (2010) A Survey on Transfer Learning. *IEEE Transactions on Knowledge and Data Engineering*, 22 (10), 1345–1359. doi: 10.1109/TKDE.2009.191.

Pang, G., Shen, C., Cao, L. and van den Hengel, A. (2021) Deep Learning for Anomaly Detection: A Review. *ACM Computing Surveys*, 54 (2), 1–38. doi: 10.1145/3439950.

Tuli, S., Casale, G. and Jennings, N.R. (2022) TranAD: Deep Transformer Networks for Anomaly Detection in Multivariate Time Series Data. In: *Proceedings of the VLDB Endowment*, 15 (6), 1201–1214. doi: 10.14778/3514061.3514067.

Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A.N., Kaiser, Ł. and Polosukhin, I. (2017) Attention Is All You Need. In: *Advances in Neural Information Processing Systems*, 30. Long Beach, CA. arXiv: 1706.03762.

Wang, C., Wang, Z., Dong, H., Lauria, S., Liu, W., Wang, Y., Fadzil, F. and Liu, X. (2025) Fusionformer: A Novel Adversarial Transformer Utilizing Fusion Attention for Multivariate Anomaly Detection. *IEEE Transactions on Neural Networks and Learning Systems*, 36 (8), 14479–14492. doi: 10.1109/TNNLS.2025.3542719.

Wen, Q., Zhou, T., Zhang, C., Chen, W., Ma, Z., Yan, J. and Sun, L. (2023) Transformers in Time Series: A Survey. In: *Proceedings of the 32nd International Joint Conference on Artificial Intelligence (IJCAI)*. Macao, China. arXiv: 2202.07125.

Wu, R. and Keogh, E. (2021) Current Time Series Anomaly Detection Benchmarks are Flawed and are Creating the Illusion of Progress. *IEEE Transactions on Knowledge and Data Engineering*, 35 (3), 2421–2429. doi: 10.1109/TKDE.2021.3112126.

Xu, J., Wu, H., Wang, J. and Long, M. (2022) Anomaly Transformer: Time Series Anomaly Detection with Association Discrepancy. In: *Proceedings of the International Conference on Learning Representations (ICLR)*. Online. arXiv: 2110.02642.

Xue, Y., Yang, R., Chen, X., Liu, W., Wang, Z. and Liu, X. (2024) A Review on Transferability Estimation in Deep Transfer Learning. *IEEE Transactions on Artificial Intelligence*, 5 (12), 5894–5914. doi: 10.1109/TAI.2024.3445892.
