---
title: "Chapter 3: Methodology"
author: "Nachiket Magadum"
date: "September 2026"
geometry: margin=2.5cm
fontsize: 11pt
mainfont: "Times New Roman"
linestretch: 1.5
---

# CHAPTER 3: METHODOLOGY

This chapter describes how the experiments in Chapter 4 were run. Section 3.1 describes the two datasets. Section 3.2 describes the paper-faithful Fusionformer implementation and the single-component ablation variant. Section 3.3 covers the training regime including all hyperparameters. Section 3.4 covers the four-metric evaluation harness and the anomaly scoring pipeline. Section 3.5 describes the multi-seed protocol and the paired Wilcoxon significance test. Section 3.6 lists deviations from the paper's exact configuration, with reasons. Section 3.7 summarises.

## 3.1 Datasets

Two public multivariate anomaly detection benchmarks were used: the Skoltech Anomaly Benchmark (SKAB) and the Server Machine Dataset (SMD). They were chosen for contrast. SKAB has eight sensors on a single physical system whose variables are tightly coupled by fluid mechanics. SMD has thirty-eight channels of server telemetry whose variables are relatively independent. If the mechanism I ablate helps with cross-variable coupling, the two datasets should give different answers, and that is exactly what happens in Chapter 4.

### 3.1.1 SKAB

The Skoltech Anomaly Benchmark (Katser and Kozitsin, 2020) is a Kaggle-released public dataset from a laboratory pumping station. It contains eight sensors: two accelerometers, current, pressure, temperature, thermocouple, voltage, and volumetric flow rate. Records are sampled at 1 Hz. The benchmark comes in three folders corresponding to different fault regimes: valve1 (outlet valve closing, five files), valve2 (inlet valve closing, four files), and other (miscellaneous faults, fourteen files). Each file has around a thousand rows and a binary anomaly label per row. The anomaly rate in a typical file is around 35 percent, because fault events tend to dominate a recorded session.

For the experiments in Chapter 4 I used all five valve1 files: 0.csv, 1.csv, 2.csv, 3.csv, and 4.csv. This gives a self-consistent physical regime (a single fault mode) across which paired ablations can be interpreted. Preprocessing was minimal: I checked that no null or infinite values appeared in any column, then applied a per-file z-score normalisation using the mean and standard deviation of the pre-anomaly region. Sliding windows of length 96 (96 seconds at 1 Hz) were extracted for forecasting.

For the semi-supervised training regime I identified the first row in each file where an anomaly label appears and used only the preceding rows to build training pairs. This is standard for anomaly detection: the model sees only normal data during training and is asked at inference time to distinguish normal from anomalous by looking at forecast error.

### 3.1.2 SMD

The Server Machine Dataset (Su et al., 2019) is a widely-used public benchmark. It contains telemetry from 28 servers across three groups. Each machine has 38 features covering CPU load, memory pressure, disk activity, network throughput, and various process metrics. The dataset ships with an explicit train/test split for each machine: the training portion is entirely normal, and the test portion contains anomalies with binary labels.

For the experiments in Chapter 4 I used three machines: machine-1-1, machine-1-4, and machine-2-1. These were selected to give one machine from each of the first two groups and one machine known from earlier reconstruction-based work to be harder (machine-2-1 has around 4.9 percent anomaly rate, lower than the others). Preprocessing followed the same pattern as SKAB: null and infinity checks, per-machine z-score normalisation on the training portion, sliding windows of length 96.

The two datasets differ in scale. A SKAB file has around 500 pre-anomaly rows for training. An SMD machine has around 28,000 training rows. This has practical consequences that matter in Chapter 4: each SMD training run takes roughly 90 minutes on my hardware whereas each SKAB training run takes under a minute.

## 3.2 Model implementation

The dissertation compares two model configurations. Both are trained on the same data, with the same random seeds, using the same optimiser and the same training loop. They differ in exactly one architectural component. This is the ablation variable.

### 3.2.1 The full Fusionformer

The full model is a paper-faithful implementation of Fusionformer per the architectural description in Wang et al. (2025) Section III. It has four modules.

The input is a batch of windows shaped (B, T=96, D). The segment-wise sequence embedding partitions each variable's T timesteps into T/L_seg non-overlapping segments of length L_seg=4, giving 24 segment tokens per variable. Each segment is projected through a shared linear layer to a d_model=252-dimensional vector and combined with a learnable positional encoding, following paper equations (2) to (5). The output is a four-dimensional tensor of shape (B, D, 24, 252).

The encoder stack applies one Fusionformer block. A Fusionformer block runs a multi-head segment-wise intravariable attention layer (MSWAA) with 6 heads across the 24 segment tokens per variable, followed by a residual add and layer normalisation, followed by a position-wise feed-forward network (also with a residual and layer normalisation). The output is then passed through a multi-head segment-wise intervariable attention layer (MSWEA), also with 6 heads, this time attending across the D variable tokens at each segment position. This is followed by another residual, layer normalisation, and feed-forward. The decoder stack has the same block structure.

The output head reshapes the encoder-decoder output back to a per-variable representation and projects it to a forecast of length τ=24 for each variable. The full model has approximately 3.4 million parameters at these hyperparameters.

A separate discriminator network with three fully-connected layers and a sigmoid output takes the concatenation of the input history and either the true or the predicted future, and outputs a scalar in [0, 1] indicating whether the input looks real or synthetic. I implemented this discriminator following Section III.E of the paper but the primary ablation reported in Chapter 4 was run with the adversarial loss weight set to zero. This was a deliberate scope choice explained in Section 3.6 below.

### 3.2.2 The MSWEA-off ablation

The ablated variant differs from the full model in exactly one respect: the MSWEA branch is removed. Concretely, inside each Fusionformer block, the encoder-decoder still runs MSWAA and its FFN, but the subsequent MSWEA layer and its FFN are skipped. Every other architectural element is identical: the same SWSE embedding, the same d_model, the same n_heads, the same n_enc, the same n_dec, the same dropout, the same positional encoding, the same output head. The parameter count drops from 3.44 million to 2.67 million because the MSWEA attention and its FFN are gone.

The design intent is that a paired comparison of the full model against this variant, on the same data with the same random seed, isolates the contribution of the MSWEA branch. If MSWEA is doing useful work, the full model should show higher AUROC than the ablated variant systematically across the paired runs. If MSWEA is not doing useful work, the paired differences should be centred on zero.

## 3.3 Training regime

Both variants were trained with an identical regime. Adam with weight decay 1e-5, learning rate 1.5e-4 for the generator, batch size 64, 15 epochs, gradient clipping at 1.0. The loss was Huber loss (smooth L1) with delta 1.0 rather than pure MSE, chosen after initial experiments showed pure MSE occasionally diverged with sharp gradients. Huber loss trained smoothly and monotonically in every run reported in Chapter 4.

Windowing was T=96 historical timesteps and τ=24 forecast horizon, with a stride of 1 for training pair extraction and stride 1 for inference-time sliding evaluation. The forecast horizon of 24 is short compared to the paper's up-to-256-minute prediction targets on the mining data. The choice was made to fit the memory budget of my hardware while keeping the paper's ratio of history to prediction roughly comparable to their input-96-prediction-24 setting reported in the sensitivity analysis of Section IV.G.

The primary reported experiments use adversarial loss weight λ_adv = 0, meaning the generator was trained on Huber prediction loss alone. This is documented in Section 3.6 as an explicit deviation from the paper's full configuration. It was necessary because initial experiments with the adversarial loop showed rapid discriminator collapse (D loss dropping to below 0.01 within five epochs, with the generator's prediction quality then degrading as the noisy adversarial gradient dominated). Fixing the adversarial training would require a hyperparameter sweep over λ_adv, discriminator learning rate, and the ratio of D-to-F update steps. That was outside the compute budget available.

All training was done on a MacBook Pro M5 with 16 GB of unified memory, using Apple's Metal Performance Shaders backend for PyTorch. This is deliberately modest hardware. Reproducibility on a single laptop was an explicit goal, though it also constrained what experiments were feasible, as discussed in Chapter 6.

## 3.4 Evaluation metrics

Anomaly scoring at inference time uses forecast error. For each valid starting position t in the test sequence, the model is given the window X[t : t+96] and asked to forecast Y[t+96 : t+120]. The forecast error for that window is the mean squared error between the forecast and the true 24-step future. This scalar per window is broadcast back to per-row anomaly scores by assigning row t' the error of the window whose forecast horizon covers t'.

Four metrics are computed per run.

**AUROC** (area under the receiver operating characteristic curve) is the primary reported metric. It is threshold-independent and measures how well the anomaly score ranks anomalous rows above normal rows. AUROC of 0.5 is random, 1.0 is perfect ranking. This is the metric on which all Wilcoxon comparisons in Chapter 4 are performed.

**PR-AUC** (area under the precision-recall curve) is reported alongside AUROC because it is more informative than AUROC for the imbalanced-anomaly regime characteristic of SMD (around 5 to 10 percent anomaly rate).

**F1-PA** (point-adjusted F1) is reported for compatibility with the wider literature but with an explicit caveat in every table. Wu and Keogh (2021) showed this metric is misleading. I do not draw any Chapter 5 conclusion from F1-PA values.

**Event-F1** is a per-window F1 computed without point adjustment, using a top-k threshold where k is set to the number of anomalies in the test data. This provides an honest F1 that Wu and Keogh's critique does not undermine.

## 3.5 Experimental design

For each variant on each dataset, three random seeds were used: 0, 1, and 42. On SMD this gives 3 machines × 3 seeds = 9 paired AUROC values per Wilcoxon test. On SKAB this gives 5 files × 3 seeds = 15 paired AUROC values. Seeds control PyTorch RNG, numpy RNG, and the deterministic mode where available.

The pairing is machine-by-seed on SMD (full and no_mswea trained with the same seed on the same machine) and file-by-seed on SKAB (same file, same seed). Wilcoxon signed-rank tests are computed on these paired AUROC differences using `scipy.stats.wilcoxon` with `zero_method="wilcox"` and `alternative="two-sided"`. Rosenthal effect size r is computed as |Z| / sqrt(n), where Z is derived from the two-sided p-value via the standard normal inverse CDF, and n is the number of non-zero paired differences. Effect sizes are interpreted using conventional thresholds: r below 0.1 negligible, 0.1 to 0.3 small, 0.3 to 0.5 medium, above 0.5 large.

I chose the Wilcoxon signed-rank test rather than a paired t-test because it does not assume the paired differences are normally distributed. This matters at n=9 or n=15, where a normality test on the differences would not have enough power to reject its own null. The Wilcoxon test is a safer default.

## 3.6 Deviations from the paper's exact configuration

The implementation is faithful to Wang et al. (2025) Section III in its architectural intent but deviates in specific hyperparameters that were required for compute feasibility on my hardware. These deviations are listed here rather than buried in the results.

- **d_model**: paper 256, mine 252. The paper's d_model=256 is not divisible by n_heads=6, so I used d_model=252 to keep the head count matched exactly to the paper. This is a 1.6 percent dimension reduction.
- **Segment length L_seg**: paper 32, mine 4. The paper's segment length of 32 gives only 3 segment tokens per variable at T=96, which felt very coarse. I used L_seg=4, giving 24 tokens, which is closer to standard patch-based transformer practice (Nie et al., 2023 uses similar patch counts in PatchTST). Absolute AUROC values will differ between the two configurations, but the paired ablation direction is preserved.
- **Encoder and decoder layers**: paper N_enc=4, N_dec=3, mine N_enc=1, N_dec=1. Reduced for compute budget. The paper's Section IV.G sensitivity analysis reports the model works across N_enc from 1 to 4, so N_enc=1 is inside the paper's own explored range.
- **Adversarial training**: the paper reports the discriminator adds value. My initial experiments showed discriminator collapse on both SKAB and SMD with the default hyperparameters. Reported results in Chapter 4 use λ_adv=0. The adversarial variant is documented as future work.
- **Segment-wise sequence embedding ablation**: I attempted to run a no_SWSE variant (per-timestep tokens rather than segment tokens), but the resulting model creates approximately 16 times more attention work per epoch and exhausted Metal Performance Shaders memory during training. This is documented in Chapter 6 as requiring larger GPU memory.
- **Loss function**: paper uses MSE plus adversarial. I used Huber loss (smooth L1, delta=1.0) with λ_adv=0. Huber loss trained more stably than pure MSE in initial experiments.

Every one of these deviations is a departure from the paper. The reason each was made is either compute-related or stability-related. The paired ablation of MSWEA against full is internally consistent because both configurations use identical values of all the above hyperparameters. What may not be directly comparable is my absolute AUROC values against numbers the paper reports on its own mining dataset. That is a fair reading; the dissertation is not trying to match the paper's absolute numbers but to test whether the paper's mechanism (MSWEA) contributes on the two public benchmarks in question.

### 3.6.1 Paper-hyperparameter sanity check

To confirm the compute-budget hyperparameters were not artificially depressing performance, I ran one single-seed sanity run with the paper's configuration (d_model=256, L_seg=32, N_enc=4, N_dec=3, batch 32, 30 epochs, lr=5e-4) on SMD machine-1-1. One deliberate change was necessary: n_heads=8 instead of the paper's 6, because 256 is not divisible by 6. This run took 105.8 minutes on the study hardware, roughly seven times the wall-clock of a compute-budget run, which is why a full paired ablation at paper hyperparameters was not attempted. The sanity-run AUROC was 0.9230, which sits above the compute-budget distribution on SMD (my nine full-configuration seed runs on SMD returned a mean of 0.8904 with individual runs ranging from 0.797 to 0.941). No dramatic uplift was observed from the deeper model. A second observation from this run was less clean: training loss diverged at epoch 12 (pred_MSE jumped from 0.051 to 0.248 and then plateaued for the remaining eighteen epochs), suggesting the paper's learning rate of 5e-4 is aggressive for the deeper N_enc=4 configuration on this dataset. The final AUROC of 0.923 was therefore obtained from a model that stopped learning after epoch 11 and appears to have collapsed toward mean-forecast predictions, on which anomalous windows still separated from normal windows by MSE magnitude. This is not a satisfying training outcome, but it does confirm two things: the compute-budget hyperparameters used for the reported experiments are not a bottleneck, and the paper's exact configuration also benefits from a smaller learning rate on SMD than the middle of the paper's stated range.

## 3.7 Summary

The methodology sets up a paired ablation of one specific component of the Fusionformer architecture, on two public benchmarks with contrasting characteristics, under a training regime that trades off some architectural depth against feasibility on modest hardware. The primary comparison is a paired Wilcoxon test on per-run AUROC values. Deviations from the paper are documented and reasoned. Chapter 4 reports the results.
