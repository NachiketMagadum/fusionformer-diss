---
title: "Chapter 1: Introduction"
author: "Nachiket Magadum"
date: "September 2026"
geometry: margin=2.5cm
fontsize: 11pt
mainfont: "Times New Roman"
linestretch: 1.5
---

# CHAPTER 1: INTRODUCTION

A water treatment pump has eight sensors on it. Temperature, pressure, current, voltage, flow rate, two accelerometers, one thermocouple. Every second, another eight numbers. Over a shift that is around 30,000 readings from a single machine. Over a plant, millions. If a valve is closing when it should not, or a bearing is starting to fail, the signature is somewhere in those numbers before anyone notices the alarm. The question is whether an algorithm can catch it in time.

This is multivariate time series anomaly detection. Given a stream of sensor readings, flag the moments that do not look like normal operation. It sounds simple. Most of the difficulty is that "normal" is not one thing. It drifts with temperature, with load, with the age of the equipment. What looked anomalous last week might be the new baseline this week. Classical statistical rules struggle with this kind of thing, and the field has spent the last decade moving to machine learning approaches that can learn a normal-behaviour representation from data (Blázquez-García et al., 2021).

Transformers have taken over most parts of sequence modelling, and multivariate time series is no exception. Since around 2021, papers proposing transformer-based anomaly detectors have arrived at a steady rate. PatchTST (Nie et al., 2023) treats short windows as patches and forecasts them. The Anomaly Transformer (Xu et al., 2022) exploits association discrepancy between prior and series attention. TranAD (Tuli et al., 2022) uses an adversarial training loop. Each paper claims a specific mechanism does something useful. Most of the time, the claim rests on comparing the full model against a small number of baselines on one or two benchmarks.

The paper that started this dissertation is one of these. Fusionformer, published in IEEE Transactions on Neural Networks and Learning Systems in August 2025, is a forecasting model that Wang, Wang, Dong, Lauria, Liu, Wang, Fadzil and Liu propose for multivariate time series anomaly detection. The application domain is open-pit mine slope failure prediction. The idea is that if you can forecast where a slope is going, you can flag the moment when the forecast disagrees sharply with reality, and treat that as a warning of instability. The model has three parts: a segment-wise sequence embedding that turns raw timesteps into fewer, higher-level tokens; a fusion attention mechanism made of two attention branches, one across time steps and one across sensor variables; and an adversarial training loop where a discriminator learns to tell real future windows from predicted ones. On the paper's own mining dataset the combined model beats every baseline they compare it against. The dataset is proprietary.

I could not get access to that dataset. What I could do is take the three components and test whether they carry over to two public benchmarks the paper does not use: the Skoltech Anomaly Benchmark, an industrial pumping station dataset with eight sensors (Katser and Kozitsin, 2020), and the Server Machine Dataset, thirty-eight-channel telemetry from twenty-eight production servers (Su et al., 2019). These two datasets sit in different regimes. SKAB has few sensors that are all physically coupled by the mechanics of the pump. SMD has many channels that are relatively independent of one another. If a mechanism like the fusion attention module is genuinely useful for anomaly detection, it should show up on both. If it only shows up on one, that tells us something about when the mechanism actually helps.

The dissertation reports what I found. On the Server Machine Dataset the fusion attention module provides no measurable benefit over a version of the model with the intervariable attention branch removed. On the Skoltech Anomaly Benchmark the same ablation gives a statistically significant improvement with a large effect size. In other words, the paper's headline mechanism seems to work on one of my two benchmarks and not on the other. The rest of this document tries to explain why, and to be honest about what could not be tested.

## 1.1 Research aim and objectives

The overall aim is a straightforward one. I wanted to reproduce Fusionformer faithfully enough to test whether its individual components, particularly the fusion attention module, actually contribute to anomaly detection performance on multivariate time series benchmarks the paper does not itself evaluate.

Three specific objectives followed from this.

**Objective 1** is to implement the paper's architecture, meaning the segment-wise sequence embedding, the multi-head fusion attention layer with its intravariable and intervariable branches, the encoder-decoder stack, and the adversarial training loop, in PyTorch, from the paper's Section III description. The implementation should match the paper's design closely enough that any residual differences can be listed explicitly.

**Objective 2** is to run a paired ablation of the fusion attention module on both SKAB and SMD. Concretely: take the full model, remove only the intervariable attention branch, train both variants on the same data with the same random seeds and the same hyperparameters, and compare the resulting anomaly detection AUROC using the Wilcoxon signed-rank test on paired per-run values.

**Objective 3** is to place the paired ablation in a wider picture. Where a component could not be tested, I wanted the reason for that documented as a limitation rather than glossed over. Two of the paper's components fell into this category: a full segment-wise embedding ablation would have required more GPU memory than my hardware could give it, and the adversarial component I could implement but could only test at one hyperparameter setting on one benchmark before the compute budget ran out.

The research questions the dissertation actually answers, given what I could test, are:

**RQ1** — On SMD and SKAB, does the multi-head intervariable attention branch of Fusionformer's fusion attention module improve anomaly detection performance over a matched variant that removes it? If it does, how large is the effect and is it statistically significant?

**RQ2** — If the answer to RQ1 differs between benchmarks, what characteristic of the two datasets can plausibly explain the difference, and what does that imply for when the mechanism is worth using?

I initially set out to answer a third question about the segment-wise sequence embedding, but the ablation that would have been needed for it did not fit in the memory of my hardware. That question is documented as future work in Chapter 6.

## 1.2 Research approach

The approach is quantitative and empirical. Every claim in Chapter 4 is backed by a specific set of training runs whose per-run outputs are archived in the code appendix.

I did the work in four passes. First, a paper-faithful implementation of Fusionformer in PyTorch based on Section III of Wang et al. (2025). The forecasting objective, the segment-wise embedding, both branches of the fusion attention module, the discriminator. Wherever the paper's description left a design choice open, I picked what standard practice in similar patch-based transformers would suggest, and noted the choice in Chapter 3.

Second, an ablation variant with exactly one difference from the full model: the intervariable attention branch removed. Same encoder-decoder depth, same embedding dimension, same number of heads, same segment length, same training regime. If the branch is doing useful work, the difference should show up as a paired AUROC gain for the full model against the ablated one.

Third, training. Both variants trained on both benchmarks, three random seeds per configuration. On SMD this gave nine paired comparisons across three machines; on SKAB it gave fifteen across five files. All runs on a MacBook Pro M5 with 16 GB of unified memory using Apple's Metal Performance Shaders backend for PyTorch. No cloud compute was used for the reported experiments.

Fourth, statistical analysis. Paired AUROC differences were tested with the Wilcoxon signed-rank test, with the Rosenthal effect size r reported alongside the p-value. Where I have used a mean, I have also given the standard deviation and the win rate in the paired comparison. The intent is to make it obvious to a reader when a small mean difference is being carried by seed variance rather than by a real effect.

The study involves no human participants, no personal data, and no biological or physical samples. It is entirely computational. No application to Brunel's Research Ethics Committee was required. The ethics statement in Appendix A confirms this.

## 1.3 Dissertation outline

The rest of the dissertation is organised as follows.

**Chapter 2** reviews the literature on multivariate time series anomaly detection, the transformer-based methods that dominate recent work, and the important critique of evaluation practice raised by Wu and Keogh (2021). It also describes the Fusionformer paper in enough detail that the ablation choices in Chapter 3 can be read against a specific architectural target.

**Chapter 3** describes the datasets used, the paper-faithful implementation, the ablation variant, the training regime, and the four-metric evaluation harness. It ends with a table listing every deviation from the paper's exact configuration, so that a reader who has the paper in front of them can see what I changed and why.

**Chapter 4** presents the results. Baseline behaviour on both datasets, the paired Wilcoxon comparisons for the fusion attention ablation on SKAB and SMD separately, per-file and per-machine breakdowns, and a direct cross-benchmark comparison of the two ablation p-values and effect sizes.

**Chapter 5** discusses what the results mean. It argues that the benchmark-dependent effect on the intervariable attention branch is best understood in terms of how tightly coupled the variables in each dataset are physically, and it acknowledges the components I could not test.

**Chapter 6** concludes with a chapter-by-chapter summary, the specific research contributions, the future work that would strengthen the finding, and a personal reflection on what I would do differently.
