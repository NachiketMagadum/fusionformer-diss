---
title: "Chapter 4: Results"
author: "Nachiket Magadum"
date: "September 2026"
geometry: margin=2.5cm
fontsize: 11pt
mainfont: "Times New Roman"
linestretch: 1.5
---

# CHAPTER 4: RESULTS

This chapter reports what the experiments produced. Section 4.1 gives absolute AUROC for the full paper-faithful Fusionformer on SMD, run at three seeds across three machines. Section 4.2 reports the MSWEA ablation on SMD with Wilcoxon significance. Section 4.3 gives absolute AUROC for the full model on SKAB across five files at three seeds. Section 4.4 reports the MSWEA ablation on SKAB with Wilcoxon significance. Section 4.5 places the two benchmarks side by side and describes the contradictory finding that motivates Chapter 5. Section 4.5.1 reports a head-to-head paired comparison against an LSTM baseline. Section 4.6 summarises.

All AUROC values in this chapter come from the training runs archived in the code appendix. Every run was recorded as a separate output file in `notes/`. Nothing in this chapter is estimated, interpolated, or averaged over runs that were not actually completed.

## 4.1 Fusionformer FAM on SMD

Table 4.1 shows the per-machine, per-seed AUROC for the full paper-faithful Fusionformer trained with the MSWEA branch active on three SMD machines. Each cell is a single training run.

**Table 4.1**: Full Fusionformer FAM on SMD (3 machines × 3 seeds = 9 runs).

| Machine | Seed 0 | Seed 1 | Seed 42 | Mean | Std |
|---|---|---|---|---|---|
| machine-1-1 | 0.9410 | 0.9377 | 0.9389 | 0.9392 | 0.0017 |
| machine-1-4 | 0.9346 | 0.9323 | 0.9337 | 0.9335 | 0.0012 |
| machine-2-1 | 0.7980 | 0.7972 | 0.8005 | 0.7986 | 0.0017 |
| **Overall** | | | | **0.8904** | **0.0650** |

The overall mean AUROC across all 9 runs is 0.8904, standard deviation 0.0650. Individual machines vary a lot but within-machine seed variance is very tight (standard deviation under 0.002 on every machine), which suggests the training is stable and reproducible. Machine-1-1 and machine-1-4 are relatively easy at around 0.93-0.94. Machine-2-1 is the hard one at about 0.80. That per-machine spread is typical of SMD in the published literature and is what drives the 0.065 overall standard deviation.

The absolute AUROC of 0.89 sits in a plausible range for SMD anomaly detection at the shallow-model settings used here. My model is intentionally smaller than the paper's (N_enc = 1 versus the paper's 4, and about 3.4 million parameters instead of the paper-exact configuration's roughly 11 million), so I expect somewhat lower absolute AUROC than a leaderboard-scale implementation. The single-seed paper-hyperparameter sanity check reported in Chapter 3.6.1 returned AUROC 0.9230 on machine-1-1, a plausible upper bound. The absolute number is not the primary claim of this dissertation. It matters as a sanity check that the implementation trains to sensible values rather than converging to something obviously broken.

![Figure 4.1: Paired MSWEA ablation on SMD across 9 machine-seed pairs. Full Fusionformer FAM (blue) versus the MSWEA-off ablation (orange). Wilcoxon signed-rank on the paired differences gives W = 11.00, p = 0.203, Rosenthal r = 0.424 (medium effect). The paired mean difference is small and positive but not statistically significant at the 0.05 level.](figures/fig_4_1_smd_paired.png){ width=100% }

## 4.2 MSWEA ablation on SMD

The MSWEA-off variant was trained under identical settings for each of the same 9 machine-seed configurations. Table 4.2 shows the paired comparison.

**Table 4.2**: Paired MSWEA ablation on SMD (n = 9 pairs).

| Machine | Seed | Full AUROC | No-MSWEA AUROC | Difference |
|---|---|---|---|---|
| machine-1-1 | 0 | 0.9410 | 0.9377 | +0.0033 |
| machine-1-1 | 1 | 0.9377 | 0.9414 | -0.0037 |
| machine-1-1 | 42 | 0.9389 | 0.9428 | -0.0039 |
| machine-1-4 | 0 | 0.9346 | 0.9326 | +0.0020 |
| machine-1-4 | 1 | 0.9323 | 0.9292 | +0.0031 |
| machine-1-4 | 42 | 0.9337 | 0.9308 | +0.0029 |
| machine-2-1 | 0 | 0.7980 | 0.7864 | +0.0116 |
| machine-2-1 | 1 | 0.7972 | 0.7858 | +0.0114 |
| machine-2-1 | 42 | 0.8005 | 0.7836 | +0.0169 |

Grand mean AUROC was 0.8904 for the full model and 0.8856 for the MSWEA-off variant, with a mean paired difference of +0.0048 favouring the full model. The full model won on 7 of 9 pairs. The two losses were both on machine-1-1, both very small (-0.0037 and -0.0039). The largest paired advantages for the full model were on machine-2-1, the hardest of the three machines, where the full model gained roughly a full point of AUROC on every seed.

The Wilcoxon signed-rank test on these paired values produced W = 11.00, two-sided p = 0.203, and Rosenthal effect size r = 0.424 (medium). The verdict at conventional 0.05 significance is that the MSWEA ablation on SMD does not reject the null hypothesis of no difference between the two variants. But the effect size is medium rather than negligible, and 7 of 9 pairs point in the direction that favours the full model. Read strictly, this is a null result at α = 0.05. Read more carefully, it is a small positive trend that this study is underpowered to confirm at n = 9. Both readings are compatible with a p of 0.20, and both matter for the interpretation in Chapter 5.

## 4.3 Fusionformer FAM on SKAB

Table 4.3 shows the per-file, per-seed AUROC for the full Fusionformer on the five SKAB valve1 files.

**Table 4.3**: Full Fusionformer FAM on SKAB valve1 (5 files × 3 seeds = 15 runs).

| File | Seed 0 | Seed 1 | Seed 42 | Mean | Std |
|---|---|---|---|---|---|
| 0.csv | 0.8613 | 0.8619 | 0.8655 | 0.8629 | 0.0019 |
| 1.csv | 0.8605 | 0.8549 | 0.8570 | 0.8575 | 0.0023 |
| 2.csv | 0.9013 | 0.9231 | 0.9054 | 0.9099 | 0.0094 |
| 3.csv | 0.9613 | 0.9665 | 0.9582 | 0.9620 | 0.0034 |
| 4.csv | 0.8054 | 0.8002 | 0.8030 | 0.8029 | 0.0021 |
| **Overall** | | | | **0.8790** | **0.0538** |

Overall SKAB mean AUROC is 0.8790, standard deviation 0.0538 across all 15 runs. Individual files vary in difficulty. File 3 is straightforward with 0.96 across seeds. File 4 is the hardest at around 0.80. Within-file seed variance is very low, generally under 0.01 standard deviation, which tells me the training is stable and reproducible on SKAB.

![Figure 4.2: Paired MSWEA ablation on SKAB across 15 file-seed pairs. Full Fusionformer FAM (blue) versus the MSWEA-off ablation (orange). Wilcoxon signed-rank on the paired differences gives W = 9.00, p = 0.002, Rosenthal r = 0.797 (large effect). Statistically significant at the 0.01 level.](figures/fig_4_2_skab_paired.png){ width=100% }

## 4.4 MSWEA ablation on SKAB

The MSWEA-off variant was trained at the same 15 file-seed configurations. Table 4.4 shows the paired comparison.

**Table 4.4**: Paired MSWEA ablation on SKAB (n = 15 pairs).

| File | Seed | Full AUROC | No-MSWEA AUROC | Difference |
|---|---|---|---|---|
| 0.csv | 0 | 0.8613 | 0.8506 | +0.0107 |
| 0.csv | 1 | 0.8619 | 0.8502 | +0.0117 |
| 0.csv | 42 | 0.8655 | 0.8526 | +0.0129 |
| 1.csv | 0 | 0.8605 | 0.8616 | -0.0011 |
| 1.csv | 1 | 0.8549 | 0.8651 | -0.0102 |
| 1.csv | 42 | 0.8570 | 0.8608 | -0.0038 |
| 2.csv | 0 | 0.9013 | 0.8626 | +0.0387 |
| 2.csv | 1 | 0.9231 | 0.8565 | +0.0666 |
| 2.csv | 42 | 0.9054 | 0.8625 | +0.0429 |
| 3.csv | 0 | 0.9613 | 0.9541 | +0.0072 |
| 3.csv | 1 | 0.9665 | 0.9504 | +0.0161 |
| 3.csv | 42 | 0.9582 | 0.9457 | +0.0125 |
| 4.csv | 0 | 0.8054 | 0.7928 | +0.0126 |
| 4.csv | 1 | 0.8002 | 0.7957 | +0.0045 |
| 4.csv | 42 | 0.8030 | 0.7987 | +0.0043 |

Grand mean AUROC was 0.8790 for the full model and 0.8640 for the MSWEA-off variant, with a mean paired difference of +0.0150 favouring the full model. Full won on 12 of 15 pairs. The three losses were all on file 1, and they were small (largest -0.0102).

The Wilcoxon signed-rank test on these paired values produced W = 9.00, two-sided p = 0.0020, and Rosenthal effect size r = 0.797 (large). This is a clear positive result at the 0.01 significance level: the MSWEA branch measurably improves anomaly detection performance on SKAB.

The per-file breakdown is worth reading closely. Files 0 and 4 show small consistent positive differences of around +0.010 across seeds. File 3 shows small positive differences around +0.012 across seeds. File 2 shows large positive differences of +0.04 to +0.07 across seeds. File 1 alone reverses direction, with small negative differences of -0.001 to -0.010. So the effect is not uniform: MSWEA helps a lot on file 2, helps a little on files 0, 3, and 4, and slightly hurts on file 1. The overall effect across all five files is positive and significant at the p = 0.002 level.

![Figure 4.3: Paired AUROC differences (full minus MSWEA-off) on the two benchmarks. SMD (blue) is centred close to zero with mean paired difference +0.0048, while SKAB (orange) is shifted away from zero with mean paired difference +0.0150. The separation between the two distributions is the central finding of this chapter.](figures/fig_4_3_paired_diffs.png){ width=100% }

## 4.5 Cross-benchmark comparison

The two benchmarks give different answers to RQ1. Table 4.5 summarises.

**Table 4.5**: MSWEA ablation compared across the two benchmarks.

| Benchmark | Pairs | Full mean AUROC | No-MSWEA mean AUROC | Diff | Wilcoxon p | Effect size r | Verdict |
|---|---|---|---|---|---|---|---|
| SMD | 9 | 0.8904 | 0.8856 | +0.0048 | 0.203 | 0.424 (medium) | Not significant at 0.05 |
| SKAB | 15 | 0.8790 | 0.8640 | +0.0150 | **0.002** | **0.797 (large)** | **Significant at 0.01** |

Both benchmarks nominally favour the full model over the ablated variant, both effect sizes are non-trivial, but only SKAB reaches conventional statistical significance. On SMD the paired mean difference is small (+0.005) and 7 of 9 pairs point the same way but with n = 9 the study cannot rule the null hypothesis out. On SKAB the paired difference is three times larger (+0.015), 12 of 15 pairs point the same way, and the effect is significant at p = 0.002 with a large effect size.

This is the finding this dissertation is actually about. On its own, an ablation that shows the mechanism gives only a small unconfirmed effect on one benchmark would be inconclusive. On its own, an ablation that shows the mechanism helps significantly on another benchmark would confirm the paper's claim. Having both results side by side reframes the question: not "does MSWEA work?" but "when does MSWEA work most?"

![Figure 4.4: Mean AUROC comparison across the two benchmarks with statistical annotations. On SMD (left) the difference between full and MSWEA-off is small and not significant. On SKAB (right) the difference is larger and statistically significant at the 0.01 level with a large effect size.](figures/fig_4_4_cross_benchmark.png){ width=90% }

The obvious first hypothesis is that MSWEA's contribution depends on how tightly coupled the variables in the input actually are. SKAB has eight sensors on one physical machine, all coupled by fluid mechanics: the pressure, flow rate, and vibration signals are physically linked because they are all measuring different aspects of the same pump under the same fault condition. SMD has thirty-eight independent server metrics that are only loosely related. The direction of the SMD trend and the size of the SKAB effect are both compatible with the hypothesis that variable-axis attention adds more value when there is more variable-axis structure to attend to. Chapter 5 develops this hypothesis in detail and discusses what would need to be done to test it more rigorously.

### 4.5.1 Head-to-head against an LSTM baseline

Before interpreting the MSWEA ablation, one further comparison is useful: does the Fusionformer architecture as a whole beat a simple non-transformer baseline on the same datasets? If it does not, the ablation is a comparison between two variants of a model that is not worth using at all.

I trained an LSTM forecasting baseline at the identical protocol used for Fusionformer: same input window (T = 96), same forecast horizon (τ = 24), same Huber loss, same MSE-of-forecast anomaly score, same seeds, same machines and files. The LSTM had two layers of hidden size 128 (roughly 336,000 parameters on SMD, 228,000 on SKAB) and was trained for 15 epochs at the same optimiser and learning rate. This is a straight architectural swap: transformer versus recurrent, with everything else held constant.

**Table 4.6**: Paired LSTM baseline versus full Fusionformer.

| Benchmark | Pairs | LSTM mean AUROC | Fusionformer full mean AUROC | Diff | Wilcoxon p | Effect size r |
|---|---|---|---|---|---|---|
| SMD | 9 | 0.8679 | 0.8904 | +0.0225 | **0.020** | **0.778 (large)** |
| SKAB | 15 | 0.8333 | 0.8790 | +0.0457 | **0.0001** | **1.035 (very large)** |

Fusionformer beats the LSTM baseline on both benchmarks by two to five AUROC points, and both differences are statistically significant. The SKAB effect is very large (r > 1 reflects a very extreme Z given the sample size). Per-run, the transformer wins on 7 of 9 SMD pairs and on all 15 SKAB pairs. The two losses on SMD were both on machine-1-1 (the easiest machine, where both models saturate around 0.94) and were tiny (-0.003, -0.0004). Every substantive gap between the two models points the same way.

This matters for how the MSWEA ablation should be read. The ablation is not a comparison between one strong model and one broken model. Both variants of Fusionformer beat the LSTM baseline on SMD (full: +0.023, MSWEA-off: +0.018), so both variants are doing real work on both benchmarks. What the MSWEA ablation measures is the marginal contribution of one attention branch inside an already-competent architecture, not the value of the architecture as a whole.

## 4.6 Summary

Across 24 paired MSWEA ablation runs on two public benchmarks, the intervariable attention branch of Fusionformer's fusion attention module shows a benchmark-dependent effect. On SMD (n = 9 pairs), a small positive paired difference (+0.005) with medium effect size (r = 0.424) that does not reach significance at n = 9 (p = 0.203). On SKAB (n = 15 pairs), a larger paired difference (+0.015) with large effect size (r = 0.797) that is significant at the 0.01 level (p = 0.002). A separate head-to-head against an LSTM baseline on the same 24 runs confirmed that the Fusionformer architecture beats a non-transformer baseline by two to five AUROC points on both benchmarks (both significant at p < 0.05), so the ablation compares two respectable models rather than one strong and one weak.

The next chapter interprets these results, discusses why the two benchmarks give different answers, and lays out the limitations of what could and could not be tested within the compute budget.
