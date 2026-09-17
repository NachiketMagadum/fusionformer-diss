---
title: "Chapter 6: Conclusion"
author: "Nachiket Magadum"
date: "September 2026"
geometry: margin=2.5cm
fontsize: 11pt
mainfont: "Times New Roman"
linestretch: 1.5
---

# CHAPTER 6: CONCLUSION

This chapter closes the dissertation. Section 6.1 gives a chapter-by-chapter summary. Section 6.2 sets out the specific research contributions. Section 6.3 lists the future research and development that would strengthen or extend what is here. Section 6.4 is a personal reflection on what I learned, what I would do differently, and what I got wrong.

## 6.1 Summary of the dissertation

**Chapter 1** framed the problem. Fusionformer (Wang et al., 2025) is a recent transformer architecture proposed for multivariate time series forecasting and anomaly detection, evaluated in its original paper on proprietary open-pit mine data. The paper introduces three architectural components (SWSE, FAM, adversarial training) but does not test them individually or on public benchmarks. The dissertation set out to reproduce the model faithfully and to isolate whether the fusion attention module actually contributes to anomaly detection on two contrasting public benchmarks.

**Chapter 2** reviewed the literature. Anomaly detection in multivariate time series moved from statistical methods (Isolation Forest, LOF) through recurrent architectures (LSTM autoencoders) to transformers. Within the transformer family the current design questions concern how to embed continuous timesteps, how to attend across time and variables, and how to compose a loss function that produces useful anomaly scores. Wu and Keogh (2021) provide the important critique that shaped my evaluation choices: point-adjusted F1 is systematically misleading, and threshold-independent metrics like AUROC are more trustworthy.

**Chapter 3** described the two datasets (SKAB and SMD), the paper-faithful Fusionformer implementation and the single-component ablation variant that differs from the full model only in whether the MSWEA branch is present, the training regime, the four-metric evaluation harness, and the paired multi-seed statistical protocol. It ended with an explicit table of every deviation from the paper's exact configuration, with a reason for each.

**Chapter 4** reported the results. Nine paired SMD runs (three machines, three seeds each) gave a mean full-model AUROC of 0.8904 and a mean MSWEA-off AUROC of 0.8856, with Wilcoxon p = 0.203, Rosenthal r = 0.424 (medium). Fifteen paired SKAB runs (five files, three seeds each) gave a mean full-model AUROC of 0.8790 and a mean MSWEA-off AUROC of 0.8640, with Wilcoxon p = 0.002, Rosenthal r = 0.797 (large). The two benchmarks give different answers on whether MSWEA improves anomaly detection: a small inconclusive positive on SMD and a large significant positive on SKAB. A separate paired head-to-head against an LSTM baseline confirmed that the Fusionformer architecture as a whole beats a non-transformer forecaster on both benchmarks (SMD: +0.023 AUROC, p = 0.020; SKAB: +0.046 AUROC, p = 0.0001).

**Chapter 5** interpreted this. The working hypothesis, given two data points, is that MSWEA is useful when the input variables are physically coupled (as in SKAB's eight sensors on a single pump) and not useful when they are relatively independent (as in SMD's thirty-eight server metrics). This is consistent with the Fusionformer paper's success on its own mining data, which presumably falls in the coupled regime. The chapter is explicit that this is a mechanism proposed on two data points, not proven, and that both the SWSE and adversarial components of the paper could not be tested within the compute budget of the study hardware.

## 6.2 Research contributions

Five specific contributions come out of this dissertation.

**A paper-faithful open-source reproduction of Fusionformer.** The paper does not release code. My implementation, published in the accompanying Jupyter notebook and modular Python codebase, is the first public implementation of the architecture that I am aware of. Someone else can now run the model, modify the components, and extend the ablation without starting from the paper's Section III alone.

**A paired MSWEA ablation on two public benchmarks with statistical rigour.** Twenty-four training runs of the paper-faithful model split across two datasets, each ablation compared under a paired multi-seed protocol with Wilcoxon signed-rank testing and Rosenthal effect size reporting. Every per-run AUROC is archived in the code appendix. The result is a benchmark-dependent finding on the paper's headline mechanism.

**A paired non-transformer baseline comparison on the same protocol.** Twenty-four LSTM forecasting runs trained on the same T = 96, τ = 24, seed protocol as the Fusionformer runs, compared pair-by-pair. Fusionformer beats the LSTM baseline by two to five AUROC points on both benchmarks (SMD p = 0.020, SKAB p = 0.0001). The MSWEA ablation is therefore a comparison between two variants of an architecture that beats an obvious non-transformer alternative, not between one good model and one broken one.

**A specific claim about when MSWEA earns its parameters.** The dissertation argues, on the basis of two data points and a mechanism, that MSWEA is worth its parameters on tightly coupled sensor data and not on loosely coupled telemetry. This is a narrower claim than the paper makes and, if it holds up in future work, a more practically useful one. A practitioner deciding whether to include MSWEA in a new pipeline can ask a specific question about their data (are the variables physically or causally coupled?) rather than treating FAM as always-on or always-off.

**A methodological template for reproduction studies.** The combination of multi-seed evaluation, paired Wilcoxon significance testing, Rosenthal effect size reporting, and the Wu-and-Keogh-aligned four-metric harness is not specific to Fusionformer or to anomaly detection. It is a template that generalises to any ablation study where the question is whether a specific component contributes rather than whether an overall architecture wins on average. I hope it is useful to other students doing similar reproductions.

## 6.3 Future research and development

Five directions extend this work meaningfully.

**Complete the three-component ablation on cloud compute.** The SWSE and adversarial ablations were not testable on my hardware. Both would run in reasonable wall-clock time on a 40 GB A100 or similar. I have prepared the deployment scripts and both experiments could be completed in around eighteen hours of GPU time. This is the immediate next step and it directly closes the gap between the DDR proposal and what the dissertation was able to deliver.

**A third benchmark to test the coupling hypothesis directly, on GPU.** MSL (Hundman et al., 2018) has fifty-five channels of Mars Science Laboratory rover telemetry with intermediate physical coupling and is the natural third benchmark. I attempted an MSL sweep on the same MacBook M5 hardware in the final week of the project. Two full-variant runs completed on channel M-1 (seeds 0 and 1) at wall-clock times of 518 and roughly 500 minutes each, and both returned test AUROC very close to chance (0.532 and comparable), which is consistent with the shallow compute-budget configuration failing to converge on MSL's fifty-five-feature, fifty-percent-anomaly-rate structure. The sweep was killed before completing the remaining twenty-five runs because the projected wall-clock exceeded the submission window. If MSWEA on MSL, run at the paper's exact configuration on a cloud GPU, comes out with an effect between SKAB's (large, significant) and SMD's (small, inconclusive), the coupling hypothesis gains support. If it does not, an alternative mechanism (dataset size, anomaly rate, feature dimensionality) needs investigation. Approximately ten hours of GPU compute on a single A100 would deliver a full paired ablation on three MSL channels, and the loader, training script, and launcher scripts are already in place in the code repository (`msl_loader.py`, `train_ff_forecast_and_score_anomaly.py --dataset msl`, `run_msl_sweep.sh`).

**A synthetic benchmark disentangling coupling from size.** Simulate multivariate time series with tunable coupling strength and tunable training set size, then run the same MSWEA ablation across a grid of both. This is the definitive way to answer whether the effect I observe is about coupling per se or about a coupling-size interaction. It is more work but it produces a claim that generalises beyond the two datasets I happened to use.

**Extending the SMD study to more machines.** My SMD ablation used three of the twenty-eight available machines, giving n = 9 paired comparisons. Increasing to ten or fifteen machines would take the sample size to thirty or forty-five pairs, which has enough statistical power to detect small effects if they exist. The current SMD null could then be read confidently as "no effect" rather than "possibly a small effect the study is underpowered to detect."

**Edge deployment of the trained FAM model.** The DDR proposed an EdgeFusion pathway that I did not deliver. The FAM model at my configuration has 3.44 million parameters, which is a plausible size for embedded deployment after 8-bit quantisation. A concrete edge deployment study would benchmark inference latency on Raspberry Pi 5 or NVIDIA Jetson Nano, comparing quantised FAM to a distilled student model. That is a natural continuation project.

## 6.4 Personal reflections

Writing this section is uncomfortable but the Brunel template asks for it and I want to answer honestly.

The most valuable thing I learned in this project was the discipline of paired multi-seed evaluation combined with statistical significance testing. Before I started, I would have looked at a mean AUROC difference of 0.02 between two models and treated it as evidence that one was better. I now understand why that reflex is misleading: the seed-to-seed variance on a single model can easily exceed 0.02 on some machines. The Wilcoxon test with an explicit effect size forced me to accept an unambiguous null on SMD and to see that a mean paired improvement of 0.015 on SKAB was actually a large effect only because it was systematic across pairs. Being able to sit with a null result rather than trying to rescue it with a story is a skill I did not have when I started.

The biggest mistake I made was building the wrong model for the first two months of the project. My initial implementation was reconstruction-based, which felt natural given how much of the anomaly detection literature is reconstruction-based, and I did not read the Fusionformer paper carefully enough at the start to notice that Fusionformer is a forecasting model. I found this in early September while re-reading the paper for proofreading. Rebuilding the paper-faithful implementation cost me about a week of compute and a lot of discarded prior work. In hindsight, the fix is obvious: read the target paper's method section three times before writing a single line of code, not once at the beginning and once at the end. I have carried this into how I read papers now.

The other thing I underestimated was how much of a dissertation is writing rather than experiments. My internal budget going in was maybe two weeks of writing after the experiments were done. In practice the writing has taken longer than the experiments once the pivot to the paper-faithful model happened, and I have had to rewrite three of the six chapters twice. Next time I would allocate writing and experimental time as equal budgets from the start.

Honesty about limitations was another learned discipline. My earlier drafts of the discussion chapter tried to argue past the SWSE and adversarial gaps rather than state them plainly. My supervisor Professor Xiaohui Liu, whose paper this dissertation reproduces, made clear through his willingness to encourage the study that the honest reporting of what could and could not be tested was more valuable than a completed-looking dissertation that hid its gaps. That was a research-culture lesson I did not know I needed and I am grateful for it.

Finally, I would say I underestimated the value of picking a supervisor whose own paper I could reproduce and ablate. This dissertation would have been much less interesting if I had picked a topic where the supervisor had no direct stake in the finding. The genuine intellectual friction of testing a mechanism whose author was one office over made me more careful about the evidence than I would have been otherwise.

What I would do differently on a similar project: read the target paper three times before coding, budget writing time equal to experimental time, run one small end-to-end experiment before scaling up to catch architecture mismatches early, and treat honest limitations as part of the contribution rather than as concessions to be minimised.
