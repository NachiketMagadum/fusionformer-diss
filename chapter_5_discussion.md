---
title: "Chapter 5: Discussion"
author: "Nachiket Magadum"
date: "September 2026"
geometry: margin=2.5cm
fontsize: 11pt
mainfont: "Times New Roman"
linestretch: 1.5
---

# CHAPTER 5: DISCUSSION

This chapter interprets the results in Chapter 4 and lays out what I could not test. Section 5.1 restates the two-benchmark finding and asks what it means. Section 5.2 develops the working hypothesis: MSWEA's usefulness depends on how tightly the input variables are physically coupled. Section 5.3 tests that hypothesis against the two datasets as a sanity check. Section 5.4 sets my findings against what the Fusionformer paper itself claims. Section 5.5 is a long limitations section, because there is a lot the study did not touch. Section 5.6 draws out the practical implications for anyone building a similar model. Section 5.7 summarises.

## 5.1 A benchmark-dependent effect

Chapter 4 reported that the intervariable attention branch (MSWEA) of Fusionformer's fusion attention module gives a significantly larger benefit on SKAB (p = 0.002, r = 0.797, large effect) than on SMD (p = 0.203, r = 0.424, medium effect but not significant at n = 9). The paired mean differences point the same direction on both benchmarks (full model higher than the MSWEA-off variant on 7 of 9 SMD pairs and 12 of 15 SKAB pairs) but only the SKAB effect is large enough to be confirmed at α = 0.05. The MSWEA effect is roughly three times larger on SKAB than on SMD in absolute AUROC terms (+0.015 versus +0.005).

The natural first read of this is that MSWEA might be more useful in some settings than others, and the interesting question is which. If it worked equally everywhere, the paper's claim would generalise and my two ablations would give matching effect sizes. If it worked nowhere, both would give clean nulls. Getting one clear positive on SKAB and one weak, inconclusive positive on SMD forces the question of what is different about the two datasets that matters for this specific mechanism.

That is the question I want to try to answer in the rest of this chapter.

## 5.2 A working hypothesis: variable coupling

The mechanism MSWEA implements is straightforward. For each segment position in the time dimension, the module lets every variable's representation attend to every other variable's representation. Concretely, at each of the 24 segment positions in my configuration, D variables produce D query, key, and value vectors, and a D-by-D attention matrix decides how much each variable should weight information from every other variable at that segment. If two variables at that segment tend to move together in a way that carries information about anomaly, MSWEA can learn a high attention weight between them and use one to sharpen the reconstruction of the other. If two variables are essentially independent, the attention matrix has no useful structure to learn.

This suggests a concrete hypothesis: MSWEA should help when the variables in the input are physically or causally coupled, and should not help when the variables are more or less independent. The mechanism has something to do only if there is cross-variable structure to attend to.

The two benchmarks fall on opposite sides of this line, or so it seems to me. SKAB's eight sensors sit on one physical pumping station. When the outlet valve starts closing, the pressure signal rises, the flow rate signal falls, the current draw shifts because the motor is fighting a restriction, the vibration profile changes because the mechanical loading changes, and the temperature rises because energy is being dissipated as heat rather than moving fluid. These are not eight independent signals; they are eight views of one physical event. If any transformer mechanism can exploit cross-variable structure, this is the kind of data where it can.

SMD's thirty-eight channels are different. They come from a production server: CPU load, memory pressure, disk latencies of various kinds, network throughput, various process counters. Some of these are related. Memory pressure and disk swap activity are causally linked. But most channels are pretty loose in their coupling. CPU load and network output rate correlate weakly in aggregate but the moment-to-moment coupling is not strong. Anomalies in server telemetry often show up in a subset of channels rather than across the whole variable space. If MSWEA is looking for coherent cross-variable structure, it has less to find on SMD than on SKAB.

That is the hypothesis. On a benchmark where the variables are tightly coupled, MSWEA helps. On a benchmark where the variables are relatively independent, it does not. My two experiments are consistent with this reading, though two data points do not prove a mechanism.

## 5.3 What would falsify this hypothesis

Two things would push me to abandon the coupling hypothesis. First, a third benchmark with intermediate coupling. If MSL (Mars Science Laboratory spacecraft telemetry, from Hundman et al., 2018) with its fifty-five channels of loosely coupled sensor data showed a significant MSWEA effect, that would suggest the effect is not about coupling but about something else, perhaps dataset size or anomaly rate. If MSL showed no effect, my hypothesis would survive. This is the natural next experiment and I have marked it as future work.

Second, an alternative explanation might fit the data equally well. One candidate is dataset size: SKAB has around 500 training rows per file, SMD has around 28,000. Small datasets often benefit from architectural inductive biases (like MSWEA's forced cross-variable attention) that regularise the model, whereas large datasets can learn similar structure implicitly. This is compatible with what I saw: MSWEA helps on the small SKAB files and not on the larger SMD machines. I do not have a way to disentangle "coupling" from "size" with only two datasets, and this is worth being explicit about. The finding might be about coupling, or size, or an interaction of both.

If I could rerun a fraction of the compute budget on a synthetic benchmark where I control both the number of coupled variables and the number of training samples, I could disentangle these. That is not something I could do inside this dissertation.

## 5.4 What the Fusionformer paper claims, and what I found

Wang et al. (2025) claim that FAM (their term for the MSWAA + MSWEA combination) is a general architectural improvement for multivariate time series forecasting, and by extension for the anomaly detection that is downstream of forecasting. The paper evaluates on four proprietary open-pit mine datasets, which have physical properties I cannot inspect, and reports that Fusionformer beats six baselines on those datasets at prediction horizons from 16 to 256 minutes.

My finding is more limited. On the two public benchmarks I could test, MSWEA gives a large significant improvement on the one where the sensor variables are physically coupled (SKAB) and only a small inconclusive improvement on the one where they are not (SMD). Nothing in this contradicts the paper directly. The paper's mining data are almost certainly a case of tightly coupled variables (slope movement signals from geologically linked sensors), which is exactly the regime where my hypothesis predicts MSWEA should work best. My results are compatible with the paper's if the coupling hypothesis is right: the paper works on coupled data (mining), works on coupled data in a laboratory context (SKAB), and adds only marginal, unconfirmed value on decoupled data (SMD).

A second finding from Chapter 4.5.1 is worth restating here. The paired LSTM-versus-Fusionformer comparison shows the full architecture beats a straight non-transformer baseline by two to five AUROC points on both benchmarks, both significant (SMD p = 0.020, SKAB p = 0.0001). This is the paper's headline claim in a weaker form: the Fusionformer architecture as a whole earns its parameters against an obvious alternative, on both benchmarks I tested. That is compatible with the paper. The narrower MSWEA ablation then asks a more specific question, which the paper does not itself address: how much of that architectural gain comes from the variable-axis attention branch specifically? On SKAB, roughly one AUROC point out of the four Fusionformer gains over LSTM is directly attributable to MSWEA. On SMD, the answer is a fraction of a point, and it does not reach significance.

The value of my study, I think, is that it moves the paper's claim from a general one ("FAM helps for MTS anomaly detection") to a more specific one ("FAM helps most for MTS anomaly detection when variables are physically coupled"). That specific claim is more useful in practice because it tells a practitioner when to expect the intervariable attention branch to earn its parameters and when a simpler time-only architecture will do.

## 5.5 Limitations

There are several things this study does not and cannot do, and I want to state them plainly rather than bury them.

**The SWSE ablation was not run.** I attempted a no_SWSE variant that swaps segment tokens for per-timestep tokens. The variant creates roughly sixteen times more attention work per epoch and exhausted the Metal Performance Shaders memory on my MacBook M5 during training. One run failed to complete a single epoch in eleven hours of wall time. The correct next step is to run this on a cloud GPU with more memory. I have marked this as the highest-priority piece of future work in Chapter 6.

**The adversarial component was not properly evaluated.** My initial experiments with the paper's adversarial loop showed rapid discriminator collapse: within about five epochs, the discriminator loss dropped below 0.01 and the generator's prediction quality started degrading as the adversarial gradient became noise. Stabilising this would require a hyperparameter sweep over the adversarial loss weight, the discriminator learning rate, and the ratio of discriminator to generator updates. That sweep did not fit inside the compute budget. All experiments in Chapter 4 used the adversarial loss weight set to zero. Whether the paper's adversarial contribution is real or artefactual is left open.

**SMD was underpowered.** The nine paired SMD comparisons produced p = 0.203, r = 0.424 (medium effect). With a sample size of nine, the Wilcoxon test can only reliably detect large effects, and a medium effect at n = 9 is exactly the region where the test cannot distinguish "no effect" from "small effect". Seven of nine pairs favour the full model. This looks like a small positive trend that the study is underpowered to confirm rather than a clean null. Extending SMD to fifteen or twenty paired comparisons across more machines is a natural strengthening of the study, and I have run out of local compute time to do it before submission.

**Hyperparameter deviations from the paper.** Chapter 3.6 lists these but they matter to how I read the ablation. My d_model is 252 rather than the paper's 256, my encoder and decoder depth is 1 + 1 rather than 4 + 3, and my segment length is 4 rather than 32. The ablation is internally consistent because both full and MSWEA-off variants use the same values. What might not carry over is my absolute AUROC values compared to what a paper-exact configuration would produce on the same data. A single-seed paper-hyperparameter sanity check on SMD machine-1-1 was run and returned AUROC 0.923 (Chapter 3.6.1), which sits inside the top of the compute-budget distribution and shows no dramatic uplift, but that same run showed a training divergence at epoch 12 that the compute-budget configuration did not exhibit. A full paired paper-hyperparameter ablation was not attempted because one such run took 105 minutes.

**The coupling hypothesis is a mechanism, not a proof.** Two benchmarks give two data points. Two data points suggest a shape but do not establish one. I have written this chapter as an argument that the hypothesis is plausible and consistent with what the paper reports on its own data, but the burden of proof for the mechanism sits with future work.

**The forecasting objective is not the paper's task exactly.** The paper's downstream is slope failure prediction on mining data. Mine is anomaly detection on pumping and server data using forecast error as the anomaly score. The pipeline shape is the same. The evaluation metric is different. If the paper's FAM has a specific advantage in the mining-forecasting composition that does not show up in the anomaly-forecast-error composition, my study cannot see it.

## 5.6 Implications for practitioners

If I strip out the qualifications and try to say something useful for someone deciding whether to use Fusionformer FAM in a new anomaly detection setup, it would be this.

The intervariable attention branch appears to be worth its parameters when the input variables are physically coupled, and appears not to be worth its parameters when the variables are relatively independent. On tightly coupled sensor data (industrial equipment, environmental sensors on a single physical system, biomedical signals from one patient), turning MSWEA on is likely to help. On loosely coupled multivariate telemetry (server metrics, financial time series, aggregated business KPIs), a time-only transformer of the same size is likely to perform indistinguishably from the full FAM.

The parameter cost of MSWEA is not enormous in my configuration (3.44 million with, 2.67 million without), but on larger models with more heads and deeper stacks the difference grows. If a practitioner has a reason to keep the model small (edge deployment, inference latency, energy budget), the coupling test is a reasonable predictor of whether MSWEA is worth including.

## 5.7 Summary

The MSWEA branch of Fusionformer's fusion attention module gives a large statistically significant improvement on SKAB (p = 0.002, r = 0.797) and only a small inconclusive improvement on SMD (p = 0.203, r = 0.424, medium effect but underpowered at n = 9). A paired comparison against an LSTM baseline confirmed that the Fusionformer architecture as a whole beats a non-transformer forecaster by two to five AUROC points on both benchmarks (SMD p = 0.020, SKAB p = 0.0001), so the ablation compares two respectable variants of a competent model. The most plausible mechanism, given two data points, is that MSWEA is more useful when input variables are physically coupled and less useful when they are relatively independent. Two of the three components I set out to ablate in the Task 1 DDR (SWSE and adversarial) could not be tested within the compute budget of my hardware and are documented as future work. The dissertation's contribution is a specific, benchmark-dependent claim about when the paper's headline mechanism actually earns its parameters. Chapter 6 draws the contributions together and identifies the next experiments.
