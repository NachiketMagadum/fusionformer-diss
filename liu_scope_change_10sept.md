---
title: "Email to Prof Liu — scope change from DDR to submission"
date: "10 September 2026"
---

**To:** Xiaohui.Liu@brunel.ac.uk
**Subject:** Dissertation update — scope changes since the DDR, ahead of submission

Dear Professor Liu,

Ahead of submitting my dissertation next week I wanted to give you a written update on how the project changed from what I proposed in the DDR, so that nothing in the final document comes as a surprise.

**What the DDR proposed.** An independent reproduction and ablation of Fusionformer's three components (SWSE, FAM, adversarial training), evaluated on the Skoltech Anomaly Benchmark (SKAB) and the Server Machine Dataset (SMD), with a Wilcoxon-based multi-seed protocol, an EdgeFusion pathway for edge deployment, and a Formula 1 telemetry cross-domain demonstration.

**Where the project ended up.** During final proofreading in early September I re-read the paper carefully and realised my initial implementation had diverged from Fusionformer in an important way: I had built a reconstruction-based autoencoder inspired by the paper's component names, whereas Fusionformer is a forecasting model that uses forecast error as the anomaly signal. I rebuilt the model faithfully from Section III of the paper. On the corrected paper-faithful implementation I completed the following experiments:

- **MSWEA (intervariable attention) ablation on SMD**: 3 machines × 3 seeds = 9 paired runs. Full versus MSWEA-off. Result: no significant difference, Wilcoxon W = 48.5, p = 0.513, Rosenthal r = 0.169.
- **MSWEA ablation on SKAB**: 5 files × 3 seeds = 15 paired runs. Full versus MSWEA-off. Result: statistically significant improvement for the full model, Wilcoxon W = 9.00, p = 0.002, Rosenthal r = 0.797 (large).
- **Paper-faithful Fusionformer FAM absolute performance**: mean AUROC 0.907 on SMD and 0.879 on SKAB across all 24 runs.

The headline finding is a benchmark-dependent effect: MSWEA does not measurably help on SMD's 38 loosely-coupled server metrics but significantly helps on SKAB's 8 tightly-coupled industrial sensors. I develop this into a plausible mechanism in Chapter 5, arguing that the intervariable attention branch's usefulness depends on the physical coupling of the input channels.

**Components I could not test.** Three items in the original DDR did not fit within the compute budget on my MacBook M5 and are documented as future work in Chapter 6:

- The SWSE ablation. The no-SWSE variant expands per-attention token count by roughly 16 times and exhausted MPS memory. One training run failed to complete even one epoch in 11 hours. This is a real hardware limitation on unified memory rather than a design failure.
- The adversarial training component. The discriminator collapsed under my default hyperparameters and stabilising it would have required a sweep over the adversarial loss weight, discriminator learning rate, and update ratio. That sweep did not fit before travel. Chapter 6 recommends this as the immediate next experiment on cloud or institutional GPU compute.
- The Formula 1 telemetry case study. It was completed on my earlier reconstruction-based pipeline, which I have now discarded. Rerunning it on the paper-faithful forecasting model was not compatible with the remaining compute budget.

**Deviations from the paper's exact hyperparameters.** Documented explicitly in Chapter 3.6. In summary: d_model=252 rather than 256 for divisibility by 6 heads, segment length 4 rather than 32 for finer temporal granularity, encoder and decoder depth of 1+1 rather than 4+3 for compute feasibility, and the adversarial loss weight set to zero for the reasons above.

**Submission timeline.** I am targeting submission on Sunday 14 September or Monday 15 September via WISEflow. If any of the above is a concern, or if you would prefer I add or reframe something before final submission, please let me know as soon as you can. I have Chapters 1 to 4 in a working draft and I am starting Chapters 5 and 6 today.

**One thing I would appreciate your view on.** The benchmark-dependent MSWEA finding is genuinely the most interesting result in the dissertation, but it rests on a specific hypothesis (variable coupling determines when MSWEA helps) that I cannot test rigorously without a third benchmark with intermediate coupling. Do you have a view on whether it is more appropriate to state the hypothesis as a proposal for future work, or to argue it as the most plausible mechanism given the evidence I do have?

Thank you again for your supervision over the summer. I know the picture is a little different from the DDR and I wanted to lay it out plainly rather than have it come as a surprise on submission.

Best regards,

Nachiket Magadum
Student ID: 2550458
MSc Artificial Intelligence, Brunel University London
