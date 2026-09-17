---
title: "Follow-up email to Prof Liu — evening of 3 Sept"
---

**Subject:** Follow-up to this morning: results, plan, and travel notice

Dear Professor Liu,

Thank you for making time this morning even briefly. I wanted to follow up in writing because the meeting was short and there are a few things I would benefit from your input on before travelling.

**Current state of the work.** During final proofreading I re-read your paper closely and identified that my earlier reconstruction-based implementation had departed from Fusionformer on several architectural points. I rebuilt the model this week to match the paper's design: SWSE embedding, MSWAA and MSWEA in the FAM, discriminator, and forecasting objective. A first end-to-end run on SMD machine-1-1 with paper-matched hyperparameters (d_model=252 for divisibility by n_heads=6, L_seg=4, N_enc=1, T=96, tau=24) produced clean monotonically-decreasing training loss and Test AUROC 0.9389.

**Datasets.** I am using SMD (Server Machine Dataset, 28 server telemetry machines with anomaly labels) as the primary evaluation benchmark, and I plan to extend to MSL (NASA Mars Science Laboratory telemetry) as a genuinely different-domain cross-validation. The paper's mining dataset is proprietary and I do not have access; I have made this a limitation in the write-up.

**Ablation plan before travel.** In the next two days I will run 3 model variants (full paper, no MSWEA, no SWSE) across 3 SMD machines at 3 random seeds each, with Wilcoxon signed-rank on 9 paired AUROC values per comparison. This will answer whether MSWEA and SWSE individually contribute to performance on SMD.

**Three technical questions I would appreciate your view on:**

1. The paper reports d_model=256 with n_heads=6; 256 is not divisible by 6, so I use d_model=252. Was the paper's implementation possibly using non-standard head dimensions, or is 252 the appropriate reading?
2. In MSWEA, "all segments within U^{tim} are merged" — I have implemented this as per-segment attention across the D variables (embed_dim = d_model). Is this the intended interpretation, or is a fully-flattened variant preferred?
3. Did you tune the adversarial loss weight? My initial value caused discriminator collapse on SMD; a warmup schedule with lower weight is in the ablation.

If you are able to share the paper's exact configuration file or preprocessing details, that would materially strengthen the reproduction. Understood if not possible.

**Travel.** A family situation means I need to travel to India on Saturday 6 September. I will take my laptop and continue the writing and remaining analysis from there without interruption, and I will attend all our scheduled meetings by video call as usual. I am targeting submission of the final report between Monday 15 September and Sunday 20 September via WISEflow. Please let me know if you would prefer a different date within that window or would like to see a mid-week draft.

Thank you again for your supervision.

Best regards,

Nachiket Magadum
Student ID: 2550458
MSc Artificial Intelligence, Brunel University London
