---
title: "Email to Lina Huang — GPU cluster access request"
---

**To:** Lina.Huang@brunel.ac.uk
**Cc:** Yongmin.Li@brunel.ac.uk, Xiaohui.Liu@brunel.ac.uk
**Subject:** Urgent GPU cluster access request — MSc dissertation, travelling 6 Sept

Dear Lina,

I am writing to request access to the GPU cluster service Prof Yongmin Li announced this week. I am an MSc Artificial Intelligence student (2550458) working on my dissertation under Prof Xiaohui Liu on an independent reproduction and ablation of the Fusionformer architecture (Wang et al., 2025, TNNLS).

**The request in one line:** approximately 24 hours of A100 or L40 GPU time to complete a component ablation that is computationally infeasible on my local hardware.

**Context.** The Fusionformer forecasting model with paper-matched hyperparameters (d_model=252, n_heads=6, sequence length 96) trains on my MacBook M5 in about 95 minutes per run. However, one of the ablation variants I need to run to answer my third research question ("no SWSE" — using per-timestep tokens instead of segment embeddings) creates approximately 16 times more attention work per epoch and runs out of Metal Performance Shaders memory. A single run has been active on my laptop for 11 hours without completing one epoch. On a 40 GB A100 the same 9 runs should complete in 4-5 hours.

**What I would run.** Nine training runs (3 machines from SMD × 3 random seeds) of the paper-faithful Fusionformer with the SWSE component ablated. The full source code is self-contained (approximately 400 lines of PyTorch, no external services required). I can bring a completed application form, an SSH public key, and can be ready to submit jobs today if given access.

**Timing constraint.** I am travelling to India on Saturday 6 September for a family situation. I would very much appreciate any way to have access set up before Friday evening so I can run this remaining experiment before travel. My dissertation submission target is 20 September.

I have read Prof Li's announcement and I am reading the application form and user guide now. If it would help, I am happy to come to campus this afternoon or tomorrow to complete any onboarding.

Please let me know what I need to provide.

Thank you very much for considering this.

Best regards,

Nachiket Magadum
Student ID: 2550458
MSc Artificial Intelligence, Brunel University London
nachiketmagadum@gmail.com
Supervisor: Prof Xiaohui Liu
