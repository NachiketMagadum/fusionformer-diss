---
title: "Meeting prep: Prof Liu, Thursday 3 September 2026"
subtitle: "Personal briefing — not for sharing"
author: "Nachiket Magadum"
date: "3 September 2026, 05:30"
mainfont: "Times New Roman"
fontsize: 11pt
geometry: margin=2.5cm
---

# Meeting one-liner (what this meeting is for)

Report progress honestly, disclose the architecture rebuild done this week, confirm 29 September submission window given the family emergency, and get Liu's sign-off on the corrected direction.

# The three things you must say clearly (in order)

1. **"I re-read your paper carefully during proofreading and realised my earlier implementation deviated from Fusionformer on four material points. I rebuilt the model faithfully this week and I have first results."**

2. **"Sanity check on SMD machine-1-1 with the paper-matched architecture (d_model=252, n_heads=6, SWSE, MSWAA+MSWEA, discriminator, forecasting objective with forecast-error anomaly scoring) gives AUROC 0.9434. Loss decreases cleanly across 30 epochs."**

3. **"A family emergency requires me to travel to India on 6 September. I would like to submit on 28 or 29 September rather than the earlier date, to complete the corrected reproduction properly with multi-seed multi-machine ablation and a rewritten manuscript."**

# What to show on screen (if he asks)

- `fusionformer_true.py` — walk him through the architecture blocks briefly: SWSE (his paper's eq 2-5), MSWAA (eq 6-7), MSWEA (eq 8-11), FusionformerDiscriminator (Section III-E).
- `notes/ff_true_smd_machine-1-1_seed42.txt` — the 0.9434 result with hyperparameter table.
- `PIVOT_PLAN.md` — the honest one-page explanation of what changed and why.

# The four deviations from the paper you should disclose upfront

Get these on the table before he finds them:

1. **We used d_model=252, not 256.** 256 is not divisible by 6, so we dropped to the nearest multiple of 6 to keep n_heads=6 exactly. 1.6% dimension reduction.
2. **We used learning rate 1e-4, not 1e-3.** The paper's grid starts at 5e-3 down to 1e-4; 1e-3 diverged for us after epoch 1 on SMD. Lower LR trained stably.
3. **We tested MSWEA as per-segment attention across variables rather than fully merged.** Interpreted paper equation 8's "all segments... are merged" pragmatically because the literal merge produces an intractably large attention matrix (443M parameters). Standard practice from iTransformer and PatchTST.
4. **We do not have access to the paper's mining data.** Using SMD (and SKAB, MSL if extended) as public MVAD proxies. Explicitly a task-adjacent reproduction, not a like-for-like replication.

If he pushes back on any of these, ask him for the correct interpretation and adjust. Do not defend.

# Points to raise proactively

- Cross-benchmark plan: SMD (in progress), MSL from Telemanom (ready to run), possibly SKAB. Multi-seed with Wilcoxon on all.
- Ablation plan: MSWEA-off, SWSE-off, adversarial-on (with warmup + λ_adv sweep). Each × 3 seeds × N machines.
- Adversarial training initially collapsed (D_loss to 0) with default λ=0.05 and no warmup. Fix: 10-epoch pure-MSE warmup then adversarial at λ=0.01. This is the "task-transfer boundary" claim now backed by a concrete failure mode I observed and diagnosed.
- Compute budget: 5h per run at paper-matched settings on M5. Applied three legitimate speedups (batch 32→128, epochs 30→20, N_enc 2→1) to get ~50 min per run without changing architectural claims. N_enc=1 is inside the paper's own sensitivity sweep range.

# NOT to raise unless he asks

- Do NOT go into detail on the family situation. One line, professional. If asked what/who, say "a family member is unwell and I need to be there."
- Do NOT mention the earlier reconstruction-based implementation as if it was the plan all along. Own the correction cleanly: "I found the deviations during proofreading and fixed them."
- Do NOT mention Attewell, travel dates beyond 6 Sept, or any unrelated context.
- Do NOT bring up Claude / AI assistant usage unless he raises it — the ethics appendix already covers it.

# Backup answers for likely questions

**"Why did you not test on my slope-failure data?"**
"Access to the specific mining dataset was not achievable in the available timeframe. I noted this as the primary future work direction. If you can share preprocessing scripts or a sanitised version, I would run it as an extension."

**"How confident are you in the FAM ablation results?"**
"The earlier reconstruction-based ablation results are being replaced. The new paper-faithful ablation is in progress — SMD machine-1-1 base result 0.9434, next step is 5 machines × 3 seeds × 4 model variants. I will run Wilcoxon on all pairs and report effect sizes."

**"How many attention heads did you use?"**
"Six, matching your paper. d_model is 252 rather than 256 so the divisibility works — 1.6% deviation."

**"What is the forecasting horizon?"**
"T=96 historical steps, τ=24 forecast horizon. Standard for MVTS forecasting benchmarks. The paper's slope failure application uses longer horizons, but 24 is standard for the public benchmarks I am using."

**"When exactly will I see the full manuscript?"**
"Full draft with corrected results by 25 September. Final submitted version by 28 or 29 September via WISEflow."

**"Can you finish this in a month while travelling?"**
"Yes. Compute runs on my laptop; I can bring it. Writing polish and results integration are feasible from India. The pivot is code-heavy this week, then writing-heavy for the next three weeks."

# Ask by end of meeting

Three explicit yes/no confirmations to walk out with:

1. Approval to submit on 28 or 29 September.
2. Approval of the paper-faithful reproduction direction (with the four disclosed deviations).
3. Any specific results or ablations he would prioritise, ordered.

Write down whatever he says immediately after the meeting.

# Opening line to have ready

"Thank you for making time this morning. I have three things to update you on: a technical correction I made this week, current results, and one logistical request. I will keep it to fifteen minutes unless you have questions."

Then take a breath and start with item 1 (the architecture rebuild).
