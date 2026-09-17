---
title: "Pre-meeting note to Prof Liu — Thu 3 Sept 2026"
author: "Nachiket Magadum"
date: "Sent Wed 2 Sept 2026 (draft prepared Mon 1 Sept)"
mainfont: "Times New Roman"
fontsize: 11pt
geometry: margin=2.5cm
---

## Suggested email / Teams message

**Subject:** Dissertation progress update and Thursday meeting prep

Dear Professor Liu,

Ahead of our meeting on Thursday 3 September I want to send a short progress update and one logistical ask.

**Where the dissertation stands.** The main body is drafted (approximately 12,000 words across six chapters, following the CS5500 Task 2 template), the code is stitched into a single Jupyter notebook, and every experiment on SKAB and SMD reported so far has been run under a paired multi-seed protocol with Wilcoxon significance testing. The three headline results from the ablation study are:

- FAM versus TimeOnly: no significant difference on SKAB (Wilcoxon W = 79, p = 0.518, r = 0.148 across 23 files) or on SMD (W = 48.5, p = 0.513, r = 0.169 across 15 machine-seed pairs).
- SWSE degrades performance in every configuration tested on SKAB (single-seed sweep, currently being re-run at 3 seeds to enable per-configuration significance testing).
- Full three-component paper architecture significantly underperforms FAM alone on SMD across 15 machine-seed pairs (p = 0.018, r = 0.610, large effect).

**Work in flight before submission.** I have four items still open that I would rather complete than skip:

1. A cross-benchmark validation on MSL (Mars Science Laboratory spacecraft telemetry, from Hundman et al. 2018) to confirm the FAM null on a third dataset that is genuinely different in domain from the two we have tested. Data acquisition and script are ready; expected runtime approximately one hour.
2. Re-running the SKAB SWSE configuration study at 3 seeds so that each of the eight configurations has a Wilcoxon comparison against FAM alone.
3. An adversarial-weight sensitivity sweep on one SMD machine to rule out the objection that the adversarial training failure could be a hyperparameter artefact.
4. A preliminary edge-deployment feasibility experiment (dynamic int8 quantisation of the trained FAM model) to close the EdgeFusion loop from the Task 1 DDR.

**Logistical request.** A family emergency requires me to travel to India on 6 September. I can bring my laptop and I will continue working on the dissertation from there, but I would appreciate the flexibility to submit closer to the module deadline (28-29 September) rather than the earlier date I had provisionally targeted. This would give me time to complete the four items above properly rather than dropping them. If this is acceptable I would target submission on Sunday 28 September or Monday 29 September via WISEflow.

**What I would find most useful in Thursday's meeting.**

1. Any feedback on the framing of the null and negative findings in Chapter 5 that you would like me to change.
2. Confirmation of the paper's exact hyperparameters for FusionAttention (specifically the number of attention heads used) so that I can re-run the ablation with matched settings if I have diverged.
3. A view on whether the four in-flight items above are the right priorities for the remaining month, or whether you would prefer me to swap any of them for something else.

I will bring the draft dissertation and the code notebook to the meeting.

Thank you again for your supervision across the summer, and for the flexibility around the family situation.

Best regards,

Nachiket Magadum
Student ID: 2550458
MSc Artificial Intelligence, Brunel University London

---

## Notes for the meeting itself (private, not for the email)

**Time expected:** 30-45 minutes.

**Opening the meeting:**

- Lead with the state of the work, not the emergency. Only mention travel and the 29 September ask after the substantive progress is on the table.
- The family emergency is the reason for the extension request; keep the detail vague ("a family situation requires me to travel") unless he asks. Do not overexplain.

**What to show live on screen if he asks:**

- Chapter 4.3 SKAB and SMD ablation tables and figures 4.1, 4.2, 4.5 (the FAM null).
- Chapter 4.5 multi-seed SMD full-paper result (the significant negative finding). This is the strongest evidence and the most publishable outcome.
- The Wilcoxon significance script (Section 8 of the notebook) if he wants numbers computed live.

**Points to raise proactively:**

- The four in-flight items in the email above, in order.
- Ask for confirmation of paper hyperparameters — this is the most important technical clarification I need from him.
- Ask whether he wants a mention of possible follow-up work on his own slope-failure dataset in Chapter 6.3 (already there, but flag it so he sees it).

**Not to raise unless he asks:**

- Anthropic AI usage disclosure is in the ethics appendix. Do not proactively explain unless he raises it.
- Do not go into detail on the family situation. If he asks, keep it short and professional.
- Attewell outcome is unrelated to the dissertation.

**Backup answers for likely questions:**

- "Why did you not test on my slope-failure data?" → Access to the specific preprocessing pipeline was not achievable in the available timeframe. Noted as the primary future work direction in Chapter 6.3. If he offers access, accept.
- "Are you confident in the FAM null?" → Two independent benchmarks with 3 seeds each give p = 0.518 and 0.513. MSL cross-validation in flight will make it three benchmarks. Also noting SMD n = 15 is underpowered for very small effects — this is now an explicit limitation in Chapter 5.5.
- "How many attention heads did you use, and how many does the paper use?" → I used num_heads = 1. If the paper uses more, I have a script ready to re-run the sweep with matched heads before submission.
- "Why did the adversarial training fail?" → Discriminator collapse on SKAB (loss stayed at ln 2), partial success on SMD. I did not tune the adversarial weight and I am running a sensitivity sweep now to rule out that objection.
- "You promised EdgeFusion in the DDR — where is it?" → A preliminary quantisation experiment is queued (int8 on trained FAM model). Full deployment to Raspberry Pi 5 remains future work. Chapter 6.3 sets out the pathway.

**Ask by end of meeting:**

- Confirmation of the 29 September submission window.
- Confirmation of paper's num_heads value.
- Any specific revisions to make before that date.
