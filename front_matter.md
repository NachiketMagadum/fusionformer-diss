---
title: ""
author: ""
date: ""
geometry: margin=2.5cm
fontsize: 11pt
mainfont: "Times New Roman"
linestretch: 1.5
---

**Department of Computer Science**

**MSc Artificial Intelligence**

**Academic Year 2025-2026**

&nbsp;

&nbsp;

&nbsp;

# Independent Reproduction and Ablation of Fusionformer for Multivariate Time Series Anomaly Detection

## A Paired Multi-Seed Study on SKAB and SMD

&nbsp;

&nbsp;

**Nachiket Magadum**

Registration Number: 2550458

&nbsp;

A report submitted in partial fulfilment of the requirement for the degree of Master of Science

&nbsp;

&nbsp;

Brunel University London

Department of Computer Science

Uxbridge, Middlesex UB8 3PH

United Kingdom

\newpage

# ABSTRACT

Multivariate time series anomaly detection has become a central task in industrial monitoring, server operations, and safety-critical systems, and transformer-based architectures have emerged as the dominant approach in the recent literature. This dissertation is an independent reproduction and rigorous ablation of Fusionformer (Wang et al., 2025), a recent forecasting-based transformer architecture co-authored by this dissertation's supervisor and published in IEEE Transactions on Neural Networks and Learning Systems. The paper proposes three architectural components (a segment-wise sequence embedding, a fusion attention module built from time-axis and variable-axis attention branches, and an adversarial training loop) and reports strong results on a proprietary open-pit mine slope failure dataset. The paper does not test the components individually or on public multivariate anomaly detection benchmarks.

I built a paper-faithful implementation of Fusionformer in PyTorch and ran a paired multi-seed ablation of the variable-axis attention branch on two public benchmarks: the Skoltech Anomaly Benchmark (SKAB, eight tightly-coupled industrial pump sensors) and the Server Machine Dataset (SMD, thirty-eight relatively independent server telemetry channels). Twenty-four training runs were executed in total, nine paired comparisons on SMD and fifteen paired comparisons on SKAB, with Wilcoxon signed-rank tests and Rosenthal effect sizes reported for each benchmark separately.

The findings are benchmark-dependent. On SMD the variable-axis attention branch shows only a small, statistically inconclusive improvement over an ablation that removes it (mean AUROC 0.8904 with the branch versus 0.8856 without, Wilcoxon p = 0.203, Rosenthal r = 0.424, a medium effect on which the study is underpowered at n = 9). On SKAB the same branch gives a statistically significant improvement (mean AUROC 0.8790 versus 0.8640, Wilcoxon p = 0.002, r = 0.797, a large effect). The dissertation argues that the most plausible mechanism, given two data points, is that the variable-axis attention branch is more useful when input variables are physically coupled and less useful when they are relatively independent. A separate paired comparison against an LSTM baseline confirmed that the Fusionformer architecture as a whole beats a non-transformer forecaster on both benchmarks by three to five AUROC points (SMD p = 0.020, SKAB p = 0.0001), so the ablation is a comparison between two respectable models rather than one strong model and one broken one. Two of the three components proposed in the paper (the segment-wise sequence embedding and the adversarial training loop) could not be ablated within the compute budget of the study hardware and are documented as future work. A third-benchmark sweep on MSL (Hundman et al., 2018) was attempted but abandoned after two full runs converged only to near-chance AUROC at the compute-budget configuration, indicating the deeper paper-exact model on cloud GPU is the correct next step for a full MSL comparison.

The contributions are a paper-faithful open-source Fusionformer implementation, a paired statistical ablation of the variable-axis attention branch on two contrasting benchmarks, a paired non-transformer baseline comparison that confirms the architecture is doing real work, a specific mechanistic hypothesis about when the branch earns its parameters, and a reusable methodological template for deep-learning ablation studies.

**Keywords**: anomaly detection, multivariate time series, transformer reproduction, ablation study, Wilcoxon signed-rank, SKAB, SMD.

\newpage

# ACKNOWLEDGEMENTS

I am grateful to Professor Xiaohui Liu for supervising this dissertation. His willingness to have me reproduce and ablate one of his own co-authored papers, and to encourage honest reporting of null and negative findings, was central to the direction of the work. His feedback across the summer meetings materially improved what this dissertation is about.

I would also like to acknowledge the open-source data providers whose work made this study possible: the Skoltech team behind the SKAB benchmark, and Su et al. (2019) for the Server Machine Dataset. Reproducibility in machine learning research depends entirely on these public resources.

Thank you to my family for their patience during the final month of writing while I was travelling.

&nbsp;

**Plagiarism Declaration**

I certify that the work presented in this dissertation is my own unless referenced. I have used AI-based coding assistance as a productivity aid during prototyping, in line with Brunel University's guidance on AI use in dissertations. All committed code was read, understood, refactored, and validated by me before inclusion in the submitted work. Every experimental number in this dissertation was produced by running scripts I wrote and understand.

Signature: ______________________________________

Date: ______________________________________

&nbsp;

**TOTAL NUMBER OF WORDS: 13193**

\newpage

# TABLE OF CONTENTS

**CHAPTER 1: INTRODUCTION**

- 1.1 Research aim and objectives
- 1.2 Research approach
- 1.3 Dissertation outline

**CHAPTER 2: LITERATURE REVIEW**

- 2.1 Multivariate time series anomaly detection
- 2.2 Transformers for time series
- 2.3 Anomaly detection with transformers
- 2.4 The Fusionformer paper
- 2.5 Evaluation practice and the point-adjustment problem
- 2.6 Summary

**CHAPTER 3: METHODOLOGY**

- 3.1 Datasets
- 3.2 Model implementation
- 3.3 Training regime
- 3.4 Evaluation metrics
- 3.5 Experimental design
- 3.6 Deviations from the paper's exact configuration
  - 3.6.1 Paper-hyperparameter sanity check
- 3.7 Summary

**CHAPTER 4: RESULTS**

- 4.1 Fusionformer FAM on SMD
- 4.2 MSWEA ablation on SMD
- 4.3 Fusionformer FAM on SKAB
- 4.4 MSWEA ablation on SKAB
- 4.5 Cross-benchmark comparison
  - 4.5.1 Head-to-head against an LSTM baseline
- 4.6 Summary

**CHAPTER 5: DISCUSSION**

- 5.1 A benchmark-dependent effect
- 5.2 A working hypothesis: variable coupling
- 5.3 What would falsify this hypothesis
- 5.4 What the Fusionformer paper claims, and what I found
- 5.5 Limitations
- 5.6 Implications for practitioners
- 5.7 Summary

**CHAPTER 6: CONCLUSION**

- 6.1 Summary of the dissertation
- 6.2 Research contributions
- 6.3 Future research and development
- 6.4 Personal reflections

**REFERENCES**

**APPENDIX A: ETHICAL APPROVAL**

**APPENDIX B: CODE**

**APPENDIX C: REPRODUCIBILITY STATEMENT**

\newpage
