---
title: "Chapter 2: Literature Review"
author: "Nachiket Magadum"
date: "September 2026"
geometry: margin=2.5cm
fontsize: 11pt
mainfont: "Times New Roman"
linestretch: 1.5
---

# CHAPTER 2: LITERATURE REVIEW

This chapter surveys the work that this dissertation sits on top of. Section 2.1 covers the general problem of anomaly detection in multivariate time series and the shift from classical to deep-learning methods. Section 2.2 covers how transformers have been adapted from language to time series. Section 2.3 covers the specific subfield of transformer-based anomaly detection. Section 2.4 describes the Fusionformer paper (Wang et al., 2025) in enough detail that the ablation design in Chapter 3 makes sense. Section 2.5 covers the important critique of the field's evaluation practice raised by Wu and Keogh (2021), which shaped my choice of metrics. Section 2.6 summarises where the gap this dissertation fills sits inside the wider literature.

## 2.1 Multivariate time series anomaly detection

The task can be stated in one line. Given a sequence of readings from D sensors, produce for each timestep a score that separates normal windows from anomalous windows. That is it. The engineering difficulty sits behind the definition of normal.

Anomaly detection has a long history in statistics. Chandola et al. (2009) grouped the classical methods into three families: distance-based, density-based, and reconstruction-based. Isolation Forest (Liu, Ting and Zhou, 2008) treats each row as a point and isolates it with random splits; anomalous points get isolated in fewer splits on average. Local Outlier Factor (Breunig et al., 2000) compares the local density around a point to the density around its neighbours. These are fast and interpretable but they do not know about time. A point that looks anomalous in isolation might be perfectly ordinary in context. A pump running at 500 rpm is fine on Monday morning and a fault on Sunday night, and the difference is not in the number.

Recurrent neural networks were the first learning-based response to that limitation. Hochreiter and Schmidhuber's (1997) LSTM gave models a way to hold state across time, and Malhotra et al. (2016) showed that an LSTM encoder-decoder trained on reconstruction loss could produce useful per-window anomaly scores. Hundman et al. (2018) applied a similar idea to NASA spacecraft telemetry and released the MSL and SMAP datasets. These methods are noticeably better than the classical ones, but they inherit the problems that come with recurrent networks. Training is slow because it is sequential. Long-range dependencies are hard to preserve because the state has to be passed through many timesteps (Bengio, Simard and Frasconi, 1994).

Blázquez-García et al. (2021) survey the field up to about 2020 and note that by then the centre of gravity had shifted from statistical methods to deep learning, and within deep learning, from RNNs to attention-based architectures. The rest of this chapter follows that shift.

## 2.2 Transformers for time series

Vaswani et al. (2017) removed recurrence from sequence modelling. Every position could attend to every other position in a single self-attention pass, which meant the model could learn long-range dependencies without threading information through hidden states, and training could be parallelised across the sequence dimension. In natural language processing this became the default within a year.

Adapting the same idea to time series took longer, mostly because the input structure is different. Language tokens are discrete, of a limited vocabulary, and the sequence length is bounded by document size. Time series values are continuous and the sequence length can be arbitrary. Two design choices had to be made. First, how to embed continuous scalars into vectors that self-attention can operate on. Second, how to prevent quadratic attention cost from killing the compute budget at long sequence lengths.

Informer (Zhou et al., 2021) proposed a sparse attention mechanism that limits attention to a subset of queries, cutting the cost from quadratic to log-linear. FEDformer decomposed the sequence into trend and seasonal components. PatchTST (Nie et al., 2023) took a different route. Instead of feeding raw timesteps as tokens, it grouped adjacent timesteps into non-overlapping patches and embedded each patch. A one-hundred-step window with a patch length of ten becomes ten tokens, which is short enough for standard attention. Patching also seems to help the model learn local temporal structure inside each patch before global attention operates across patches. This idea shows up under different names across recent papers, and it appears in Fusionformer as the segment-wise sequence embedding.

The other decision that varies across the recent literature is what to attend to. Standard practice attends across time steps within a variable. iTransformer (Liu et al., 2024) inverts this, treating each variable as a token and attending across variables at each timestep. There is genuine disagreement in the field about which is more useful for what task. Fusionformer's answer is to do both, in two separate attention branches, and to combine them.

## 2.3 Anomaly detection with transformers

Once transformer sequence models existed, transformer anomaly detectors followed. There are two main framings.

The reconstruction-based framing takes a window of observations and asks the model to reproduce it, with the reconstruction error providing the anomaly score. USAD (Audibert et al., 2020) uses a pair of autoencoders in an adversarial setup. The Anomaly Transformer (Xu et al., 2022) computes an association discrepancy score based on the difference between prior-association and series-association attention distributions, and uses that as the anomaly signal.

The forecasting-based framing takes a window of past observations, predicts the immediate future, and treats the discrepancy between prediction and actual as the anomaly signal. This is what Fusionformer does. It is also what TranAD (Tuli et al., 2022) does, using an adversarial refinement loop to sharpen the forecasts.

Zhang et al. (2019) and Ruff et al. (2021) provide broader reviews of deep anomaly detection, both noting that the field has a persistent problem: strong published results on standard benchmarks that fail to reproduce on new data. That observation is what the next section is about.

## 2.4 The Fusionformer paper

Wang, Wang, Dong, Lauria, Liu, Wang, Fadzil and Liu (2025) published Fusionformer in the August issue of IEEE Transactions on Neural Networks and Learning Systems. The paper's stated task is multivariate time series forecasting, with a specific downstream application: predicting slope movement in open pit mines to warn of slope failure before it happens.

The model is described in the paper's Section III. There are three components, and Figure 2.1 shows how they fit together in the forecasting pipeline.

![Figure 2.1: Fusionformer architecture from Wang et al. (2025) Section III. The three components ablated in this dissertation are marked. SWSE (segment-wise sequence embedding) reduces the raw input into a smaller number of learned segment tokens per variable. FAM combines two attention branches (MSWAA over segments within each variable and MSWEA across variables at each segment). The adversarial component adds a discriminator that pushes the generator toward outputs that match the real distribution rather than only minimising point-wise prediction error.](figures/fig_2_1_fusionformer_architecture.png){ width=95% }

The first is the segment-wise sequence embedding, or SWSE. A window of T timesteps is partitioned per variable into non-overlapping segments of length L_seg, giving L_sn = T / L_seg segments per variable. Each segment is passed through a linear projection with a learnable positional encoding to produce a d_model-dimensional vector. The output is a four-dimensional tensor of shape (batch, D variables, L_sn segments, d_model). The intent is the same as PatchTST: reduce sequence length while preserving temporal structure inside each patch.

The second is the fusion attention mechanism, or FAM. This is where the paper's central architectural claim lives. Two multi-head attention blocks act on the SWSE output in sequence. The first, called Multi-head Segment-Wise Intravariable Attention (MSWAA), attends across the L_sn segment tokens within each variable independently. The second, called Multi-head Segment-Wise Intervariable Attention (MSWEA), attends across the D variable tokens at each segment position. The paper argues that combining these two attention flows lets the model capture both intra-variable temporal patterns and inter-variable coupling in one architecture. This is the mechanism I ablate in Chapter 4.

The third is an adversarial training loop. Alongside the forecasting Fusionformer, a discriminator is trained to distinguish the concatenation of history plus real future from the concatenation of history plus predicted future. The generator's loss combines a prediction MSE term with an adversarial term that encourages predicted futures the discriminator cannot separate from real ones. The intended benefit is a better match to the underlying data distribution than pure MSE alone provides.

The paper evaluates on four proprietary open-pit mine datasets. On the paper's own benchmarks, the full Fusionformer beats six baselines including LSTM, TCN, Informer and FEDformer at prediction horizons from 16 to 256 minutes. There is a hyperparameter sensitivity analysis in Section IV.G that reports the model is fairly robust to segment length, encoder depth, and head count.

Two things the paper does not do. It does not provide an ablation isolating the contribution of each of the three components. And it does not evaluate on the standard public benchmarks that the wider MVTS anomaly detection field uses. Both of these are natural next questions for anyone wanting to know how much of the paper's headline improvement comes from which component and whether the mechanism transfers beyond mining. Those questions are what this dissertation tries to answer.

## 2.5 Evaluation practice and the point-adjustment problem

Any survey of transformer-based MVTS anomaly detection now has to mention Wu and Keogh (2021). Their arXiv paper, and the follow-up work by Kim et al. (2022), argues that the field's dominant evaluation metric, point-adjusted F1 (F1-PA), is fundamentally misleading. The point-adjustment rule works like this. If any timestep inside a contiguous ground-truth anomaly segment is flagged, the entire segment is credited as correctly detected. Wu and Keogh show that under this rule, even a random anomaly score achieves F1-PA values that look competitive with state-of-the-art numbers reported in the literature. The problem is that anomaly segments in benchmark datasets are often long, and the probability of a random detector hitting at least one timestep inside a long segment is high.

They recommend a shift back to threshold-independent metrics like AUROC and PR-AUC, together with an honest F1 computed without point adjustment, ideally at a top-k threshold where k is chosen from the training label rate rather than tuned on the test data. Kim et al. (2022) reach similar conclusions with a broader empirical study.

The consequence for anyone doing new work in this area is that reported numbers on benchmark leaderboards are hard to trust unless the authors were careful about their metrics. This dissertation reports four metrics per run (AUROC, PR-AUC, F1-PA, Event-F1) and treats AUROC as the primary comparison. F1-PA is reported only because it appears in the wider literature and readers might want to look for it, but the substantive claims in Chapter 5 do not rest on it. This choice, along with the paired Wilcoxon significance testing described in Chapter 3, is a direct response to Wu and Keogh's critique.

## 2.6 Summary

Anomaly detection in multivariate time series has moved from statistical methods to deep learning to transformers over roughly the last fifteen years. Within the transformer family, the current design questions concern how to embed continuous timesteps, how to arrange attention across the time and variable dimensions, and how to construct a loss function that produces useful anomaly scores.

Fusionformer's proposal is a specific answer: segment the input to reduce sequence length, run attention twice in a fusion architecture that covers both time and variables, and shape the output distribution with an adversarial loop. The paper reports strong results on its own mining data, but does not test the components individually or on public benchmarks.

The gap this dissertation fills is a focused one. On two public multivariate anomaly detection benchmarks with contrasting characteristics, does the intervariable branch of the fusion attention module actually contribute to performance? The evaluation follows the Wu and Keogh (2021) recommendations on metrics and adds paired Wilcoxon significance testing so that any observed difference can be judged against seed variance. Chapter 3 sets out the experimental design in detail.
