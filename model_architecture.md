# Model architecture reference

Diagrams for the dissertation writeup. Convert to PNG via mermaid CLI or use
online at https://mermaid.live.

## 1. Full Fusionformer autoencoder pipeline

Reconstruction-based anomaly detection setup. Input window is passed through
input projection → positional encoding → N encoder blocks → N decoder blocks
→ output projection → reconstruction. Anomaly score = mean squared error
between input and reconstruction.

```mermaid
flowchart TD
    X["Input window<br/>x: (B, T, F)<br/>B=batch, T=30, F=features"]

    IP1["Input Projection<br/>Linear(F, d_model)<br/>d_model=32"]
    PE["Add Positional Encoding<br/>Learned (1, T, F)"]

    EB1["Encoder Block 1<br/>FusionAttention + FFN"]
    EB2["Encoder Block 2<br/>FusionAttention + FFN"]

    DB1["Decoder Block 1<br/>FusionAttention + FFN"]
    DB2["Decoder Block 2<br/>FusionAttention + FFN"]

    OP1["Output Projection<br/>Linear(d_model, F)"]

    RECON["Reconstruction<br/>x_hat: (B, T, F)"]
    LOSS["Reconstruction Loss<br/>MSE(x_hat, x)"]
    SCORE["Anomaly Score<br/>mean squared error<br/>per window"]

    X --> IP1 --> PE --> EB1 --> EB2 --> DB1 --> DB2 --> OP1 --> RECON
    RECON --> LOSS
    RECON --> SCORE
```

## 2. FusionAttention block detail

The paper's key innovation: dual-axis attention combining time-axis and
channel-axis attention with a learnable fusion weight. Post-norm transformer
block with residual connections.

```mermaid
flowchart LR
    IN["Input (B, T, F)"]

    IP["Input Projection<br/>Linear F to d_model"]

    TA["Time-axis Attention<br/>MultiHeadAttention<br/>attends across T tokens"]
    CA["Channel-axis Attention<br/>MultiHeadAttention<br/>attends across d_model channels"]

    FW["Learnable Fusion Weights<br/>softmax(w_time, w_channel)<br/>init 0.5, 0.5"]

    FUSE["Weighted Sum<br/>w_t * TA_out + w_c * CA_out"]
    OP["Output Projection<br/>Linear d_model to F"]

    ADD1["Residual + LayerNorm"]

    FF1["Linear F to ff_hidden"]
    RELU["ReLU"]
    DROP["Dropout"]
    FF2["Linear ff_hidden to F"]

    ADD2["Residual + LayerNorm"]
    OUT["Output (B, T, F)"]

    IN --> IP
    IP --> TA
    IP --> CA
    TA --> FW
    CA --> FW
    FW --> FUSE
    FUSE --> OP

    IN --> ADD1
    OP --> ADD1
    ADD1 --> FF1 --> RELU --> DROP --> FF2 --> ADD2
    ADD1 --> ADD2
    ADD2 --> OUT
```

## 3. Variant tree — all models tested

Baseline Fusionformer FAM was tested against a TimeOnly ablation (null result
on both SKAB and SMD) and against 8 SWSE configurations (all failed).

```mermaid
flowchart TD
    ROOT["Fusionformer FAM<br/>baseline autoencoder<br/>38,744 params"]

    ABL["Ablation: TimeOnly<br/>drops channel attention<br/>23,856 params<br/>NULL FINDING (0.9285 vs 0.9285 on SMD)"]

    S1["SWSE + FAM (seg=5, 50ep) -0.23"]
    S2["SWSE + FAM (seg=2, 50ep) -0.32"]
    S3["SWSE + FAM (seg=5, 20ep) -0.26"]
    S4["SWSE + TimeOnly -0.30"]
    S5["SWSE + FAM + PreNorm -0.56"]
    S6["SWSE + FAM + RevIN -0.34"]
    S7["SWSE + TimeOnly + RevIN -0.34"]
    S8["SWSE + FAM + Adversarial -0.40"]

    SKAB["SKAB benchmark<br/>23 files, 8 features<br/>138 training runs"]
    SMD["SMD benchmark<br/>5 machines, 38 features<br/>15 training runs"]

    ROOT --> ABL
    ROOT --> S1
    ROOT --> S2
    ROOT --> S3
    ROOT --> S4
    ROOT --> S5
    ROOT --> S6
    ROOT --> S7
    ROOT --> S8
    ROOT --> SKAB
    ROOT --> SMD
    ABL --> SKAB
    ABL --> SMD

    style ROOT fill:#2E7D32,color:#fff
    style ABL fill:#F4A261
    style S1 fill:#C62828,color:#fff
    style S2 fill:#C62828,color:#fff
    style S3 fill:#C62828,color:#fff
    style S4 fill:#C62828,color:#fff
    style S5 fill:#C62828,color:#fff
    style S6 fill:#C62828,color:#fff
    style S7 fill:#C62828,color:#fff
    style S8 fill:#C62828,color:#fff
```

## 4. Data flow — inference pipeline

For anomaly scoring at inference time.

```mermaid
flowchart LR
    RAW["Raw data<br/>N rows, F features"]
    STD["StandardScaler<br/>per file/machine"]
    WIN["Sliding Windows<br/>seq_len=30"]
    TRAIN["Trained Fusionformer<br/>eval mode"]
    RECON2["Reconstruction<br/>x_hat"]
    ERR["Per-window MSE<br/>(x - x_hat)^2 mean"]
    ROW["Per-row scores<br/>broadcast window scores"]
    METR["Metrics<br/>AUROC, PR-AUC, F1-PA, Event-F1"]

    RAW --> STD --> WIN --> TRAIN --> RECON2 --> ERR --> ROW --> METR
```

## Rendering

**Fastest**: paste any code block above into https://mermaid.live

**To PNG for dissertation**:

```bash
# Install once
npm install -g @mermaid-js/mermaid-cli

# Convert this whole file to individual PNGs
mmdc -i notes/model_architecture.md -o figures/architecture.png
```

**In VS Code**: install "Markdown Preview Mermaid Support" extension, then Cmd+K V to preview.
