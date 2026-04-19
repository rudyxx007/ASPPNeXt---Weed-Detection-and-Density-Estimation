# Module 3: Crop Masking Architecture (U-Net + EfficientNet)

This diagram shows the architecture of the crop segmentation model, featuring an EfficientNet-B5 encoder and a U-Net decoder with SCSE attention.

```mermaid
graph TD
    A[Input Augmented RGB Image] --> B[Encoder: EfficientNet-B5];

    subgraph Encoder
        direction TB
        B --> S1[Stage 1];
        S1 --> S2[Stage 2];
        S2 --> S3[Stage 3];
        S3 --> S4[Stage 4];
        S4 --> S5[Center Block];
    end

    subgraph Decoder
        direction TB
        S5 --> D1[Decoder Block 1];
        S4 -- Skip Connection --> D1;
        D1 --> D2[Decoder Block 2];
        S3 -- Skip Connection --> D2;
        D2 --> D3[Decoder Block 3];
        S2 -- Skip Connection --> D3;
        D3 --> D4[Decoder Block 4];
        S1 -- Skip Connection --> D4;
    end

    D4 --> H[Final Upsampling & Classification Head];
    H --> Z[Output: Crop Probability Map];

    style Z fill:#1976d2,color:#fff
    style B fill:#e8f5e9
    style D1 fill:#f1f8e9
    style D2 fill:#f1f8e9
    style D3 fill:#f1f8e9
    style D4 fill:#f1f8e9
```
