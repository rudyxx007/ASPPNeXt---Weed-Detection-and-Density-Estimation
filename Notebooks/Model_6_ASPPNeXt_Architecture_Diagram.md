# Module 6: ASPPNeXt Multi-Modal Architecture

This diagram provides a high-level overview of the ASPPNeXt model, which fuses RGB and Depth information for advanced segmentation.

```mermaid
graph TD
    subgraph "Dual Encoder"
        A[Input: RGB Image] --> Enc_RGB[ASPPNeXt Encoder];
        B[Input: Depth Map] --> Enc_Depth[ASPPNeXt Encoder];
    end

    subgraph "Fusion"
        Enc_RGB -- RGB Features --> DAAF[DAAF Block<br/>(Dual-Attention Adaptive Fusion)];
        Enc_Depth -- Depth Features --> DAAF;
    end

    subgraph "Decoder"
        DAAF -- Fused Features --> Advanced[Advanced Components<br/>GhostASPP, CoordAttn, DySample];
        Advanced --> Head[Final Classification Head];
    end

    Head --> Z[Output: Multi-class Segmentation<br/>(Crop/Weed/Background)];

    style Z fill:#1976d2,color:#fff
    style DAAF fill:#fff3e0
```
