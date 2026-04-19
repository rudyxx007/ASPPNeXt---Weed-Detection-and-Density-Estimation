# Module 1: Vegetation Segmentation Architecture

This diagram details the unsupervised process for extracting vegetation areas from field images by computing and fusing multiple vegetation indices.

```mermaid
graph TD
    A[Input RGB Image] --> B{Compute 12 Vegetation Indices};
    B --> C{Weighted Fusion};
    C --> D{Multi-Otsu Thresholding};
    D --> E{Post-processing};
    E --> F[Output: Binary Vegetation Mask];

    subgraph "Indices"
        direction LR
        ExG; ExR; CIVE; VEG; NDI; GLI; AGRI; VARI; MVI; BGI; CIg; Intensity;
    end

    Indices --> B;

    style F fill:#1976d2,color:#fff
```
