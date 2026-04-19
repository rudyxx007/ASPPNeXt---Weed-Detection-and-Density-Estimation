# Custom U-Net Architecture with EfficientNet-B7 Backbone and Attention

This document outlines the architecture of a custom U-Net model that leverages EfficientNet-B7 as its encoder backbone and includes attention mechanisms in the skip connections.

```mermaid
graph TD
    subgraph "Encoder (EfficientNet-B7)"
        direction LR
        A[Input Image <br> 224x224x3] --> B{Stem <br> Conv 3x3}
        B --> C{Stage 1 <br> MBConv}
        C --> D{Stage 2 <br> MBConv}
        D --> E{Stage 3 <br> MBConv}
        E --> F{Stage 4 <br> MBConv}
        F --> G{Stage 5 <br> MBConv}
    end

    subgraph "Bridge"
        H[Bottleneck <br> MBConv Block]
    end

    subgraph "Decoder (Upsampling Path)"
        direction LR
        I[UpConv 1] --> J[Concat 1]
        J --> K[Conv Block 1]
        K --> L[UpConv 2]
        L --> M[Concat 2]
        M --> N[Conv Block 2]
        N --> O[UpConv 3]
        O --> P[Concat 3]
        P --> Q[Conv Block 3]
        Q --> R[UpConv 4]
        R --> S[Concat 4]
        S --> T[Conv Block 4]
    end

    subgraph "Output"
        U[Final Conv 1x1] --> V[Output Mask <br> 224x224x1]
    end

    subgraph "Skip Connections"
        direction TD
        Att1[Attention Block 1]
        Att2[Attention Block 2]
        Att3[Attention Block 3]
        Att4[Attention Block 4]
    end

    G --> H
    H --> I
    F --> Att1
    Att1 --> J
    E --> Att2
    Att2 --> M
    D --> Att3
    Att3 --> P
    C --> Att4
    Att4 --> S
    T --> U

    classDef encoder fill:#DDEBF7,stroke:#333,stroke-width:2px
    classDef decoder fill:#E2F0D9,stroke:#333,stroke-width:2px
    classDef bridge fill:#F8CBAD,stroke:#333,stroke-width:2px
    classDef attention fill:#FFF2CC,stroke:#333,stroke-width:2px

    class A,B,C,D,E,F,G encoder
    class I,J,K,L,M,N,O,P,Q,R,S,T decoder
    class H bridge
    class Att1,Att2,Att3,Att4 attention
```

### Architecture Details:

1.  **Encoder**: The encoder is a pre-trained EfficientNet-B7 network. It processes the input image through its successive stages, downsampling the spatial dimensions and increasing the feature depth. Each stage shown in the diagram corresponds to a set of MBConv blocks from the original EfficientNet architecture.

2.  **Bridge/Bottleneck**: This is the deepest part of the network, connecting the encoder and the decoder. It consists of the final block of the EfficientNet encoder.

3.  **Decoder**: The decoder is the upsampling path. It takes the feature map from the bottleneck and progressively upsamples it. At each upsampling step, the feature map is concatenated with features from the corresponding level of the encoder via a skip connection.

4.  **Skip Connections & Attention Blocks**: Before the features from the encoder are concatenated with the decoder's feature maps, they are passed through an **Attention Block**. This block helps the model to focus on the most relevant features from the encoder path, improving the segmentation accuracy.

5.  **Output**: The final decoder block's output is passed through a 1x1 convolution to produce the final segmentation mask with the desired number of classes.
