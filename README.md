# ASPPNeXt: Advanced Weed Detection and Density Estimation

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## 🌾 Project Overview

**ASPPNeXt** is a cutting-edge computer vision research project that combines multiple deep learning techniques for agricultural applications, specifically focusing on **weed detection and density estimation** in crop fields. The project implements a novel neural network architecture that leverages both RGB and depth information through advanced attention mechanisms and multi-modal fusion.

### 🎯 Key Features

- **Novel ASPPNeXt Architecture**: Hybrid ConvNeXt blocks with Varying Window Attention (VWA)
- **Multi-Modal Processing**: RGB and depth feature fusion using Dual-stream Adaptive Attention Fusion (DAAF)
- **Advanced Segmentation**: U-Net with EfficientNet-B5 backbone achieving 99.43% accuracy
- **Unsupervised Learning**: Ensemble vegetation segmentation with 10+ vegetation indices
- **Realistic Data Augmentation**: Environmental and geometric transformations for robust training
- **Efficient Architecture**: GhostModule integration for parameter reduction while maintaining performance

## 🏗️ Architecture Overview

```
RGB Input ──┐                    ┌── Depth Input
            │                    │
    ┌───────▼────────┐  ┌────────▼────────┐
    │ Hybrid ConvNeXt│  │ Hybrid ConvNeXt │
    │ Block ×4       │  │ Block ×4        │
    │ (VWA+GhostMLP) │  │ (VWA+GhostMLP)  │
    └───────┬────────┘  └────────┬────────┘
            │                    │
    ┌───────▼────────┐  ┌────────▼────────┐
    │ Pre-DAAF       │  │ Pre-DAAF        │
    │ GhostModule    │  │ GhostModule     │
    └───────┬────────┘  └────────┬────────┘
            │                    │
            └──────► DAAF ◄──────┘
                      │
            ┌─────────▼─────────┐
            │ Post-DAAF         │
            │ GhostModule       │
            └─────────┬─────────┘
                      │
         ┌────────────▼────────────┐
         │ Decoder (3 Stages)     │
         │ GhostASPPFELAN         │
         │ CoordAttention         │
         │ DySample Upsampling    │
         └────────────┬────────────┘
                      │
                 Final Output
```

## 📁 Project Structure

```
ASPPNeXt/
├── 1-Realistic Field Conditioning.ipynb     # Data augmentation pipeline
├── 2-Crop Masking with U-Net EfficientNet Backbone & Inplace-ABN.ipynb
├── 3-Vegetation Segmentation with Unsupervised Learning.ipynb
├── 4 - Weed Mask Generation.ipynb           # Weed-specific algorithms
├── 5 - Depth Feature Generation.ipynb       # MiDaS depth estimation
├── 6 - ASPPNeXt.ipynb                      # Main architecture
├── README.md
└── documentation.txt                        # Detailed technical docs
```

## 🚀 Pipeline Components

### 1. **Realistic Field Conditioning**
- **Purpose**: Advanced data augmentation for agricultural scenarios
- **Features**: Environmental effects (sun flare, rain, fog) + geometric transformations
- **Output**: 4x data multiplication (400→1600 train samples)
- **Technology**: Albumentations library with CUDA acceleration

### 2. **Crop Masking with U-Net**
- **Architecture**: U-Net + EfficientNet-B5 backbone + Inplace-ABN
- **Performance**: 99.43% accuracy, 97.02% mean IoU
- **Features**: SCSE attention, Focal Tversky Loss, dynamic hyperparameters
- **Classes**: Background vs. Crop segmentation

### 3. **Vegetation Segmentation (AEVS)**
- **Method**: Adaptive Ensemble Vegetation Segmentation
- **Indices**: 10+ vegetation indices (ExG, NGRDI, CIVE, etc.)
- **Techniques**: Multi-Otsu, Watershed, K-means clustering
- **Output**: Robust vegetation masks with morphological refinement

### 4. **Depth Feature Generation**
- **Model**: Intel MiDaS DPT_Large (pre-trained)
- **Processing**: Batch processing with CUDA acceleration
- **Output**: High-quality depth maps for all dataset splits

### 5. **ASPPNeXt Architecture** ⭐
- **Innovation**: Novel hybrid architecture combining ConvNeXt and attention mechanisms
- **Key Components**:
  - **GhostModuleV2**: Efficient feature generation with DFC attention
  - **Hybrid ConvNeXt Block**: VWA + Ghost MLP
  - **DAAF**: Dual-stream Adaptive Attention Fusion
  - **Decoder**: GhostASPPFELAN with multi-scale processing

## 📊 Performance Metrics

| Component | Metric | Value |
|-----------|--------|-------|
| U-Net Crop Segmentation | Accuracy | 99.43% |
| U-Net Crop Segmentation | Mean IoU | 97.02% |
| U-Net Crop Segmentation | F1-Score | 97.27% |
| U-Net Crop Segmentation | Crop IoU | 94.68% |
| Depth Generation | Processing Speed | ~4 images/batch |
| Data Augmentation | Multiplication Factor | 4x |

## 🛠️ Installation & Setup

### Prerequisites
- Python 3.8+
- CUDA-compatible GPU (recommended)
- 16GB+ RAM

### Dependencies
```bash
pip install torch torchvision
pip install opencv-python numpy albumentations
pip install segmentation-models-pytorch
pip install scikit-learn matplotlib tqdm
pip install timm scipy scikit-image
```

### Dataset Setup
1. Place CWF-788 dataset in `CWF-788/IMAGE512x384/`
2. Ensure train/test/validation splits with corresponding masks
3. Image format: 512x384 JPG with PNG masks

## 🔬 Usage

### 1. Data Preparation
```bash
# Run data augmentation
jupyter notebook "1-Realistic Field Conditioning.ipynb"
```

### 2. Depth Feature Generation
```bash
# Generate depth maps
jupyter notebook "5 - Depth Feature Generation.ipynb"
```

### 3. Model Training
```bash
# Train U-Net for crop segmentation
jupyter notebook "2-Crop Masking with U-Net EfficientNet Backbone & Inplace-ABN.ipynb"
```

### 4. Vegetation Analysis
```bash
# Run unsupervised vegetation segmentation
jupyter notebook "3-Vegetation Segmentation with Unsupervised Learning.ipynb"
```

### 5. ASPPNeXt Architecture
```bash
# Main architecture implementation
jupyter notebook "6 - ASPPNeXt.ipynb"
```

## 🔬 Research Contributions

### 1. **Novel Architecture Design**
- ASPPNeXt combines state-of-the-art techniques in a unified framework
- Efficient GhostModule integration reduces parameters while maintaining performance
- Multi-modal RGB-Depth fusion with cross-attention mechanisms

### 2. **Comprehensive Agricultural Pipeline**
- End-to-end solution from data augmentation to weed detection
- Realistic field condition simulation for robust training
- Multiple segmentation approaches for enhanced reliability

### 3. **Advanced Attention Mechanisms**
- Varying Window Attention (VWA) for adaptive receptive fields
- Dual-stream Adaptive Attention Fusion (DAAF) for multi-modal processing
- CoordAttention for spatial-channel attention in decoder

### 4. **Ensemble Methods**
- Multiple vegetation indices for robust vegetation detection
- Combination of supervised and unsupervised approaches
- Morphological refinement for improved mask quality

## 📈 Technical Innovations

- **GhostModuleV2**: Enhanced with DFC attention and Hardswish activation
- **Hybrid ConvNeXt Block**: Combines convolutional and transformer approaches
- **DAAF Fusion**: Novel dual-stream processing for RGB-Depth integration
- **GhostASPPFELAN**: Multi-scale dilated convolutions in decoder
- **DySample**: Content-aware upsampling for better detail preservation

## 🎯 Applications

- **Precision Agriculture**: Automated weed detection and management
- **Crop Monitoring**: Vegetation health assessment and growth tracking
- **Agricultural Robotics**: Real-time processing for autonomous systems
- **Research**: Agricultural computer vision and multi-modal learning

## 🔮 Future Development

### Immediate Goals
- [ ] Complete ASPPNeXt implementation and training
- [ ] Implement weed mask generation algorithms
- [ ] Full pipeline integration and testing
- [ ] Performance optimization for real-time processing

### Research Directions
- [ ] Real-time deployment on edge devices
- [ ] Temporal analysis for growth monitoring
- [ ] Multi-spectral image support
- [ ] Integration with agricultural robotics platforms

## 📚 Citation

If you use this work in your research, please cite:

```bibtex
@misc{asppnext2025,
  title={ASPPNeXt: Advanced Weed Detection and Density Estimation using Multi-Modal Deep Learning},
  author={[Rudra Naik]},
  year={2025},
  note={Research Project - AAU Internship}
}
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📞 Contact

For questions or collaborations, please reach out through the project repository.

---

**Keywords**: Computer Vision, Deep Learning, Agricultural AI, Weed Detection, Semantic Segmentation, Multi-Modal Learning, Attention Mechanisms, PyTorch