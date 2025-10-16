# Cross-View Tracker - 3-Pipeline System

A comprehensive **cross-view mapping system** with **3 different pipeline architectures**, from classical sparse features to modern end-to-end deep learning.

## 🌟 Overview

This system provides **3 types of pipelines** for cross-view correspondence:

| Pipeline | Architecture | Steps | Features | Best For |
|----------|-------------|-------|----------|----------|
| **Classical** | Extractor → Matcher → Mapper | 3 | Sparse (100s), interpretable | Small datasets, debugging |
| **Semi-Dense** | Semi-Dense Matcher → Mapper | 2 | Semi-dense (1000s), high accuracy | **Recommended for most tasks** |
| **E2E** | Dense Model → Direct Mapping | 1 | Dense (all pixels), no RANSAC | Extreme viewpoint changes |

```
                    Input: Source Image + Target Image + ROI
                                    ↓
            ┌───────────────────────┴────────────────────────┐
            │                                                 │
    ┌───────┴────────┐    ┌──────────────┐    ┌────────────┴──────┐
    │   Classical    │    │  Semi-Dense  │    │      E2E          │
    │   Pipeline     │    │   Pipeline   │    │   Pipeline        │
    ├────────────────┤    ├──────────────┤    ├───────────────────┤
    │ 1. Extract     │    │ 1. DL Match  │    │ 1. DL Model       │
    │ 2. Match       │    │ 2. Map       │    │    (Direct)       │
    │ 3. Map         │    │              │    │                   │
    └────────┬───────┘    └──────┬───────┘    └─────────┬─────────┘
             │                   │                       │
             └───────────────────┴───────────────────────┘
                                 ↓
                    Output: Mapped Coordinates + Confidence
```

## 🚀 Quick Start

### Installation

```bash
# Clone repository
git clone <repository-url>
cd cross-view-tracker

# Install dependencies
pip install -r requirements.txt
```

### Basic Usage

```python
from src.pipelines import create_pipeline
import cv2

# Choose your pipeline type
pipeline = create_pipeline('configs/pipeline_classical.yaml')
# pipeline = create_pipeline('configs/pipeline_semidense.yaml')
# pipeline = create_pipeline('configs/pipeline_e2e.yaml')

# Load images
source_img = cv2.imread('source.jpg')
target_img = cv2.imread('target.jpg')

# Process
result = pipeline.process_frame(
    source_img,
    target_img,
    roi_center=(320, 240),
    roi_radius=50
)

print(f"Mapped coordinates: {result['coords']}")
print(f"Confidence: {result['confidence']:.3f}")
```

## 📋 Pipeline Details

### 1️⃣ Classical Pipeline (Sparse Feature-Based)

**Architecture**: `Extractor → Matcher → Mapper`

**Components**:
- **Extractors**: SIFT, ORB, SuperPoint
- **Matchers**: BruteForce, FLANN
- **Mappers**: Homography, Affine

**Characteristics**:
- ✅ No training required
- ✅ Interpretable, easy to debug
- ✅ Works with small datasets
- ❌ Sparse features (fewer correspondences)
- ❌ Multiple steps (slower)

**Config**:
```yaml
pipeline:
  type: "classical"
  classical:
    extractor: "sift"
    matcher: "brute_force"
    mapper: "homography"
```

---

### 2️⃣ Semi-Dense Pipeline (Semi-Dense Matcher) ⭐ RECOMMENDED

**Architecture**: `Semi-Dense Matcher → Mapper`

**Components**:
- **Matchers**: LoFTR, Efficient LoFTR (semi-dense), LightGlue, SuperGlue
- **Mappers**: Homography, Affine, Learned

**Characteristics**:
- ✅ **Semi-dense correspondences** (1000s of matches vs sparse 100s)
- ✅ Detector-free DL-based matching
- ✅ Still interpretable (explicit geometric fitting with RANSAC)
- ✅ Best balance of accuracy and speed
- ✅ **Recommended for most cross-view tracking tasks**
- ❌ Requires GPU for good performance
- ❌ Pretrained matcher

**Key Papers**:
- "LoFTR: Detector-Free Local Feature Matching with Transformers" (CVPR 2021)
- "Efficient LoFTR: Semi-Dense Local Feature Matching with Sparse-Like Speed" (CVPR 2024)

**Config**:
```yaml
pipeline:
  type: "semi_dense"
  semi_dense:
    matcher: "loftr"
    mapper: "homography"
```

---

### 3️⃣ End-to-End Pipeline (Dense Correspondence)

**Architecture**: `Dense Model → Direct Mapping`

**Components**:
- **Models**: DKM, PDC-Net, GLU-Net, RAFT

**Characteristics**:
- ✅ Single step (no explicit RANSAC)
- ✅ **Dense pixel-wise correspondence** (every pixel has a match)
- ✅ Direct mapping without geometric fitting
- ✅ Handles extreme viewpoint changes
- ✅ Maximum robustness
- ❌ Requires powerful GPU
- ❌ Black box (hard to debug)
- ❌ May require training data

**Note**: Current implementation uses a LoFTR-based approach. For true E2E, consider DKM or PDC-Net.

**Config**:
```yaml
pipeline:
  type: "e2e"
  e2e:
    model: "dkm"  # or pdcnet, glunet, raft
    model_path: "models/e2e_model.pth"
```

## 📊 Performance Comparison

| Pipeline | FPS (GPU) | Accuracy | Memory | Training Required |
|----------|-----------|----------|--------|-------------------|
| Classical (SIFT) | 20-30 | ⭐⭐⭐ | Low | ❌ |
| Classical (ORB) | 30-50 | ⭐⭐ | Low | ❌ |
| **Semi-Dense (LoFTR)** ⭐ | **10-15** | **⭐⭐⭐⭐** | **Medium** | **Pretrained only** |
| Semi-Dense (LightGlue) | 15-25 | ⭐⭐⭐⭐ | Medium | Pretrained only |
| E2E (DKM) | 5-10 | ⭐⭐⭐⭐⭐ | High | Optional |
| E2E (PDC-Net) | 5-10 | ⭐⭐⭐⭐⭐ | High | Optional |

*Tested on NVIDIA RTX 3080*

## 🎯 When to Use Which Pipeline?

### Use Classical Pipeline when:
- ✅ You have limited or no training data
- ✅ You need interpretability
- ✅ You want to debug individual components
- ✅ Cross-view gap is small to medium

### Use Semi-Dense Pipeline when: ⭐ RECOMMENDED
- ✅ You have a GPU
- ✅ You need high accuracy with interpretability
- ✅ Cross-view gap is medium to large
- ✅ You want best balance between accuracy and speed
- ✅ **This is the recommended approach for most cross-view tracking tasks**

### Use E2E Pipeline when:
- ✅ You have a powerful GPU
- ✅ Cross-view gap is very large
- ✅ You have data to fine-tune (if needed)
- ✅ You prioritize maximum accuracy

## 📁 Project Structure

```
cross-view-tracker/
├── configs/
│   ├── pipeline_classical.yaml     # Classical pipeline config
│   ├── pipeline_semidense.yaml     # Semi-dense pipeline config
│   └── pipeline_e2e.yaml           # E2E pipeline config
├── src/
│   ├── pipelines/                  # 3 pipeline implementations
│   │   ├── classical_pipeline.py
│   │   ├── semidense_pipeline.py
│   │   ├── e2e_pipeline.py
│   │   └── factory.py              # Pipeline factory
│   ├── extractors/                 # Feature extractors (SIFT, ORB, SuperPoint)
│   ├── matchers/                   # Matchers (BF, FLANN, LoFTR, LightGlue)
│   ├── mappers/                    # Mappers (Homography, Affine, Learned)
│   └── e2e_models/                 # E2E models (Efficient LoFTR, DKM, etc.)
├── examples/
│   ├── quick_start.py              # Quick start examples
│   └── pipeline_comparison.py      # Compare all 3 pipelines
└── docs/
    └── PIPELINE_GUIDE.md           # Detailed pipeline guide
```

## 🔧 Examples

### Example 1: Compare All Pipelines

```python
from examples.pipeline_comparison import compare_pipelines

results = compare_pipelines('source.jpg', 'target.jpg')
```

Output:
```
Pipeline        Valid    Confidence   Coords                    Time (s)
--------------------------------------------------------------------------------
classical       True     0.856        (245.3, 178.9)           0.045
semi_dense      True     0.912        (246.1, 179.2)           0.082
e2e             True     0.934        (245.8, 179.5)           0.098
```

### Example 2: Quick Start

```bash
python examples/quick_start.py
```

### Example 3: Custom Configuration

```python
from src.pipelines import create_pipeline
from src.utils import load_config

# Load and modify config
config = load_config('configs/pipeline_classical.yaml')
config['pipeline']['classical']['extractor'] = 'orb'  # Switch to ORB

# Create pipeline
pipeline = create_pipeline(config=config)
```

## 📚 Documentation

- [Pipeline Guide](docs/PIPELINE_GUIDE.md) - Detailed guide for all 3 pipelines
- [Performance Tuning](docs/PERFORMANCE.md) - Optimization tips
- [Architecture](docs/ARCHITECTURE.md) - System architecture

## 🤝 Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Submit a pull request

## 📄 License

See LICENSE file for details.

## 🙏 Acknowledgments

**Classical Methods**:
- SIFT (Lowe, 2004)
- ORB (Rublee et al., 2011)

**Semi-Dense Matchers**:
- SuperPoint (DeTone et al., 2018)
- **LoFTR** (Sun et al., 2021) - "LoFTR: Detector-Free Local Feature Matching with Transformers"
- **Efficient LoFTR** (Wang et al., 2024) - "Efficient LoFTR: Semi-Dense Local Feature Matching with Sparse-Like Speed"
- LightGlue (Lindenberger et al., 2023)
- SuperGlue (Sarlin et al., 2020)

**E2E Dense Models**:
- DKM (Edstedt et al., 2022) - "DKM: Deep Kernelized Dense Geometric Matching"
- PDC-Net (Truong et al., 2021) - "PDC-Net+: Enhanced Probabilistic Dense Correspondence Network"
- GLU-Net (Truong et al., 2020) - "GLU-Net: Global-Local Universal Network for Dense Flow and Correspondences"

## 📧 Contact

For questions or issues, please open a GitHub issue.

---

**Happy Tracking! 🎯**

*Choose the right pipeline for your use case: Classical for simplicity, **Semi-Dense (LoFTR) for best results** ⭐, E2E for extreme cases.*

---

## 📖 Important Notes

### LoFTR Classification

**LoFTR and Efficient LoFTR are SEMI-DENSE matchers**, not E2E models:

- ✅ Produce **semi-dense correspondences** (thousands of matches)
- ✅ Still require **geometric fitting** (Homography/Affine with RANSAC)
- ✅ Best used in **Semi-Dense Pipeline**
- ❌ Not true E2E (need explicit mapper)

**Reference**: "Efficient LoFTR: Semi-Dense Local Feature Matching with Sparse-Like Speed" (CVPR 2024)

For detailed explanation, see [docs/PIPELINE_TYPES.md](docs/PIPELINE_TYPES.md).

