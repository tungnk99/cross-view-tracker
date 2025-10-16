# Pipeline Types and Architecture

This document clarifies the three different pipeline architectures and the role of various matching techniques.

## Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                     PIPELINE ARCHITECTURES                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  1. CLASSICAL (Sparse)                                          │
│     Extractor → Matcher → Mapper                                │
│     [SIFT/ORB] → [BruteForce/FLANN] → [Homography/Affine]      │
│                                                                  │
│  2. SEMI-DENSE (Learned Matching)                               │
│     Semi-Dense Matcher → Mapper                                 │
│     [LoFTR/LightGlue] → [Homography/Affine]                    │
│                                                                  │
│  3. END-TO-END (Dense Correspondence)                           │
│     Dense Correspondence Model                                   │
│     [DKM/PDC-Net/GLU-Net] → Direct Mapping                     │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## Pipeline Details

### 1. Classical Pipeline (Sparse Feature-Based)

**Architecture:** Extractor → Matcher → Mapper

**Components:**
- **Extractor:** SIFT, ORB, SuperPoint
- **Matcher:** BruteForce, FLANN
- **Mapper:** Homography, Affine (with RANSAC)

**Characteristics:**
- ✅ Sparse keypoint detection (hundreds to thousands of points)
- ✅ Traditional computer vision techniques
- ✅ No training required
- ✅ Fast on CPU
- ✅ Interpretable and debuggable
- ⚠️ May struggle with low-texture regions
- ⚠️ Requires distinctive features

**Example:**
```python
# configs/pipeline_classical.yaml
pipeline:
  type: "classical"
  classical:
    extractor: "sift"        # Detect ~1000 keypoints
    matcher: "brute_force"   # Match descriptors
    mapper: "homography"     # Fit geometric model with RANSAC
```

---

### 2. Semi-Dense Pipeline (Learned Matching)

**Architecture:** Semi-Dense Matcher → Mapper

**Components:**
- **Matcher:** LoFTR, Efficient LoFTR, LightGlue, SuperGlue
- **Mapper:** Homography, Affine (with RANSAC)

**Characteristics:**
- ✅ **Semi-dense correspondences** (thousands to tens of thousands of points)
- ✅ Detector-free matching (learned end-to-end)
- ✅ Superior performance on challenging scenes
- ✅ Works well with low-texture regions
- ✅ Still uses geometric fitting (interpretable)
- ⚠️ Requires GPU for good performance
- ⚠️ Slower than classical methods

**Key Papers:**
1. **LoFTR** (CVPR 2021): "LoFTR: Detector-Free Local Feature Matching with Transformers"
   - Transformer-based matching
   - Produces semi-dense correspondences
   
2. **Efficient LoFTR** (CVPR 2024): "Efficient LoFTR: Semi-Dense Local Feature Matching with Sparse-Like Speed"
   - Optimized version of LoFTR
   - **Explicitly described as "Semi-Dense"**
   - Faster inference while maintaining quality

**Why LoFTR is Semi-Dense:**
> From the Efficient LoFTR paper: "Semi-Dense Local Feature Matching" refers to 
> producing more correspondences than sparse methods (SIFT/ORB) but not as dense 
> as pixel-wise optical flow. LoFTR produces coarse-level matches on feature maps, 
> then refines to produce thousands of high-quality correspondences.

**Example:**
```python
# configs/pipeline_semidense.yaml
pipeline:
  type: "semi_dense"
  semi_dense:
    matcher: "loftr"       # Semi-dense matching (no separate detector)
    mapper: "homography"   # Geometric fitting with RANSAC
```

---

### 3. End-to-End Pipeline (Dense Correspondence)

**Architecture:** Dense Correspondence Model → Direct Mapping

**Components:**
- **Model:** DKM, PDC-Net, GLU-Net, RAFT
- **Mapping:** Direct (no explicit geometric fitting)

**Characteristics:**
- ✅ **Dense pixel-wise correspondences** (every pixel has a match)
- ✅ Single unified model
- ✅ No separate geometric fitting step
- ✅ Can handle extreme viewpoint changes
- ✅ Provides uncertainty estimates
- ⚠️ Requires GPU
- ⚠️ Slower inference
- ⚠️ Less interpretable (black box)

**True E2E Models:**

1. **DKM (Deep Kernelized Dense Matching)**
   - Dense geometric matching
   - Kernel-based representation
   - Outdoor/indoor variants

2. **PDC-Net (Probabilistic Dense Correspondence Network)**
   - Probabilistic dense matching
   - Uncertainty estimation
   - Multi-scale approach

3. **GLU-Net (Global-Local Universal Network)**
   - Global and local matching
   - Universal across domains
   - Strong generalization

4. **RAFT (Recurrent All-Pairs Field Transforms)**
   - Originally for optical flow
   - Can be adapted for cross-view matching
   - Iterative refinement

**Example:**
```python
# configs/pipeline_e2e.yaml
pipeline:
  type: "e2e"
  e2e:
    model: "dkm"  # Dense correspondence model
    # Direct mapping without explicit RANSAC
```

---

## Comparison Table

| Aspect | Classical | Semi-Dense | End-to-End |
|--------|-----------|------------|------------|
| **Match Density** | Sparse (100s) | Semi-Dense (1000s) | Dense (all pixels) |
| **Learning** | None | Matcher only | Full E2E |
| **Geometric Fitting** | ✅ RANSAC | ✅ RANSAC | ❌ Implicit |
| **Interpretability** | ⭐⭐⭐ | ⭐⭐ | ⭐ |
| **GPU Required** | ❌ | ✅ Recommended | ✅ Required |
| **Speed (CPU)** | ⚡ Fast | 🐌 Slow | 🐌 Very Slow |
| **Accuracy** | Good | Excellent | Excellent |
| **Robustness** | Moderate | High | Very High |

---

## When to Use Each Pipeline

### Use Classical Pipeline When:
- ✅ No GPU available
- ✅ Need fast inference
- ✅ Working with high-texture scenes
- ✅ Need interpretable results
- ✅ Building baseline comparisons

### Use Semi-Dense Pipeline When:
- ✅ Have GPU available
- ✅ Need high accuracy
- ✅ Working with challenging scenes (low-texture, repetitive patterns)
- ✅ Want balance of accuracy and interpretability
- ✅ **Recommended for most cross-view tracking tasks**

### Use End-to-End Pipeline When:
- ✅ Have GPU available
- ✅ Need maximum robustness
- ✅ Working with extreme viewpoint changes
- ✅ Need dense correspondence fields
- ✅ Can sacrifice interpretability for performance

---

## Implementation Notes

### Current Implementation Status

✅ **Implemented:**
- Classical Pipeline with SIFT/ORB
- Semi-Dense Pipeline with LoFTR
- Homography/Affine mappers

🔄 **Partially Implemented:**
- E2E Pipeline (using LoFTR-based approach)

❌ **Not Yet Implemented:**
- True E2E models (DKM, PDC-Net, GLU-Net)
- Learned mapper

### Efficient LoFTR Classification

Based on the paper "Efficient LoFTR: Semi-Dense Local Feature Matching with Sparse-Like Speed":

```
❌ NOT End-to-End: Efficient LoFTR is a MATCHER, not a complete pipeline
✅ IS Semi-Dense: Produces semi-dense correspondences (1000s of points)
✅ REQUIRES Mapper: Still needs geometric fitting (Homography/Affine)
```

**Correct Usage:**
```yaml
# Semi-Dense Pipeline
pipeline:
  type: "semi_dense"
  semi_dense:
    matcher: "loftr"  # or "efficient_loftr"
    mapper: "homography"
```

**Incorrect Usage:**
```yaml
# ❌ WRONG: LoFTR is not an E2E model
pipeline:
  type: "e2e"
  e2e:
    model: "efficient_loftr"  # This is a matcher, not E2E
```

---

## References

### Papers

1. **SIFT** - "Distinctive Image Features from Scale-Invariant Keypoints" (IJCV 2004)
2. **ORB** - "ORB: An efficient alternative to SIFT or SURF" (ICCV 2011)
3. **SuperPoint** - "SuperPoint: Self-Supervised Interest Point Detection and Description" (CVPR 2018)
4. **SuperGlue** - "SuperGlue: Learning Feature Matching with Graph Neural Networks" (CVPR 2020)
5. **LoFTR** - "LoFTR: Detector-Free Local Feature Matching with Transformers" (CVPR 2021)
6. **LightGlue** - "LightGlue: Local Feature Matching at Light Speed" (ICCV 2023)
7. **Efficient LoFTR** - "Efficient LoFTR: Semi-Dense Local Feature Matching with Sparse-Like Speed" (CVPR 2024)
8. **DKM** - "DKM: Deep Kernelized Dense Geometric Matching" (ECCV 2022)
9. **PDC-Net** - "PDC-Net+: Enhanced Probabilistic Dense Correspondence Network" (TPAMI 2021)
10. **GLU-Net** - "GLU-Net: Global-Local Universal Network for Dense Flow and Correspondences" (CVPR 2020)

---

## Summary

The three pipeline types represent different tradeoffs:

1. **Classical** = Fast + Interpretable, but limited accuracy
2. **Semi-Dense** = Best balance of accuracy + interpretability ⭐ **RECOMMENDED**
3. **End-to-End** = Maximum performance, but less interpretable

**Key Insight:** Efficient LoFTR is a **semi-dense matcher**, not an end-to-end model. It produces semi-dense correspondences that still require geometric fitting (RANSAC) to obtain the final transformation.

For most cross-view tracking applications, the **Semi-Dense Pipeline with LoFTR** provides the best results.

