# Compact-SDF
Code for "Learning Compact Latent Space for Representing Neural Signed Distance Functions with High-fidelity Geometry Details" (AAAI 2026).
## 💡 Abstract
Neural signed distance functions (SDFs) have been a vital representation to represent 3D shapes or scenes with neural networks. An SDF is an implicit function that can query signed distances at specific coordinates for recovering a 3D surface. Although implicit functions work well on a single shape or scene, they pose obstacles when analyzing multiple SDFs with high-fidelity geometry details, due to the non-compact representations of SDFs and the loss of geometry details. To overcome these obstacles, we introduce a method to represent multiple SDFs in a common space, aiming to recover more high-fidelity geometry details with more compact latent representations. Our key idea is to take full advantage of the benefits of generalization-based and overfitting-based learning strategies, which manage to preserve high-fidelity geometry details with compact latent codes. Based on this framework, we also introduce a novel sampling strategy to sample training queries. The sampling can improve the training efficiency and eliminate artifacts caused by the influence of other SDFs. We report numerical and visual evaluations on widely used benchmarks to validate our designs and show advantages over the latest methods in terms of the representative ability and compactness.

## 🖼️ High-Fidelity Geometry Reconstruction
<div align="center">
    <img src="assest/teaser-three-sub.png" alt="Comparison of different n bandwidths on Chair reconstruction.">
</div>

## 🖼️ Qualitative Results Video Demonstration
<div align="center">
    <video src="assest/video.mp4">
</div>


This repository provides training, mesh generation, and evaluation scripts for shape reconstruction tasks across multiple datasets including **Stanford**, **DFAUST**, and **ShapeNet**.

---

## 📁 Provided Scripts
- `train_stanford.py` — Train on the **Stanford** dataset ([http://graphics.stanford.edu/data/3dscanrep/](http://graphics.stanford.edu/data/3dscanrep/))
- `train_dfaust.py` — Train on the **DFAUST** dataset ([http://dfaust.is.tue.mpg.de/](http://dfaust.is.tue.mpg.de/))
- `train_shapenet.py` — Train on the **ShapeNet** dataset ([https://shapenet.org/](https://shapenet.org/))
- `generate_shapenet_meshes.py` — Generate meshes from ShapeNet-trained models  
- `reconstruct_dfaust.py` — Reconstruct meshes from partial point clouds on the DFAUST dataset  
- `evaluate.py` — Evaluate reconstruction quality using **Chamfer Distance (CD)**

### 🧠 Model Implementation

The core decoder model is defined in:

- `networks/deep_sdf_decoder.py`  
  
  > Our proposed method is implemented as `ours_decoder`.

---

## 🚀 Example: Single-Object Reconstruction (Stanford Armadillo)

We provide an example using the Armadillo model from the Stanford dataset to demonstrate the training and mesh reconstruction process.

### 1. Prepare the Armadillo Data

Navigate to the `data/` directory and run:

```python
python get_data.py
```

### 2. Training the Armadillo

```python
python train_stanford.py -e examples/Stanford_Armadillo
```
