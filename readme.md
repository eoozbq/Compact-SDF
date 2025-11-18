# DeepSDF-Based Multi-Dataset Shape Reconstruction

This repository provides training, mesh generation, and evaluation scripts for shape reconstruction tasks across multiple datasets including **Stanford**, **DFAUST**, and **ShapeNet**.

---

## 📁 Provided Scripts

- `train_stanford.py` — Train on the **Stanford** dataset  
- `train_dfaust.py` — Train on the **DFAUST** dataset  
- `train_shapenet.py` — Train on the **ShapeNet** dataset  
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

