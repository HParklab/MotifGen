# MotifGen

MotifGen is a deep learning model designed to predict binding motifs solely from a target receptor structure. It provides a probability profile of functional groups at surface grid points, enabling interpretable and design-oriented applications.

---

## Installation Guide

### 1. Install UCSF Chimera (Required for Hydrogen Addition)
Download and install UCSF Chimera from the official website:  
🔗 [UCSF Chimera Download](https://www.cgl.ucsf.edu/chimera/download.html)

### 2. Set Up the Environment
Run the following commands to install dependencies and set up the `motifgen` environment:

```bash
conda env create -f environment.yaml
conda activate motifgen
```

Install the required libraries:

```bash
pip install torch==1.11.0+cu113 --index-url https://download.pytorch.org/whl/cu113
pip install dgl-cu113==0.9.1.post1 dglgo -f https://data.dgl.ai/wheels/repo.html
pip install e3nn==0.5.0
pip install -e .
```

---

## Applications

### Application 1: **Motif-Site Prediction**  
MotifGen can predict ligand binding sites using a preprocessed motif profile.

### Steps Overview
1. **Preprocessing(add hydrogen on PDB)**
2. **Feature extraction** 
3. **Run MotifGen** (GPU-accelerated inference)
4. **Predict ligand binding sites**

---


### **Step 1: Prepare Input PDB File**
**Input Requirement:** PDB file with **all hydrogens attached** (heteroatoms will be ignored).  
> If you already have a PDB file with all hydrogens attached, you can skip this step.

Run the following command to add hydrogens using UCSF Chimera:

```bash
chimera --nogui src/motifgen/featurize/add_h.py prefix.pdb prefix.pdb
```


### **Step 2: Generate Grid and Surface Features**
Run feature extraction with the following command:

```bash
python src/motifgen/featurize/featurize_usage_latest.py $prefix.pdb
```

**Input:** `$prefix.pdb`  
**Output:**
- `$prefix.grid.xyz`
- `$prefix.lig.npz`
- `$prefix.prop.npz`

### **Step 3: Run MotifGen Model (GPU Recommended)**
Run the MotifGen model to predict motif scores:

```bash
python scripts/predict_motifgen.py $prefix.lig.npz original
```

**Output:** `$prefix.score.npz`

> **Tip:** Running this step on a **GPU** is highly recommended as it significantly speeds up the computation.

### **Step 4: Predict Ligand Binding Sites**
Use the binding site predictor to analyze high-probability binding sites:

```bash
python scripts/site_predictor.py $prefix
```

#### **Output Files:**
1. **Predicted Binding Sites**: `$prefix.cl.pdb`
   - Contains high-probability motif locations (**UNK**) and ligand binding sites:
     - **ZN** → 1st-ranked site
     - **Cl** → Other predicted sites

2. **Property Predictions**: `$prefix.log`
   - Contains predicted **logP** and **TPSA** values.

---

## Example Usage
Here’s an example command to run the full pipeline on `148lE.pdb`:

```bash
prefix=./example/motifsite/148lE
chimera --nogui src/motifgen/featurize/add_h.py $prefix.pdb $prefix.pdb
python src/motifgen/featurize/featurize_usage_latest.py $prefix.pdb
python scripts/predict_motifgen.py $prefix.lig.npz original
python scripts/site_predictor.py $prefix
```
