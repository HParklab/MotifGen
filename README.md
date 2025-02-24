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

⚠️ IMPORTANT: The input PDB path must be an absolute path!
When using Chimera, both input and output PDB files must be specified using absolute paths.
Run the following command to add hydrogens using UCSF Chimera:

```bash
chimera --nogui src/motifgen/featurize/add_h.py /absolute/path/to/prefix.pdb /absolute/path/to/prefix.pdb
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
abs_prefix = $(pwd)/example/motifsite/148lE
chimera --nogui src/motifgen/featurize/add_h.py $abs_prefix.pdb $abs_prefix.pdb 
python src/motifgen/featurize/featurize_usage_latest.py $prefix.pdb
python scripts/predict_motifgen.py $prefix.lig.npz original
python scripts/site_predictor.py $prefix
```


### Application 2: **MotifPepScore using MotifGen (Peptide Version)**
MotifPepScore is a **scoring network** designed to distinguish peptide binders from non-binders by integrating **MotifGen predictions** with confidence metrics from **AlphaFold2 (AF2)**.

#### Steps Overview
1. **Preprocessing (add hydrogen to PDB)** – same as Application 1  
2. **Feature extraction** (grid formation and receptor property extraction)  
3. **Run MotifGen (peptide version)**  
4. **Output generation**

---

### **Step 1: Preprocessing**
Identical to Application 1. Make sure your PDB is fully hydrogenated (absolute path when running Chimera). Example:
```bash
abs_prefix=$(pwd)/example/motifpep/4apoB
chimera --nogui src/motifgen/featurize/add_h.py $abs_prefix.pdb $abs_prefix.pdb
```

### **Step 2: Feature Extraction**

#### **Step 2-1: Grid Formation (Local Grid)**
Use the following command to create a local grid of radius 12 around a specified center of mass (COM):

```bash
python src/motifgen/featurize/accessible_COM_grids.py $prefix.pdb 27.951 -11.834 28.033
```
**Input:** `$prefix.pdb, COM of the grid`
> The last three arguments (27.951, -11.834, 28.033) represent the COM coordinates around which the grid is generated.

**Output:** `$prefix.lig.npz` (sub-files: 4apoB_grid_12.pdb, 4apoB_COM.pdb)


#### **Step 2-1:Receptor Property Extraction**
Extract receptor properties with:
```bash
python src/motifgen/featurize/featurize_receptor.py $prefix.pdb
```
**Output:** `$prefix.prop.npz` 

### **Step 3: Run MotifGen (Peptide Version)**
Run the peptide-specific MotifGen model:
```bash
python scripts/predict_motifgen.py $prefix.lig.npz peptide
```
**Output:** `$prefix.score.npz` 

### **Step 4: Output Generation with Predefined Threshold**
After prediction, visualize or extract the motif predictions with:
```bash
python scripts/visualize/visualize_output.py $prefix.score.npz
```

**Output:**  
`$prefix.score.motif_pred.pdb`  
Contains coordinates of predicted motifs  
Atom name in the PDB file reflects the predicted motif type:  
- **N** → H-bond donor  
- **O** → H-bond acceptor  
- **C** → Aliphatic  
- **R** → Aromatic  
- **B** → Charged (both donor/acceptor)  

## **Example Usage (Application 2)**

Below is an example for **4apoB**:
```bash
# 1) Preprocessing (Add Hydrogens)
prefix=./example/motifpep/4apoB
abs_prefix=$(pwd)/example/motifpep/4apoB
chimera --nogui src/motifgen/featurize/add_h.py $abs_prefix.pdb $abs_prefix.pdb

# 2-1) Local Grid Formation
python src/motifgen/featurize/accessible_COM_grids.py $prefix.pdb 27.951 -11.834 28.033

# 2-2) Receptor Property Extraction
python src/motifgen/featurize/featurize_receptor.py $prefix.pdb

# 3) Run MotifGen (Peptide Version)
python scripts/predict_motifgen.py $prefix.lig.npz peptide

# 4) Visualize or Extract Motif Predictions
python scripts/visualize/visualize_output.py $prefix.score.npz
```