

from pathlib import Path
from scipy.spatial.distance import cdist
import numpy as np
import torch
from pdb_utils import PDB 
from seq_utils import AA_one_hot, generate_pep_position_encoding


def compute_topk_dist(rec_coords_dict, pep_coords_dict, top_k=10):
    """
    Compute the top_k closest receptor residues to each peptide residue.
    ----------
    rec_coords_dict : dict
        Key: receptor residue index (int)
        Value: np.ndarray of shape (N, 3) with 3D coords of all atoms in that residue.
    pep_coords_dict : dict
        Key: peptide residue index (int)
        Value: np.ndarray of shape (M, 3) with 3D coords of all atoms in that residue.
    top_k : int, optional
        Number of closest receptor residues to find per peptide residue, by default 10

    Returns
    -------
    tuple of (list, np.ndarray, np.ndarray)
        - pep_indices : list of peptide residue indices (sorted).
        - rec_idx_topk : np.ndarray of shape (N_pep, top_k), 
                         the top_k receptor indices for each peptide residue.
        - dist_topk : np.ndarray of shape (N_pep, top_k),
                      the corresponding distances.
    """
    pep_indices = sorted(pep_coords_dict.keys())
    N_pep = len(pep_indices)

    rec_idx_topk = np.zeros((N_pep, top_k), dtype=int)
    dist_topk    = np.zeros((N_pep, top_k), dtype=float)

    # For each peptide residue, compute distances and pick the top_k
    for i, pep_idx in enumerate(pep_indices):
        pep_coords = pep_coords_dict[pep_idx]

        # Compute min dist to each receptor residue
        dist_list = []
        for rec_idx, rec_coords in rec_coords_dict.items():
            dist_matrix = cdist(rec_coords, pep_coords)
            min_dist = np.min(dist_matrix)
            dist_list.append((rec_idx, min_dist))

        # Sort ascending by distance
        dist_list.sort(key=lambda x: x[1])

        # Take top_k
        top_vals = dist_list[:top_k]


        for j, (r_idx, d_val) in enumerate(top_vals):
            rec_idx_topk[i, j] = r_idx
            dist_topk[i, j]    = d_val


    return pep_indices, rec_idx_topk, dist_topk


def process_seq_features(
    complex_inst,
    pep_indices,
    rec_idx_topk
):
    """
    Build two feature arrays:
      1) pep_feat  : shape (N_pep, 26) 
         (21-dim one-hot + 5-dim positional encoding)
      2) rec_feat  : shape (N_pep, top_k, 41)
         (21-dim receptor one-hot + 20-dim zero PSSM)

    Parameters
    ----------
    complex_inst : object
    pep_indices : list of int
        Sorted list of peptide residue indices.
    rec_idx_topk : np.ndarray of shape (N_pep, top_k)
        For each peptide residue (in pep_indices order),
        the top_k receptor residue indices.

    Returns
    -------
    pep_feat : np.ndarray, shape (N_pep, 26)
    rec_feat : np.ndarray, shape (N_pep, top_k, 41)
    """
    N_pep, top_k = rec_idx_topk.shape

    #------------------------------------------------------------------------
    # 1) Peptide features
    #------------------------------------------------------------------------
    # One-hot encoding => shape (N_pep, 21)
    pep_aa_list = []
    for pep_idx in pep_indices:
        aa_char = complex_inst.res_seq.get(pep_idx, 'X')
        pep_aa_list.append(aa_char)
    pep_seq_str = "".join(pep_aa_list)  
    pep_onehot = AA_one_hot(pep_seq_str)  

    
    #Positional encoding => shape (N_pep, 5)
    pos_enc = generate_pep_position_encoding(N_pep, d_model=5, max_len=40)

    # Concatenate => shape (N_pep, 26)
    pep_feat = np.concatenate([pep_onehot, pos_enc], axis=1)

    #------------------------------------------------------------------------
    # 2) Receptor features for the top_k residues
    #------------------------------------------------------------------------
    N_rec = complex_inst.rec_num  
    rec_aa_list = []
    for rec_i in range(N_rec):
        aa_char = complex_inst.res_seq.get(rec_i, 'X')
        rec_aa_list.append(aa_char)

    rec_seq_str = "".join(rec_aa_list)  # e.g. 'MAGD...'
    rec_onehot_all = AA_one_hot(rec_seq_str)
    

    # Gather top_k => shape (N_pep, top_k, 21)
    # Flatten rec_idx_topk -> 1D for advanced indexing
    rec_flat  = rec_idx_topk.flatten()                      # (N_pep*top_k,)
    
    rec_gather = rec_onehot_all[rec_flat]           
    rec_onehot_topk = rec_gather.reshape(N_pep, top_k, 21)  # (N_pep*top_k, 21)


    # Zero PSSM => shape (N_pep, top_k, 20) -in this model we don't use PSSM
    rec_pssm_zeros = np.zeros((N_pep, top_k, 20), dtype=int)

    # Concat => shape (N_pep, top_k, 41)
    rec_feat = np.concatenate([rec_onehot_topk, rec_pssm_zeros], axis=-1)
    

    return pep_feat, rec_feat


def process_esm_features(
    rec_esm_pt: str,
    pep_esm_pt: str,
    rec_idx_topk: np.ndarray,
    esm_layer: int = 33
):
    """
    Load receptor and peptide ESM embeddings

    Returns
    -------
    rec_esm_topk : np.ndarray, shape (N_pep, top_k, 1280)
        ESM embeddings for the receptor residues specified by rec_idx_topk.
    pep_esm_aligned : np.ndarray, shape (N_pep, 1280)
        ESM embeddings for the peptide residues in the same order as pep_indices.

    """

    # 1) Load receptor ESM from .pt
    rec_model = torch.load(rec_esm_pt)
    rec_esm_tensor = rec_model["representations"][esm_layer]
    rec_esm = rec_esm_tensor.detach().numpy()  

    # 2) Load peptide ESM from .pt
    pep_model = torch.load(pep_esm_pt)
    pep_esm_tensor = pep_model["representations"][esm_layer]
    pep_esm = pep_esm_tensor.detach().numpy()           # (N_pep, 1280)

    # 3) Gather top-k receptor ESM
    # rec_idx_topk => shape (N_pep, top_k)
    N_pep, top_k = rec_idx_topk.shape
    rec_flat = rec_idx_topk.flatten()             
    rec_esm_flat = rec_esm[rec_flat]              # shape => (N_pep*top_k, 1280)
    rec_esm_topk = rec_esm_flat.reshape(N_pep, top_k, 1280)

    # 4) peptide ESM
    pep_ranks = np.arange(N_pep)  # default order 
    pep_esm_aligned = pep_esm[pep_ranks]  # shape (N_pep, 1280)

    return rec_esm_topk, pep_esm_aligned



def process_confidence_features(
    pep_indices,       
    rec_idx_topk,      
    dist_topk,         
    npz_path: str
):
    """
    Build pAE, dist, distogram, rec_plddt, pep_plddt features for each peptide residue and its top_k receptor residues.

    Parameters
    ----------
    pep_indices  : [N_pep]
    rec_idx_topk : [N_pep, top_k]: For each peptide residue (row), the top_k receptor residue indices.
    dist_topk    : [N_pep, top_k]: distances between that peptide residue and top_k receptor residues.
    npz_path     : str - Path to the NPZ file with confidence data.


    Returns
    -------
    pAE_total      : [N_pep, top_k, 2]
    dist_expanded  : [N_pep, top_k, 1]
    distogram_topk : [N_pep, top_k, 64]
    rec_plddt_topk : [N_pep, top_k, 1]
    pep_plddt      : [N_pep, 1]
    """
    # 1) Load NPZ data
    data = np.load(npz_path)
    distogram = data["distogram"]               # shape (N_tot, N_tot, 64)
    pae       = data["predicted_aligned_error"] # shape (N_tot, N_tot)
    plddt     = data["plddt"]                   # shape (N_tot,)

    N_pep, top_k = rec_idx_topk.shape

    # 2) Expand dist_topk => shape (N_pep, top_k, 1)
    dist_expanded = dist_topk[..., np.newaxis]

    # 3) Build a companion 2D array for peptide indices => shape (N_pep, top_k)
    pep_indices = np.array(pep_indices)  
    pep_idx_2d  = np.tile(pep_indices.reshape(-1,1), (1, top_k))  # (N_pep, top_k)

    # Flatten them for vectorized indexing
    rec_flat = rec_idx_topk.flatten()  # (N_pep*top_k,)
    pep_flat = pep_idx_2d.flatten()    # (N_pep*top_k,)

    # 4) Distogram slices => shape (N_pep, top_k, 64)
    distogram_flat = distogram[rec_flat, pep_flat, :]  # (N_pep*top_k, 64)
    distogram_topk = distogram_flat.reshape(N_pep, top_k, distogram.shape[-1])


    # 5) pAE => shape (N_pep, top_k, 2)
    pae_1 = pae[rec_flat, pep_flat]
    pae_2 = pae[pep_flat, rec_flat]
    pae_concat = np.stack((pae_1, pae_2), axis=-1)  # (N_pep*top_k, 2)
    pAE_total  = pae_concat.reshape(N_pep, top_k, 2)

    # 6) Receptor pLDDT => shape (N_pep, top_k, 1)
    rec_plddt_flat = plddt[rec_flat]  # (N_pep*top_k,)
    rec_plddt_topk = rec_plddt_flat.reshape(N_pep, top_k, 1)


    # 7) Peptide pLDDT => shape (N_pep, 1)
    pep_plddt_array = plddt[pep_indices]  # (N_pep,)
    pep_plddt = pep_plddt_array.reshape(-1,1)


    return pAE_total, dist_expanded, distogram_topk, rec_plddt_topk, pep_plddt


def post_scaling(distance_data, pAE_data, pep_plddt_data, rec_plddt_data):
    distance_data = np.log1p(distance_data)
    pAE_data = np.log1p(pAE_data)
    pep_plddt_data = pep_plddt_data / 100
    rec_plddt_data = rec_plddt_data / 100

    return distance_data, pAE_data, pep_plddt_data, rec_plddt_data


def process_feature(pdb_path, npz_path, rec_esm_pt, pep_esm_pt):
    complex = PDB(pdb_path)
    complex.process_pdb()

    rec_d, pep_d = complex.dict_rec, complex.dict_pep
    # Compute top_k distances
    pep_indices, rec_idx_topk, dist_topk = compute_topk_dist(rec_d, pep_d, top_k=10)
    
    #1) compute seq features
    pep_seq_feat, rec_seq_feat = process_seq_features(complex, pep_indices, rec_idx_topk)
    #2) compute confidence features
    pAE_feat, dist_feat, distogram_feat, rec_plddt_feat, pep_plddt_feat = process_confidence_features(pep_indices, rec_idx_topk, dist_topk, npz_path)
    #3) compute esm features
    rec_esm_embedding, pep_esm_embedding = process_esm_features(rec_esm_pt, pep_esm_pt, rec_idx_topk, esm_layer=33)
    dist_embedding, pAE_embedding, pep_plddt_embedding, rec_plddt_embedding = post_scaling(dist_feat, pAE_feat, pep_plddt_feat, rec_plddt_feat)
    
    rec_node_feat = np.concatenate([rec_seq_feat, rec_plddt_embedding, rec_esm_embedding], axis = -1)
    pep_node_feat = np.concatenate([pep_seq_feat, pep_plddt_embedding, pep_esm_embedding], axis = -1)
    edge_feat = np.concatenate([dist_embedding, pAE_embedding, distogram_feat], axis = -1)

    
    return rec_node_feat, pep_node_feat, edge_feat




ROOT_DIR = "../../../example/pepscore"
pdb_path = Path(ROOT_DIR, "renum_model_complex.pdb")
npz_path = Path(ROOT_DIR, "model_confidence.npz")
rec_esm_pt = Path(ROOT_DIR, "rec_esm.pt")
pep_esm_pt = Path(ROOT_DIR, "pep_esm.pt")

rec_node_feat, pep_node_feat, edge_feat = process_feature(pdb_path, npz_path, rec_esm_pt, pep_esm_pt)
print("rec_node_feat", rec_node_feat.shape)
print("pep_node_feat", pep_node_feat.shape)
print("edge_feat", edge_feat.shape)

#save as rec_node_feat.npy
#save as pep_node_feat.npy
#save as edge_feat.npy

# rec_node_feat 저장
np.save(Path(ROOT_DIR, "rec_node_feat.npy"), rec_node_feat)

# pep_node_feat 저장
np.save(Path(ROOT_DIR, "pep_node_feat.npy"), pep_node_feat)

# edge_feat 저장
np.save(Path(ROOT_DIR, "edge_feat.npy"), edge_feat)

print("Features saved successfully!")