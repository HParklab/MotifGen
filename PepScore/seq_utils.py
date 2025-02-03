import torch
import numpy as np




def AA_one_hot(data: str) -> np.ndarray:
    """
    One-hot encode a string of amino acids (1-letter code).
    
    Parameters
    ----------
    data : str
        A string of amino acids (e.g., 'A', 'ACD', etc.), 
        each character must be in 'ACDEFGHIKLMNPQRSTVWYX'.
    
    Returns
    -------
    np.ndarray
        Shape = (len(data), 21).
        One-hot matrix for the input string,
        where each row corresponds to one amino acid.
    """
    # Available amino acids: 21 characters (including 'X' for unknown)
    AA = 'ACDEFGHIKLMNPQRSTVWYX'
    aa_to_id = {c: i for i, c in enumerate(AA)}

    # Convert each character in 'data' to an index
    AA_id_input = [aa_to_id[aa] for aa in data]
    
    # Build one-hot vectors
    one_hot_list = []
    for idx in AA_id_input:
        vec = [0] * len(AA)
        vec[idx] = 1
        one_hot_list.append(vec)

    return np.array(one_hot_list, dtype=int)


def generate_pep_position_encoding(N_pep: int, d_model: int = 5, max_len: int = 40) -> np.ndarray:
    """
    Generate positional (sinusoidal) encoding for a peptide sequence.
    Dimension of encoding = 5.
    
    Parameters
    ----------
    N_pep : int
        Number of residues in the peptide sequence.
    
    Returns
    -------
    np.ndarray
        Shape = (N_pep, 5).
        Positional encoding for the first N_pep positions.
        
    Notes
    -----
    - This implementation only precomputes 40 rows.
      If N_pep > 40, an index error could occur. Adjust accordingly if needed.
    """
    # Precompute positional encodings up to 40
    # position_encoding shape: (40, 5)
    position_encoding = np.array([
        [pos / np.power(10000, 2.0 * (j // 2) / 5) for j in range(d_model)]
        for pos in range(max_len)
    ])
    # Apply sin to even indices, cos to odd indices
    position_encoding[:, 0::2] = np.sin(position_encoding[:, 0::2])
    position_encoding[:, 1::2] = np.cos(position_encoding[:, 1::2])
    
    # Take the first N_pep rows
    return position_encoding[:N_pep, :]


