import os
import sys
import numpy as np

# Mapping from motif indices (0 to 13) to simplified motif group indices
# 0: none, 1: both, 2: acceptor, 3: donor, 4: Aliphatic, 5: Aromatic
SIMPLE_MOTIF_IDX = [
    0,  # Motif 0: none
    2,  # Motif 1: acceptor
    3,  # Motif 2: donor
    3,  # Motif 3: donor
    1,  # Motif 4: both
    1,  # Motif 5: both
    1,  # Motif 6: both
    2,  # Motif 7: acceptor
    3,  # Motif 8: donor
    5,  # Motif 9: Aromatic
    4,  # Motif 10: Aliphatic
    4,  # Motif 11: Aliphatic
    4,  # Motif 12: Aliphatic
    4   # Motif 13: Aliphatic
]

# Best cutoff probabilities for each motif index (as determined previously)
BEST_P_CUTS = {
    1: 0.32,
    2: 0.05,
    3: 0.3,
    4: 0.1,
    5: 0.1,
    6: 0.5,
    7: 0.9,
    8: 0.4,
    9: 0.48,
    11: 0.45,
    12: 0.35,
    13: 0.9
}

def draw_pdb(outprefix, coordinates, motifs):
    """
    Writes the predicted motifs to a PDB file.

    Args:
        outprefix (str): The prefix for the output file.
        coordinates (np.ndarray): Array of shape (N, 3) with the coordinates.
        motifs (np.ndarray): Array of length N with motif group indices.

    The PDB file will be named '{outprefix}.motif_pred.pdb'.
    """
    output_filename = f'{outprefix}.motif_pred.pdb'
    with open(output_filename, 'w') as outfile:
        # Atom names corresponding to motif groups
        atom_names = ['X', 'B', 'O', 'N', 'C', 'R']  # X: none, B: both, O: acceptor, N: donor, C: Aliphatic, R: Aromatic
        pdb_line_format = (
            "HETATM{atom_number:5d}  {atom_name:<3} UNK A{res_seq:4d}    "
            "{x:8.3f}{y:8.3f}{z:8.3f}  1.00 {occupancy:5.2f}\n"
        )

        for i in range(len(coordinates)):
            motif_group = motifs[i]
            if motif_group == 0:
                continue  # Skip if motif group is 'none'
            atom_name = atom_names[motif_group]
            x, y, z = coordinates[i]
            line = pdb_line_format.format(
                atom_number=i + 1,
                atom_name=atom_name,
                res_seq=i + 1,
                x=x,
                y=y,
                z=z,
                occupancy=1.00
            )
            outfile.write(line)

def main():
    # Get the prediction NPZ file from the command line arguments
    if len(sys.argv) < 2:
        print("Usage: python script_name.py prediction_file.npz")
        sys.exit(1)
    prediction_npz_file = sys.argv[1]

    # Load the predictions
    prediction_data = np.load(prediction_npz_file, allow_pickle=True)

    # Extract grids (coordinates) and P (probabilities)
    predicted_coordinates = prediction_data["grids"]  # Shape: (N, 3)
    predicted_probabilities = prediction_data["P"]    # Shape: (N, 14)

    # Initialize a dictionary to hold predictions for each motif group
    predicted_groups = {}

    # Number of predictions
    num_predictions = predicted_probabilities.shape[0]
    print(f"Processing {num_predictions} predictions")

    # Iterate over each prediction (grid point)
    for grid_idx in range(num_predictions):
        grid_probs = predicted_probabilities[grid_idx]  # Probabilities for each motif
        grid_coord = predicted_coordinates[grid_idx]    # Coordinate of the grid point

        # Iterate over each motif index
        for motif_idx in range(14):
            # Skip if the motif index is not in the cutoff thresholds
            if motif_idx not in BEST_P_CUTS:
                continue

            motif_prob = grid_probs[motif_idx]
            cutoff_prob = BEST_P_CUTS[motif_idx]

            # Check if the probability exceeds the cutoff
            if motif_prob > cutoff_prob:
                # Get the simplified motif group index
                group_idx = SIMPLE_MOTIF_IDX[motif_idx]

                # Initialize the list for the group index if it doesn't exist
                if group_idx not in predicted_groups:
                    predicted_groups[group_idx] = []

                # Append the coordinate to the group's list
                predicted_groups[group_idx].append(grid_coord)

                # After adding, remove duplicates from all groups
                # This step is crucial to match the original algorithm
                for key in predicted_groups.keys():
                    group_coords = np.array(predicted_groups[key])
                    unique_coords = np.unique(group_coords, axis=0)
                    predicted_groups[key] = unique_coords.tolist()

    # Collect all coordinates and motif group indices
    total_coordinates = []
    total_motifs = []

    for group_idx in predicted_groups.keys():
        coords = np.array(predicted_groups[group_idx])
        num_coords = coords.shape[0]
        motifs = [group_idx] * num_coords

        total_coordinates.extend(coords)
        total_motifs.extend(motifs)

    # Convert lists to numpy arrays
    total_coordinates = np.array(total_coordinates)
    total_motifs = np.array(total_motifs)

    # Write the predicted motifs to a PDB file
    output_prefix = os.path.splitext(prediction_npz_file)[0]
    draw_pdb(output_prefix, total_coordinates, total_motifs)

if __name__ == "__main__":
    main()
