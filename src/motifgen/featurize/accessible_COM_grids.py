import numpy as np
import sys
import os
from scipy.spatial import cKDTree


# Parameters
params = {'dcut': 4.5, 'clash': 0.7, 'probe': 1.3}


def radii_by_elem(elems):
    """
    Returns a list of scaled atomic radii for the given elements.
    """
    atomic_radii = {
        "C": 2.0,   # Aliphatic carbons; default if unknown
        "N": 1.5,
        "O": 1.4,
        "S": 1.85,
        "H": 1.2,
        "F": 1.47,
        "Cl": 1.75,
        "Br": 1.85,
        "I": 2.0,
        "P": 1.8
    }
    return [0.9 * atomic_radii.get(e, 2.0) for e in elems]  # Default radius if element is unknown


def read_pdb(pdb_file, chain_allowed=None):
    """
    Reads a PDB file and returns element symbols, coordinates of the allowed chain,
    atom names, and coordinates of other chains.
    """
    xyz_chain = []
    xyz_other = []
    atms_other = []
    elems_other = []

    with open(pdb_file, 'r') as f:
        for line in f:
            if not line.startswith('ATOM'):
                continue
            if line[13] == 'H':
                continue  # Skip hydrogens
            chain = line[21]
            crd = [float(line[30:38]), float(line[38:46]), float(line[46:54])]

            if chain_allowed is None or chain == chain_allowed:
                xyz_chain.append(crd)
            else:
                atm = line[12:16].strip()
                if atm[0] not in 'CNSO':
                    continue
                atms_other.append(atm)
                elems_other.append(atm[0])
                xyz_other.append(crd)

    return elems_other, np.array(xyz_chain), atms_other, np.array(xyz_other)


def filter_by_contacts(xyzs, grids, padding=8.0, ncut=10, depth_cut=3.5):
    """
    Filters grids based on contacts with atoms to define an outer shell.
    """
    kd_grids = cKDTree(grids)
    kd_xyzs = cKDTree(xyzs)
    neighbors = kd_grids.query_ball_tree(kd_xyzs, padding)

    # Filter grids with sufficient atomic neighbors
    n_contacts = np.array([len(n) for n in neighbors])
    idx = np.where(n_contacts > 20)[0]
    grids = grids[idx]
    neighbors = [neighbors[i] for i in idx]

    # Compute distances from grids to the mean position of neighboring atoms
    vectors = np.array([np.mean(xyzs[n], axis=0) - g for n, g in zip(neighbors, grids)])
    distances = np.linalg.norm(vectors, axis=1)
    idx = np.where(distances < depth_cut)[0]
    grids = grids[idx]

    # Remove isolated grids
    kd_grids = cKDTree(grids)
    grid_neighbors = kd_grids.query_ball_tree(kd_grids, 4.0)
    idx = [i for i, n in enumerate(grid_neighbors) if len(n) > ncut]

    return grids[idx]


def filter_by_chain(grids, xyzs, distance):
    """
    Filters grids based on proximity to given coordinates within a certain distance.
    """
    kd_grids = cKDTree(grids)
    kd_xyzs = cKDTree(xyzs)
    indices = kd_xyzs.query_ball_tree(kd_grids, distance)
    indices = np.unique(np.concatenate(indices)).astype(int)
    return grids[indices]


def sasa_grids(xyz, elems, probe_radius=1.8, n_samples=50, d_clash=0.7):
    """
    Computes the Solvent Accessible Surface Area (SASA) grids.
    """
    atomic_radii = {
        "C": 2.0,
        "N": 1.5,
        "O": 1.4,
        "S": 1.85,
        "H": 0.0,
        "F": 1.47,
        "Cl": 1.75,
        "Br": 1.85,
        "I": 2.0,
        "P": 1.8
    }
    centers = xyz
    radii = np.array([0.9 * atomic_radii.get(e, 2.0) for e in elems])
    n_atoms = len(elems)

    # Generate points on a sphere using the Golden Section Spiral
    inc = np.pi * (3 - np.sqrt(5))
    off = 2.0 / n_samples
    pts0 = []
    for k in range(n_samples):
        y = k * off - 1 + (off / 2)
        r = np.sqrt(1 - y * y)
        phi = k * inc
        pts0.append([np.cos(phi) * r, y, np.sin(phi) * r])
    pts0 = np.array(pts0)

    kd_atoms = cKDTree(xyz)
    neighbors_list = kd_atoms.query_ball_tree(kd_atoms, 8.0)

    pts_out = []
    for i, (neighbors, center, radius) in enumerate(zip(neighbors_list, centers, radii)):
        if i in neighbors:
            neighbors.remove(i)
        n_neighbors = len(neighbors)

        # Generate sample points around the atom
        pts = pts0 * (radius + probe_radius) + center

        # Expand neighbor coordinates for distance calculations
        neighbor_coords = xyz[neighbors]
        r2_neighbors = (radii[neighbors] + probe_radius) ** 2

        # Determine points not overlapping with neighbors
        outsiders = []
        for pt in pts:
            d2 = np.sum((neighbor_coords - pt) ** 2, axis=1)
            if np.all(d2 >= r2_neighbors * 0.99):
                outsiders.append(pt)
        pts_out.extend(outsiders)

    pts_out = np.array(pts_out)

    # Remove overlapping points
    kd_pts = cKDTree(pts_out)
    clash_list = kd_pts.query_ball_tree(kd_pts, d_clash)
    incl = np.ones(len(pts_out), dtype=bool)

    for _ in range(5):
        nclash = np.array([len(clash) - 1 for clash in clash_list])
        if np.sum(nclash) == 0:
            break
        to_exclude = np.where(nclash > 0)[0]
        incl[to_exclude] = False
        pts_out = pts_out[incl]
        kd_pts = cKDTree(pts_out)
        clash_list = kd_pts.query_ball_tree(kd_pts, d_clash)
        incl = np.ones(len(pts_out), dtype=bool)

    grids = pts_out

    return grids


if __name__ == "__main__":
    infile = sys.argv[1]
    x_com = float(sys.argv[2])
    y_com = float(sys.argv[3])
    z_com = float(sys.argv[4])

    chain_allowed = "all"
    xyz_COM = np.array([[x_com, y_com, z_com]])

    chain_dcut = 12  # Fixed cutoff distance
    prefix = os.path.splitext(os.path.basename(infile))[0]

    print("prefix:", prefix)
    infile = os.path.abspath(infile)
    base_dir = os.path.dirname(infile)

    # Read PDB file
    elems, xyz_chain, atms, xyz_full = read_pdb(infile, chain_allowed=chain_allowed)
    print("Center of Mass coordinates:", xyz_COM)

    # Generate SASA grids
    grids = sasa_grids(xyz_full, elems, probe_radius=1.8,
                       n_samples=50, d_clash=params['clash'])
    grids = filter_by_contacts(xyz_full, grids, depth_cut=params['dcut'])

    # Save COM
    out_name2 = prefix + "_COM.pdb"
    with open(os.path.join(base_dir, out_name2), "w") as out:
        out.write("HETATM    1  CA  CA  X   1    %8.3f%8.3f%8.3f\n" % (xyz_COM[0, 0], xyz_COM[0, 1], xyz_COM[0, 2]))

    # Filter grids based on chain_dcut
    if len(xyz_full) != len(xyz_chain):  # If there are atoms from other chains
        grids = filter_by_chain(grids, xyz_COM, distance=chain_dcut)

    # Save grids to NPZ file
    npz_name = prefix + ".lig.npz"
    np.savez(os.path.join(base_dir, npz_name),
             name=['grid%04d' % i for i in range(len(grids))],
             xyz=grids)

    # Save grids in PDB format
    out_name4 = prefix + "_grid_%s.pdb" % chain_dcut
    with open(os.path.join(base_dir, out_name4), "w") as out:
        for i, x in enumerate(grids, 1):
            out.write("HETATM %4d  CA  UNK A %3d    %8.3f%8.3f%8.3f  1.00 %5.2f\n" %
                      (i, i, x[0], x[1], x[2], 0.52))