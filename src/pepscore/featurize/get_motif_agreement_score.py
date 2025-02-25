import sys
import numpy as np
import os
from peptide_motifs import get_interaction_from_model

from scipy.spatial import distance_matrix

"""
receptor_pdb = sys.argv[1] #receptor.pdb #motifnet was runned with this pdb
motifnet_output = sys.argv[2] # motifpep_output.npz 
complex_pdb = sys.argv[3] #./renum_model_complex.pdb
"""
receptor_pdb = "/home/seeun/MotifGen/example/pepscore/receptor.pdb"
motifnet_output = "/home/seeun/MotifGen/example/pepscore/motifpep_output.npz"
complex_pdb = "/home/seeun/MotifGen/example/pepscore/renum_model_complex.pdb"

import subprocess as sp


class Atom:
    def __init__(self):
        pass
    def parsed_line(self, line):
        self.header=line[:6]
        self.atm_no = int(line[6:11])
        self.atm_name = line[12:16].strip()
        self.res_name = line[17:20]
        self.res_no = int(line[22:26].strip())
        self.chain_ID = line[21:22]
        self.R = np.array([float(line[30:38]), float(line[38:46]),  float(line[46:54])])
        

def get_contact(file_name):
    contact_pairs = {}
    #model_name = file_name.split('_')[0]
    ##pdb_name = model_name + '_unrelaxed.pdb'
    pdb_name = file_name
    

    chain_A = []
    chain_B = []
    count = 0
    with open(pdb_name, 'r') as fp:
        for line in fp:
            if not line.startswith('ATOM'):
                continue
            atom=Atom()
            atom.parsed_line(line)
            if atom.chain_ID == 'A':
                chain_A.append(atom)
            elif atom.chain_ID == 'B':
                chain_B.append(atom)


    #makes contact pair tuple
    contact_pair = []
    for atom_1 in chain_A:
        for atom_2 in chain_B:
            diff = np.subtract(atom_1.R, atom_2.R)
            distance = np.linalg.norm(diff)
            if distance <6:
                contact_pair.append((atom_1.res_no, atom_2.res_no))
    #make set
    contact_pairs = set(contact_pair)
    count = len(contact_pairs)
    print('contact_pairs', contact_pairs)
    return count


def superpose_to(mdl,ref):
    tm_dat = sp.run(['TMalign', mdl, ref, '-m', 'matrix.txt'], stdout=sp.PIPE).stdout.decode('utf-8').split('\n')
    with open("matrix.txt", "r") as fp:
        tm_dat = fp.readlines()

    #print('tm_dat', tm_dat)
    idx=0
    for line in tm_dat:
        if 'm               t[m]        u[m][0]        u[m][1]        u[m][2]' in line:
            idx = tm_dat.index(line)
    rot_matrix =  tm_dat[idx+1:idx+4]


    t= []
    u= []
    for line in rot_matrix:
        ls = line.strip().split()
        float_fig = []
        for fig in ls:
            float_fig.append(float(fig))
        t.append(float_fig[1])
        u.append(float_fig[2:])

    T = np.array(t,dtype=float)
    U = np.array(u,dtype=float)
    return T, U.T

from pathlib import Path
def gen_superposed_model(model_complex_pdb, rec_pdb):
    #receptor structure of model_complex_pdb
    model_rec_pdb = model_complex_pdb.replace(".pdb", "_receptor_AF2.pdb")

    wrt_rec = ""
    with open(model_complex_pdb, 'r') as fp:
        for line in fp:
            if not line.startswith("ATOM"):
                wrt_rec += line
                continue
            if line[21] == "A":
                wrt_rec += line
    fp.close()

    with open(model_rec_pdb, 'w') as fp:
        fp.write(wrt_rec)
    

    T_mul, trans_U_mul = superpose_to(model_rec_pdb, rec_pdb)

    wrt_sup = ""
    with open(model_complex_pdb, 'r') as fp:
        for line in fp:
            if not line.startswith("ATOM"):
                wrt_sup += line
                continue
            x = float(line[30:38].strip())
            y = float(line[38:46].strip())
            z = float(line[46:54].strip())
            R_coord = np.array([x, y, z])
            #generate superposed pdb

            L = T_mul + np.dot(R_coord, trans_U_mul)
            wrt_sup += line[:30] + "%8.3f%8.3f%8.3f"%(L[0], L[1], L[2]) + line[54:]
    fp.close()

    superposed_pdb = model_complex_pdb.replace(".pdb", "_sup.pdb")
    with open(superposed_pdb, 'w') as fp:
        fp.write(wrt_sup)
    fp.close()

    print("superposed_pdb", superposed_pdb)
    return superposed_pdb


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


def extract_motifs_from_manual(motifnet_output, best_p_cuts):
    ks = [1,2,3,4,5,6,7,8,9,11,12,13]
    SIMPLEMOTIFIDX = [0,
            2,3,3,
            1,6,1,
            2,3,
            5,4,4,4,4] #0:none 1:both, 2:acceptor 3:donor 4: Ali 5: Aro 6: Backbone

    pred_threshold_dict = best_p_cuts

    prediction = np.load(motifnet_output)
    pred_xyzs = prediction["grids"]
    pred_probs = prediction["P"]

    pred_dict = {} #key = grouped_pred_cat, value = pred_xyz
    n_grid = pred_probs.shape[0]


    pred_each_dict = {} #key: cat value: pred_xyz
    for kth in range(n_grid):
        #kth grid point
        pred_grid = pred_probs[kth]
        #print("pred_grid", pred_grid.shape, pred_grid)

        for ith in range(len(pred_grid)):
            if ith not in ks:
                continue
            pred_prob = pred_grid[ith]  #ith category prob of kth grid point
            pred_threshold = pred_threshold_dict[ith] 
            grouped_pred_cat = SIMPLEMOTIFIDX[ith]
            #print("grouped_pred_cat", grouped_pred_cat)
            if pred_prob > pred_threshold:
                #print("pred_prob: ", pred_prob, "pred_threshold: ", pred_threshold) 
                if grouped_pred_cat not in pred_dict:
                    pred_dict[grouped_pred_cat] = []
                pred_dict[grouped_pred_cat].append(pred_xyzs[kth])

                if ith not in pred_each_dict:
                    pred_each_dict[ith] = []
                pred_each_dict[ith].append(pred_xyzs[kth])
            
        
    for key in pred_dict.keys():
        pred_dict[key] = np.vstack(pred_dict[key])
        #print("pred_dict[key]", pred_dict[key])
        pred_dict[key] = np.unique(pred_dict[key], axis=0)

    #for key in pred_dict.keys():
    #    print("key: ", key, "pred_dict[key]", pred_dict[key].shape)
    return pred_dict, n_grid


def gaussian_kernel(x1, x2, sigma=1.0):
    """
    Compute the Gaussian kernel between x1 and x2:
    k(x, y) = exp(-||x - y||^2 / (2 * sigma^2))
    """
    #check if x1 and x2 are the np.array
    x1 = np.array(x1)
    x2 = np.array(x2)
    # Euclidean distance squared
    dist_sq = np.sum((x1 - x2) ** 2)
    # Gaussian kernel
    return np.exp(-dist_sq / (2 * sigma ** 2))

def gaussian_overlap_score(set1, set2, option = sum, sigma=1.0):
    """
    Compute the Gaussian overlap score between two sets of 3D points.
    #set 1-> answer_xyz
    #set 2 -> pred_xyz
    #can be two way -> first) only find the closest answer_xyz (option = max)
    #second -> include all answer_xyz (option = sum)
        
    for point1 in set1:
        for point2 in set2:
            score += gaussian_kernel(point1, point2, sigma)
                        
    """
    score = 0.0
    # Pairwise Gaussian kernel evaluations between all points in set1 and set2

    print("===============================================")
    print("Num answer_xyz", len(set1))
    print("Num pred_xyz", len(set2))
    print("===============================================")
    

    if option =="sum":
        for point1 in set1:
            for point2 in set2:
                score += gaussian_kernel(point1, point2, sigma)
        return score
    

    return score


def get_motifnet_info(motifnet_output_dict, pdb_motif_dict): 
    score_dict  = {}
    
    # Step 1: Calculate the sum scores for each motif type
    for motif_type in motifnet_output_dict.keys():
        answer_xyz = motifnet_output_dict[motif_type]
        if int(motif_type) not in pdb_motif_dict.keys():
            score_dict[f"{motif_type}_sum"] = 0
            continue
        print(f"Processing motif type: {motif_type}")
        pred_xyz = pdb_motif_dict[motif_type]
        #calc the dist matrix between answer_xyz and pred_xyz
        dist = distance_matrix(answer_xyz, pred_xyz)
        #for each pred_xyz  -> find the closest dist
        for i in range(len(pred_xyz)):
            min_dist = np.min(dist[:, i])
            print("min_dist", min_dist)
        score = gaussian_overlap_score(answer_xyz, pred_xyz, option='sum', sigma=1.0)
        score_dict[f"{motif_type}_sum"] = score
    
    print("score_dict sum", score_dict)
    # Step 2: Generate scaled score (each score divided by the np.sqrt(number of answer_xyz))
    scaled_score_dict = {}
    for key in score_dict.keys():
        motif_type = key.split("_")[0]
        new_key = f"{motif_type}_scaled"
        if int(motif_type) not in pdb_motif_dict:
            scaled_score_dict[new_key] = 0
            print("==continued==", motif_type)
            continue
        
        motif_num = len(motifnet_output_dict[int(motif_type)])
        print(f"motif_num for {motif_type}: {motif_num}")
        scaled_score_dict[new_key] = score_dict[key] / np.sqrt(motif_num)
    
    print("scaled_score_dict", scaled_score_dict)
    # Merge scaled_score_dict into score_dict
    score_dict.update(scaled_score_dict)
    
    #only get the "1_scaled", "2_scaled", "3_scaled", "4_scaled", "5_scaled" concatenated array
    motifagreement_score = np.array([score_dict[f"{i}_scaled"] for i in range(1, 6)])
    return motifagreement_score

#motifpep prediction
motif_pred, n_grid = extract_motifs_from_manual(motifnet_output, BEST_P_CUTS)

sup_model_pdb = gen_superposed_model(complex_pdb, receptor_pdb)
#first get the interactions from model pdb
xyzs, cats = get_interaction_from_model(sup_model_pdb, ligchain="B") #from main
pred_model_dict = {}
for cat, xyz in zip(cats, xyzs):
    pred_model_dict.setdefault(cat, []).append(xyz)

#compare motif_pred and xyzs -> get motifagreement_score
score_dict = get_motifnet_info(motif_pred, pred_model_dict )

#save as motif.npy
num_contact = get_contact(complex_pdb)
print("num_contact", num_contact)

#add num_contact to score_dict as a array
num_contact  = np.array([num_contact])
score_dict = np.concatenate((score_dict, num_contact))

#save as motif.npy order 1_scaled, 2_scaled, 3_scaled, 4_scaled, 5_scaled, num_contact
np.save("motif.npy", score_dict)

#load and check
loaded_score = np.load("motif.npy", allow_pickle=True)
print("loaded_score", loaded_score)
