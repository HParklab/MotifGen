import sys
import numpy as np
import os


receptor_pdb = sys.argv[1] #receptor.pdb #motifnet was runned with this pdb
motifnet_output = sys.argv[2] # motifpep_output.npz 
complex_pdb = sys.argv[3] #./renum_model_complex.pdb

def superpose_to(mdl,ref):
    tm_dat= os.popen(f'TMalign {mdl} {ref}').readlines() 
    #print('tm_dat', tm_dat)
    idx=0
    for line in tm_dat:
        if 'm          t(m)         u(m,1)         u(m,2)         u(m,3)' in line:
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

def gen_superposed_model(model_complex_pdb, rec_pdb):
    #CMD: TMscore TB_pdb rec_pdb -o superposed.pdb
    base_model_dir = os.path.dirname(model_complex_pdb)
    os.chdir(base_model_dir)

    #receptor structure of model_complex_pdb
    model_rec_pdb = model_complex_pdb.replace(".pdb", "_receptor_AF2.pdb")
    print("model_rec_pdb", model_rec_pdb)


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

    #reformat pdb
    reformat_sup_pdb = reformat_pdb(superposed_pdb)
    print("reformat_sup_pdb", reformat_sup_pdb)

    return reformat_sup_pdb


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
        print("pred_grid", pred_grid.shape, pred_grid)

        for ith in range(len(pred_grid)):
            if ith not in ks:
                continue
            pred_prob = pred_grid[ith]  #ith category prob of kth grid point
            pred_threshold = pred_threshold_dict[ith] 
            grouped_pred_cat = SIMPLEMOTIFIDX[ith]
            print("grouped_pred_cat", grouped_pred_cat)
            if pred_prob > pred_threshold:
                print("pred_prob: ", pred_prob, "pred_threshold: ", pred_threshold) 
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

    for key in pred_dict.keys():
        print("key: ", key, "pred_dict[key]", pred_dict[key].shape)
    return pred_dict, n_grid



extract_motifs_from_manual(motifnet_output, BEST_P_CUTS)
sup_model_pdb = gen_superposed_model(complex_pdb, receptor_pdb)

#CMD:  python get_motif_agreement_score.py ../../example/PepScore/receptor.pdb ../../example/PepScore/motifpep_output.npz ../../example/PepScore/renum_model_complex.pdb