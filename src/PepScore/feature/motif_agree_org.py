import os, sys, glob
import numpy as np
import subprocess as sp
from scipy.spatial import distance_matrix
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
import myutils
from main_MotifNet import attach_H,reformat_pdb
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial import distance
sys.path.insert(0, "/home/seeun/works/MotifNet/hpark_script")
from peptide_motifs import main

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

def get_COM_peptide(pdb, pep_chain = "B"):
    pep_ca_R = []
    with open(pdb, "r") as fp:
        for line in fp:
            if not line.startswith("ATOM"):
                continue
            if line[21] == pep_chain:
                if line[12:16] == " CA ":
                    x = float(line[30:38].strip())
                    y = float(line[38:46].strip())
                    z = float(line[46:54].strip())
                    pep_ca_R.append([x,y,z])
    fp.close()

    pep_ca_R = np.array(pep_ca_R)
    COM_R = np.mean(pep_ca_R, axis = 0)
    return COM_R

def get_peptide(pdb, pep_chain = "B"):
    mul_pep_ca = []
    with open(pdb, "r") as fp:
        for line in fp:
            if not line.startswith("ATOM"):
                continue
            if line[21] == pep_chain:
                if line[12:16] == " CA ":
                    x = float(line[30:38].strip())
                    y = float(line[38:46].strip())
                    z = float(line[46:54].strip())
                    mul_pep_ca.append([x,y,z])
    fp.close()

    mul_pep_ca = np.array(mul_pep_ca)

    return mul_pep_ca

class Atom:
    def __init__(self,line):
        self.header=line[:6]
        self.atm_no = int(line[6:12].strip())
        self.atm_name = line[12:17].strip()
        self.res_name = line[17:20]
        self.res_no = int(line[22:26].strip())
        self.chain_ID = line[21:22]
        self.R = np.array([float(line[30:38].strip()), float(line[38:46].strip()),  float(line[46:54].strip())])

def calc_E(reformat_pdb):
    chain_A_coords = []
    chain_B_coords = []
    chain_A_res_nos = []
    chain_B_res_nos = []
    contact_pairs = {}
    count = 0

    with open(reformat_pdb, 'r') as fp:
        for line in fp:
            if not line.startswith('ATOM'):
                continue
            atom = Atom(line)
            if atom.chain_ID == "A":
                chain_A_coords.append(atom.R)
                chain_A_res_nos.append(atom.res_no)
            elif atom.chain_ID == "B":
                chain_B_coords.append(atom.R)
                chain_B_res_nos.append(atom.res_no)

    chain_A_coords = np.array(chain_A_coords)
    chain_B_coords = np.array(chain_B_coords)
    tree_B = cKDTree(chain_B_coords)

    for idx, atom_A in enumerate(chain_A_coords):
        results = tree_B.query_ball_point(atom_A, r=6.0)
        if results:
            res_no_A = chain_A_res_nos[idx]
            for res_idx in results:
                res_no_B = chain_B_res_nos[res_idx]
                if res_no_B not in contact_pairs.get(res_no_A, []):
                    count += 1
                    contact_pairs.setdefault(res_no_A, []).append(res_no_B)

    return count, contact_pairs


def get_rec_num(reformat_pdb):
    rec_num = []
    with open(reformat_pdb, 'r') as fp:
        for line in fp:
            if not line.startswith('ATOM'):
                continue
            atom = Atom(line)
            if atom.chain_ID == "A":
                if atom.res_no not in rec_num:
                    rec_num.append(atom.res_no)
    fp.close()

    #raise error if max(rec_num) != len(rec_num)
    if max(rec_num) != len(rec_num):
        raise ValueError("Error: max(rec_num) != len(rec_num)")
    return len(rec_num)

def calc_pae(pdbfilename, npz_name, contact_pairs): #directory_name):#number is ptm (-> PAE)
    rec_num = get_rec_num(pdbfilename)
    pdbname = pdbfilename.split('/')[-1]

    data = np.load(npz_name)
    lst = data.files
    pae = data['predicted_aligned_error']

    pae_score = 0
    pae_inter = 0
    pae_inter = 0.5 * (np.min(pae[:rec_num, rec_num:]) + np.min( pae[rec_num:, :rec_num]))

    
    for prot, pep_s in contact_pairs.items():
        for pep in pep_s:
            x = prot-1 # receptor residue number
            y = pep-1 + rec_num # peptide residue number
            mean_res = 0.5 * (1/ (1 + pae[x][y])  + 1/ (1+  pae[y][x]))
            pae_score += mean_res
    print('=====pae_score====', pae_score, '========pae_inter====', pae_inter)
    #pae_score: higher the better(TB)
    #pae_intrxn: lower the better(TB)

    return pae_score, pae_inter 

def get_pepdock_template(dir_path):
    templ_path = glob.glob(f"{dir_path}/TB/templ_re_????.pdb")
    if len(templ_path) != 1:
        print("Error: templ_re_????.pdb is not 1", templ_path, dir_path)
        sys.exit()

    return templ_path[0]

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
    #print("dist_sq", dist_sq)
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
    

    if option == sum:
        for point1 in set1:
            for point2 in set2:
                score += gaussian_kernel(point1, point2, sigma)
        print("score", score)
        return score
    
    elif option == max:
        for point2 in set2:
            max_score = 0
            for point1 in set1:
                score = gaussian_kernel(point1, point2, sigma)
                if score > max_score:
                    max_score = score
            score += max_score

    return score


    #using benchmark: motif_zscore is set to -1.0
def extract_motifs_from_manual(score_npz_path, motif_zscore = -1.0):
    ks = [1,2,3,4,5,6,7,8,9,11,12,13]
    SIMPLEMOTIFIDX = [0,
            2,3,3,
            1,6,1,
            2,3,
            5,4,4,4,4] #0:none 1:both, 2:acceptor 3:donor 4: Ali 5: Aro 6: Backbone

    final_pcut = "0.21 0.13 0.21 0.9  0.14 0.3  0.25 0.38 0.47 0.54 0.38 0.65"
    final_pcut =  np.array([float(x) for x in final_pcut.split()])

    std = [ 0.07, 0.03, 0.03, 0.15, 0.04, 0.11, 0.05, 0.06, 0.06, 0.10, 0.06, 0.12]

    #generate shifted pcut
    #make motif_zscore as array 
    motif_zscore = np.array([motif_zscore for _ in range(len(ks))])
    print("motif_zscore", motif_zscore)
    print("std", std)
    shifted_final_pcut = final_pcut + motif_zscore * std
    

    pred_threshold_dict = {k: shifted_final_pcut[idx] for idx, k in enumerate(ks)}

    prediction = np.load(score_npz_path)
    print("data", prediction.files) #['P', 'grids', 'xyz_bb', 'A_bb']

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


    for key in pred_each_dict.keys():
        pred_each_dict[key] = np.vstack(pred_each_dict[key])

    return pred_dict, n_grid 



def get_motifnet_info(motifnet_output_dict, pdb_motif_dict): 
    score_dict  = {}
    
    # Step 1: Calculate the sum scores for each motif type
    for motif_type in motifnet_output_dict.keys():
        answer_xyz = motifnet_output_dict[motif_type]
        print(f"Processing motif type: {motif_type}")
        if int(motif_type) not in pdb_motif_dict.keys():
            score_dict[f"{motif_type}_sum"] = 0
            continue
        pred_xyz = pdb_motif_dict[motif_type]
        #calc the dist matrix between answer_xyz and pred_xyz
        dist = distance_matrix(answer_xyz, pred_xyz)
        print("dist", dist)
        print(f"answer_xyz: {answer_xyz}, pred_xyz: {pred_xyz}")
        score = gaussian_overlap_score(answer_xyz, pred_xyz, option='sum', sigma=1.0)
        print("score", score)
        score_dict[f"{motif_type}_sum"] = score
    
    # Step 2: Generate scaled score (each score divided by the np.sqrt(number of answer_xyz))
    scaled_score_dict = {}
    for key in score_dict.keys():
        motif_type = key.split("_")[0]
        new_key = f"{motif_type}_scaled"
        if motif_type not in pdb_motif_dict:
            scaled_score_dict[new_key] = 0
            continue
        motif_num = len(motifnet_output_dict[motif_type])
        print(f"motif_num for {motif_type}: {motif_num}")
        scaled_score_dict[new_key] = score_dict[key] / np.sqrt(motif_num)
    
    # Merge scaled_score_dict into score_dict
    score_dict.update(scaled_score_dict)
    
    print("score_dict", score_dict)
    
    return score_dict

            
def get_pae_info(complex_pdb, rec_pdb, templ_COM):
    #Generate superposed model
    sup_TB_pdb = gen_superposed_model(complex_pdb, rec_pdb)
    mul_pep_ca = get_peptide(sup_TB_pdb, pep_chain="B")
    mul_dist = distance.cdist(templ_COM.reshape(1, -1), mul_pep_ca, metric='euclidean')
    
    # Calculate minimum distance and contact pairs (template similarity & interaction score)
    min_dist_mul = np.min(mul_dist)
    num_mul, contact_pairs = calc_E(sup_TB_pdb)

    # Calc PAE score and PAE inter
    npz_path = complex_pdb.replace("_unrelaxed_templ_tcoffee_all.pdb", "_info_templ_tcoffee_all_emb.npz")
    pae_score, pae_inter = calc_pae(sup_TB_pdb, npz_path, contact_pairs)
    print("pae_score", pae_score, "pae_inter", pae_inter)

    return min_dist_mul, num_mul, pae_score, pae_inter


receptor_pdb = sys.argv[1] #receptor.pdb #motifnet was runned with this pdb
motifnet_output = sys.argv[2] # motifpep_output.npz 
complex_pdb = sys.argv[3] #./renum_model_complex.pdb



new_test = ["Alpha_adaptinC2_1_PRM_0533", "Clathrin_propel+Clat_1_PRM_0516", "PDZ_1_PRM_0435", "PDZ_3_PRM_0372", "PDZ_6_PRM_0380", "VHS_1_PRM_0534"]
ROOT_DB = "/home/seeun/works/galaxydesign/example/"

for dir_name in new_test:
    is_type = "new_test"
    dir_path = os.path.join(ROOT_DB, dir_name)
    #Get Answer Motifs by running MotifNet
    #main_rec_MotifNet(os.path.join(ROOT_DB, "train", dir_name))
    score_npz_path = glob.glob(f"{ROOT_DB}/{dir_name}/TB/templ_re_*score_model_230.npz")
    if len(score_npz_path) != 1:
        print("Error: score_npz_path is not 1", score_npz_path, dir_name)
        continue
    score_npz_path = score_npz_path[0]
    answer_dict, n_grid = extract_motifs_from_manual(score_npz_path, -1) #key = grouped_pred_cat, value = pred_xyz


    # prepare 
    rec_pdb = os.path.join(dir_path, "receptor.pdb")
    TB_pdbs = glob.glob(f"{dir_path}/TB/out*_unrelaxed_templ_tcoffee_all.pdb")
    print("dir_path", dir_path)
    templ_pdb = get_pepdock_template(dir_path)
    templ_COM = get_COM_peptide(templ_pdb, pep_chain="B")

    csv_header = "rec_name,binder_type,pdb_name,min_dist_mul,num_mul,pae_score,pae_inter," + ",".join( list(map(str, answer_dict.keys())) )
    csv_rows = [csv_header]

    count = 0
    for TB_pdb in TB_pdbs:
        count += 1
        directory, filename = os.path.split(TB_pdb)
        rec_name = os.path.abspath(directory).split("/")[-2]; directory = os.path.basename(directory)  
        sup_TB_pdb = gen_superposed_model(TB_pdb, rec_pdb)
        # Extract xyzs and categories from the TB model (using atomic interactions)
        xyzs, cats = main(sup_TB_pdb, ligchain="B")
        print("xyzs", len(xyzs), "cats", len(cats))
        if len(xyzs) == 0:
            print("Error: No atomic interactions found in", sup_TB_pdb)
            continue
        

        # Build predicted model atomic interaction dictionary
        pred_model_dict = {}
        for cat, xyz in zip(cats, xyzs):
            pred_model_dict.setdefault(cat, []).append(xyz)

        # Calculate motifnet score
        score_dict = get_motifnet_info( answer_dict, pred_model_dict )
        print("score_dict", score_dict)
        sys.exit()
        if count > 20:
            break
        
    