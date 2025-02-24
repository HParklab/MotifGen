#!/usr/bin/env python
import os,sys
import numpy as np
import torch
import time
from motifgen.SE3.wrapperC2 import * 
from motifgen.FullAtomNet import *


TYPES = list(range(14))
ntypes = len(TYPES)
Ymode = 'node'
n_l1out = 8 # assume important physical properties e.g. dipole, hydrophobicity et al
nGMM = 3 # num gaussian functions
NLAYERS=10
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

params_loader = {
          'shuffle': False,
          'num_workers': 8 if '-debug' not in sys.argv else 1,
          'pin_memory': True,
          'collate_fn': collate,
          'batch_size': 2 if '-debug' not in sys.argv else 1}

# default setup
set_params = {
    'ball_radius'  : 12.0,
    'edgemode'     : 'dist',
    "xyz_as_bb"    : True,
    #"upsample"     : upsample_category,
    "randomize"    : 0.2, # Ang, pert the rest
    "randomize_lig": 0.5, # Ang, pert the motif coord! #reduce noise...
    'debug'        : ('-debug' in sys.argv),
    'origin_as_node': (Ymode == 'node'),
    'inference': True
    }

accum       = 1
silent      = False

def load_params(rank, mode):
    if mode not in ['original', 'peptide']:
        raise ValueError("mode should be either original or peptide")
    
    model_name = "model.pkl" if mode == "original" else "model_pep.pkl"
    model_path = os.path.join(os.path.dirname(__file__), "../params", model_name)
    print("==mode===", mode)
    print("model_path", model_path)
    device = torch.device("cuda:%d"%rank if (torch.cuda.is_available()) else "cpu")
    print("model device", device)
    checkpoint = torch.load(model_path, map_location=device)
    
    model = SE3TransformerWrapper( num_layers=NLAYERS,
                                   l0_in_features=65+N_AATYPE+3, num_edge_features=2,
                                   l0_out_features=ntypes, #category only
                                   l1_out_features=n_l1out,
                                   ntypes=ntypes,
                                   drop_out=0.0,
                                   nGMM = nGMM )
    
    model.to(device)

    trained_dict = {}
    for key in checkpoint["model_state_dict"]:
        if key.startswith("module."):
            newkey = key[7:]
            trained_dict[newkey] = checkpoint["model_state_dict"][key]
        else:
            trained_dict[key] = checkpoint["model_state_dict"][key]
    model.load_state_dict(trained_dict)
    
    return model


                    
def get_preds(w, cs, Gsize):
    s = 0
    Ppreds = []
    for i,b in enumerate(Gsize):
        # ntype x 2
        wG = (w[s:s+b]/b)[:,:,None].repeat(1,1,2) # N x ntype
        cG = torch.transpose(cs[:,s:s+b,:],0,1) #N x ntype x 2
        q = torch.sum(wG*cG,dim=0) #node-mean; predict correct probability
        Qs = torch.nn.functional.softmax(q, dim=1)
        s += b
        Ppreds.append(np.array(Qs[:,0].cpu()))


    return np.array(Ppreds)


def main(model, trgs, prefix_dir, npzout=True):
    #update set_params 
    set_params['root_dir'] = prefix_dir
    runset = Dataset(trgs, **set_params)

    loader = data.DataLoader(runset,
                             worker_init_fn=lambda _: np.random.seed(),
                             **params_loader)
    
    with torch.no_grad(): 
        t0 = time.time()
        
        Pvalues = []
        grids = []
        
        # iter through grid points
        for G, node, edge, info in loader:
            if G == None:
                continue
            Gsize      = torch.tensor([v["numnode"] for v in info]).to(device)
            w,c,x,R = model(to_cuda(G, device), to_cuda(node, device), to_cuda(edge, device))
        
            P = get_preds(w["c"], c, Gsize)
            Pvalues.append(P)
            grids += [v['xyz'] for v in info]
            
        Pvalues = np.concatenate(Pvalues)
        grids = np.array(grids)
        
        if npzout == True:
            outnpz = os.path.join(prefix_dir, info[0]["pname"]+".score.npz")
            print("outnpz path", outnpz)
            if Pvalues.shape[0] != grids.shape[0]:
                print("Warning!! grid points and probability size don't match each other!")
            np.savez(outnpz, P=Pvalues, grids=grids)

        t1 = time.time()
        print("%s: Finished %d grid points in %.3f secs using batch=%d"%(info[0]["pname"],
                                                                         len(grids),t1-t0,
                                                                         params_loader['batch_size']))

                
#main
if __name__ == "__main__":
    infile = sys.argv[1]
    infile = os.path.abspath(infile)
    prefix_dir = os.path.dirname(infile)


    mode = sys.argv[2] #original or peptide

    if infile.endswith('.npz'):
        npzs = [infile]
    else:
        npzs = [l[:-1] for l in open(infile)] #line should be abspath of each npzs
        if not npzs[0].endswith('.npz'):
            sys.exit("input file should either end with '.npz' or contain a list of '.npz' files")
        
    model = load_params(0, mode)


    for npz in npzs:
        trgs = [npz.replace('.lig.npz','')+'.'+a  for a in np.load(npz,allow_pickle=True)['name']]

        print("Running ",npz,len(trgs))
        main(model, trgs, prefix_dir, npzout=True)

