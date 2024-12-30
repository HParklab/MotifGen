#!/usr/bin/env python
from SE3.wrapperC2 import * 

import os, sys
import numpy as np
from os import listdir
from os.path import join, isdir, isfile
import torch
from FullAtomNet import *
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

TYPES = list(range(14))
ntypes = len(TYPES)
Ymode = 'node'
n_l1out = 8 # assume important physical properties e.g. dipole, hydrophobicity et al
nGMM = 3 # num gaussian functions
dropout = 0.2
NLAYERS=10
w_reg = 1e-5
LR = 1.0e-4

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
    "upsample"     : upsample_category,
    "randomize"    : 0.2, # Ang, pert the rest
    "randomize_lig": 0.5, # Ang, pert the motif coord! #reduce noise...
    'debug'        : ('-debug' in sys.argv),
    'origin_as_node': (Ymode == 'node')
    }

accum       = 1
modelname   = 'original'
silent      = False

def load_params(rank):
    device = torch.device("cuda:%d"%rank if (torch.cuda.is_available()) else "cpu")
    
    model = SE3TransformerWrapper( num_layers=NLAYERS,
                                   l0_in_features=65+N_AATYPE+3, num_edge_features=2,
                                   l0_out_features=ntypes, #category only
                                   l1_out_features=n_l1out,
                                   ntypes=ntypes,
                                   drop_out=dropout,
                                   nGMM = nGMM )
    
    model.to(device)
    
    return model

def main(trgs, npzout=True, root_dir='./'):

    runset = Dataset(trgs, **set_params)
    set_params['root_dir'] = root_dir
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

            Pvalues.append(P)
            grids += [v['xyz'] for v in info]
            
        Pvalues = np.concatenate(Pvalues)
        grids = np.array(grids)
        
        if npzout == True:
            outnpz = info[0]["pname"]+".score.npz"
            
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
    if infile.endswith('.npz'):
        npzs = [sys.argv[1]]
    else:
        npzs = [l[:-1] for l in open(sys.argv[1])]
        if not npzs[0].endswith('.npz'):
            sys.exit("input file should either end with '.npz' or contain a list of '.npz' files")
        
    for npz in npzs:
        trgs = [npz.replace('.lig.npz','')+'.'+a  for a in np.load(npz,allow_pickle=True)['name']]
        print("Running ",npz,len(trgs))
        main(trgs, npzout=True)

