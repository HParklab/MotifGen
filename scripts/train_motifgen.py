#!/usr/bin/env python
import os, sys
import numpy as np
from os import listdir
from os.path import join, isdir, isfile
import torch


from motifgen.SE3.wrapperC2 import * 
from motifgen.FullAtomNet import *

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
    'root_dir'     : #edit this line to your dataset path
    'ball_radius'  : 12.0,
    'edgemode'     : 'dist',
    "xyz_as_bb"    : True,
    "upsample"     : upsample_category,
    "randomize"    : 0.2, # Ang, pert the rest
    "randomize_lig": 0.5, # Ang, pert the motif coord! #reduce noise...
    'debug'        : ('-debug' in sys.argv),
    'origin_as_node': (Ymode == 'node')
    }

train_set = Dataset(np.load("data/train_datas.npy"), **set_params)
valid_set = Dataset(np.load("data/valid_datas.npy"), **set_params)
max_epochs  = 400
accum       = 1
retrain     = False
silent      = False

mode = sys.argv[1]

if mode == 'peptide':
    modelname   = 'peptide'
else:
    modelname   = 'original'

#freeze_schedule
# ~155: none
# 155~: 1~6
def freeze_params(model,rank):
    for name, param in model.named_parameters():
        if not name.startswith('se3'): continue
        i = int(name.split('.')[2])
        if i <= 10:
            if rank == 0:
                print("freeze", name, i)
            param.requires_grad = False

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
    
    train_loss = {"total":[], "bb":[], "ori":[], "cat":[], "grp":[], "reg":[]}
    valid_loss = {"total":[], "bb":[], "ori":[], "cat":[], "grp":[], "reg":[]}
    epoch = 0
    optimizer   = torch.optim.Adam(model.parameters(), lr=LR)
    
    if not retrain and os.path.exists("models/%s/model.pkl"%modelname): 
        if not silent: print("Loading a checkpoint")
        checkpoint = torch.load(join("models", modelname, "model.pkl"), map_location=device)

        trained_dict = {}
        for key in checkpoint["model_state_dict"]:
            if key.startswith("module."):
                newkey = key[7:]
                trained_dict[newkey] = checkpoint["model_state_dict"][key]
            else:
                trained_dict[key] = checkpoint["model_state_dict"][key]
        
        model.load_state_dict(trained_dict,strict=False)
        
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
        #freeze_params(model,rank) # freeze -- later stage

        epoch = checkpoint["epoch"]+1
        train_loss = checkpoint["train_loss"]
        valid_loss = checkpoint["valid_loss"]
        if not silent: print("Restarting at epoch", epoch)
        
    else:
        if not silent: print("Training a new model")
        best_models = []
        if not isdir(join("models", modelname)) and rank == 0:
            if not silent: print("Creating a new dir at", join("models", modelname))
            os.mkdir(join("models", modelname))

        # initialize here
        for i, (name, layer) in enumerate(model.named_modules()):
            if isinstance(layer,torch.nn.Linear) and \
               "WOblock" in name or 'SBblock' in name:
                #print(name,isinstance(layer,torch.nn.ModuleList))
                if isinstance(layer,torch.nn.ModuleList):
                    for l in layer:
                        if hasattr(l,'weight'): l.weight.data.fill_(0.0)
                else:
                    if hasattr(layer,'weight'):
                        layer.weight.data.fill_(0.0)

        # freeze
        for name, param in model.named_parameters():
            is_se3 = name.startswith("se3.graph_modules")
            if not is_se3: continue
        
            ilayer = int(name.split(".")[2])
        
            #if ilayer >= 16:
            #    print("freeze", name)
            #    param.requires_grad = False
                    
    print("loaded")
    return model, optimizer, epoch, train_loss, valid_loss

def is_same_simpleidx(label,idxs=np.arange(ntypes)):
    return torch.tensor([float(motif.SIMPLEMOTIFIDX[i]==motif.SIMPLEMOTIFIDX[label]) for i in idxs])

                    
def category_loss(cat, w, cs, Gsize, device, header=""):
    s = 0
    loss1 = torch.tensor(0.0)
    loss2 = torch.tensor(0.0)
    ps = []
    ts = []

    for i,b in enumerate(Gsize):
        # ntype x 2
        wG = (w[s:s+b]/b)[:,:,None].repeat(1,1,2) # N x ntype
        cG = torch.transpose(cs[:,s:s+b,:],0,1) #N x ntype x 2
        q = torch.sum(wG*cG,dim=0) #node-mean; predict correct probability
        Qs = torch.nn.functional.softmax(q, dim=1)

        ## label: P
        Ps = torch.tensor([float(k==cat[i]) for k in range(ntypes)]).repeat(2,1)
        Ps = torch.transpose(Ps,0,1).to(device)
        Ps[:,1] = 1.0-Ps[:,0]

        Ps_cat = is_same_simpleidx(cat[i]).to(device)
        Ps_cat = Ps_cat.repeat(2,1)
        Ps_cat = torch.transpose(Ps_cat,0,1)
        Ps_cat[:,1] = 1-Ps_cat[:,0]
        
        L = -Ps*torch.log(Qs+1.0e-6)
        L1 = torch.mean(L[:,0]) # ture
        L2 = torch.mean(L[:,1]) # false
        L = 0.5*L1 + 0.5*L2

        Lcat = -Ps_cat*torch.log(Qs+1.0e-6)/(sum(Ps_cat)+1.0e-6)
        Lcat1 = torch.mean(Lcat[:,0])
        Lcat2 = torch.mean(Lcat[:,1])
        Lcat = torch.mean(Lcat)

        print("C %s %2d %4.2f %6.3f %6.3f %6.3f %6.3f | "%(header[i],cat[i],Qs[cat[i],0],float(L1),float(L2),float(Lcat1),float(Lcat2))," %4.2f"*ntypes%tuple(Ps[:,0]),":"," %4.2f"*ntypes%tuple(Qs[:,0]))
        
        loss1 = loss1 + L
        loss2 = loss2 + Lcat
        s += b

    B  = len(Gsize)
        
    return loss1/B, loss2/B

def l1_loss_Y(truth, w, x, R, cat, Gsize, device, header='', mode='node_weight'):
    #w: nnode x ntype
    #v: nnode x 3
    s = 0
    loss = torch.tensor(0.0).to(device)
    R = R.to(device)

    n = 0
    for i,b in enumerate(Gsize):
        t = truth[i].float()
        vG = x[1][s:s+b] #l1 pred: N x channel x 3
        if cat[i] in TYPES and cat[i] != 0:
            n += 1
            icat = TYPES.index(cat[i])
            rot = R[icat] # 4x1 #weight on channel dim
            
            #wG = w[s:s+b,cat[i]]/b # mean
            #wG = torch.squeeze(wG)
            if mode == 'simple':
                wv = torch.transpose(torch.mean(torch.squeeze(vG),dim=0),0,1) # 3x32
                wv = torch.squeeze(rot(wv))
                
            elif mode == 'simpler':
                wv = torch.transpose(torch.mean(vG,dim=0),0,1) # 3x1
                
            elif mode == 'node_weight':
                norm = vG.shape[0]*vG.shape[1]
                wG = w[s:s+b]/norm # N x n1out
                wv = torch.einsum('ij,ijk->k', wG, vG)
                #wv = rot(wv)

            elif mode == 'node':
                wv = torch.transpose(torch.einsum('i,ij->ij',w[s],vG[0]),0,1)
                #print(w[s], vG[0], wv)
                wv = rot(wv)
                
            f1 = torch.nn.ReLU()
            f2 = torch.nn.MSELoss()
            # penalize only if smaller than 0.5
            #mag = f(torch.tensor(1.0).to(device),torch.sum(wv*wv))
            mag = f1(1.0-torch.sum(wv*wv))
            err = f2(wv,t)
        
            loss = loss + err + mag
            #print("Y %s %1d %2d %5.2f %5.2f | %8.3f %8.3f %8.3f : %8.3f %8.3f %8.3f"%(header[i],i,cat[i],err,mag,
            #                                                                          float(wv[0]),float(wv[1]),float(wv[2]),
            #                                                                          float(t[0]),float(t[1]),float(t[2])))
            s = s + b
            
    if n > 1: loss = loss/n
        
    return loss, n 

def l1_loss_B(truth, w, x, R, cat, Gsize, device, header="",mode='node_weight'):
    s = 0
    loss = torch.tensor(0.0).to(device)
    n = 0
    
    for i,b in enumerate(Gsize):
        t = truth[i].float()
        xG = x[0][s:s+b] #l0 pred: N x channel
        vG = x[1][s:s+b] #l1 pred: N x channel x 3

        if cat[i] in TYPES and cat[i] not in [0]: # skip null
            n += 1
            icat = TYPES.index(cat[i])
            rot = R['b'][icat].to(device) # 4 x nGMM
            sig = R['s'][icat].to(device) # 4 x nGMM
            
            #wG = w[s:s+b,cat[i]]/b # mean
            #wG = torch.squeeze(wG)
            if mode == 'simple':
                wv = torch.transpose(torch.mean(torch.squeeze(vG),dim=0),0,1) # 3x32
                wv = torch.squeeze(rot(wv))
                
            elif mode == 'simpler':
                wv = torch.transpose(torch.mean(vG,dim=0),0,1) # 3x1
                
            elif mode == 'node_weight':
                norm = vG.shape[0]*vG.shape[1]
                wG = w[s:s+b]/norm # N x n1out
                wv = torch.einsum('ij,ijk->k', wG, vG)
                #wv = rot(wv)

            elif mode == 'node':
                wv = torch.transpose(torch.einsum('i,ij->ij',w[s],vG[0]),0,1) # w[s]: n_l1out, vG: n_l1out x 3 -> n_1out x 3
                wv = torch.transpose(rot(wv),0,1) # convert per category; n1_out x nGMM
                
            f1 = torch.nn.ReLU()
            f2 = torch.nn.MSELoss()
            
            # penalize only if smaller than 0.5
            mag = f1(2.25-torch.sum(wv*wv)) # make at least 1.5 Ang (1bond apart)
            
            # is this the best? shouldn't it communicate with wv somehow or pertype-weight?
            sigma = sig(xG[0])+1.0 # sigma = torch.tensor([1.0 for k in range(nGMM)])
            
            dev = torch.mean((wv-t)*(wv-t),dim=1)
            #norm = 1.0/np.sqrt(2.0*np.pi)/sigma
            norm = 1.0/sigma
            P = norm*torch.exp(-dev/(sigma*sigma)) #>0
            dev = -torch.log(P+0.001)

            # allow negative? torch.sum(P)/3.0 ensures positive loss btw.
            err = -torch.log((torch.sum(P)+0.001)/3.0) # this will reward "converged Gaussian modes"...
            
            #reg = 0.1*(sigma*sigma-1.0)
            reg = 0.1*torch.mean(sigma*sigma-1.0,dim=0)
        
            loss = loss + torch.sum(err) + mag + reg
            try:
                l = "B %s: %1d %2d %5.2f %5.2f | s %5.2f %5.2f %5.2f e %5.2f %5.2f %5.2f t %6.2f %6.2f %6.2f "%(header[i],i,cat[i],err,mag,
                                                                                                                sigma[0], sigma[1], sigma[2],
                                                                                                                dev[0],dev[1],dev[2],
                                                                                                                float(t[0,0]),float(t[0,1]),float(t[0,2]))                                                          
                #for k in range(nGMM):
                #    l += " , %6.2f %6.2f %6.2f"%(float(wv[k,0]),float(wv[k,1]),float(wv[k,2]))
                #print(l)
            except:
                pass
                                     
            s = s + b
            
    if n > 1: loss = loss/n
        
    return loss

def train_model(rank, world_size, dumm):
    #start_epoch = epoch
    count = np.zeros(ntypes)
    gpu = rank % world_size
    dist.init_process_group(backend="gloo", world_size=world_size, rank=rank)
    
    device = torch.device("cuda:%d"%rank if (torch.cuda.is_available()) else "cpu")
    torch.cuda.set_device(device)
  
    model, optimizer, start_epoch, train_loss, valid_loss = load_params(rank)

    if rank == 0:
        print("Nparams:", count_parameters(model))

    ddp_model = DDP(model, device_ids=[gpu], find_unused_parameters=False)  
  
    train_sampler = data.distributed.DistributedSampler(train_set, num_replicas=world_size, rank=rank)
    train_loader = data.DataLoader(train_set, sampler=train_sampler, **params_loader)
  
    valid_sampler = data.distributed.DistributedSampler(valid_set, num_replicas=world_size, rank=rank)
    valid_loader = data.DataLoader(valid_set, sampler=valid_sampler, **params_loader)
  
    for epoch in range(start_epoch, max_epochs):
        ddp_model.train()
        temp_loss = train_an_epoch(ddp_model,optimizer,train_loader,rank,epoch,True)
        for k in train_loss:
            train_loss[k].append(np.array(temp_loss[k]))

        optimizer.zero_grad()

        ddp_model.eval()
        temp_loss = train_an_epoch(ddp_model,optimizer,valid_loader,rank,epoch,False)
        for k in valid_loss:
            valid_loss[k].append(np.array(temp_loss[k]))

        # Update the best model if necessary:
        if rank == 0:
            if np.min([np.mean(vl) for vl in valid_loss["total"]]) == np.mean(valid_loss["total"][-1]):
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': ddp_model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_loss': train_loss,
                    'valid_loss': valid_loss}, join("models", modelname, "best.pkl"))

            torch.save({
                'epoch': epoch,
                'model_state_dict': ddp_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'valid_loss': valid_loss}, join("models", modelname, "model.pkl"))
    
def train_an_epoch(ddp_model,optimizer,loader,rank,epoch,is_train):
    temp_loss = {"total":[], "bb":[], "ori":[], "cat":[], "grp":[], "reg":[]}
    b_count,e_count=0,0
    device = torch.device("cuda:%d"%rank if (torch.cuda.is_available()) else "cpu")
    
    for G, node, edge, info in loader:
        if G == None: continue
        with torch.cuda.amp.autocast(enabled=False):
            with ddp_model.no_sync():
                b_count += 1
                cat      = torch.tensor([v["motifidx"] for v in info]).to(device)
                yaxis     = torch.tensor([v["yaxis"] for v in info]).to(device) # y-vector
                bbxyz      = torch.tensor([v["dxyz"] for v in info]).to(device) # bb-vector
                Gsize      = torch.tensor([v["numnode"] for v in info]).to(device)

                try:
                    w,c,x,R = ddp_model(to_cuda(G, device), to_cuda(node, device), to_cuda(edge, device))
                except:
                    continue
        
                header = [v["pname"]+" %8.3f"*3%tuple(v["xyz"].squeeze()) for v in info]

                lossC,lossG = category_loss(cat, w['c'], c, Gsize, device, header)
                lossY,n  = l1_loss_Y(yaxis, w['y'], x, R['y'], cat, Gsize, device, header, mode=Ymode)
                lossB    = l1_loss_B(bbxyz, w['b'], x, R, cat, Gsize, device, header, mode=Ymode)
          
                loss = lossC + lossG + lossY+lossB
                if is_train:
                    l2_reg = torch.tensor(0.).to(device)
                    for param in ddp_model.parameters(): l2_reg += torch.norm(param)
                    loss = loss + w_reg*l2_reg 
                    temp_loss["reg"].append(l2_reg.cpu().detach().numpy()) #store as per-sample loss

                temp_loss["total"].append(loss.cpu().detach().numpy()) #store as per-sample loss
                temp_loss["cat"].append(lossC.cpu().detach().numpy()) #store as per-sample loss
                temp_loss["grp"].append(lossG.cpu().detach().numpy()) #store as per-sample loss
          
                if n > 0:
                    temp_loss["bb"].append(lossB.cpu().detach().numpy()) #store as per-sample loss
                    temp_loss["ori"].append(lossY.cpu().detach().numpy()) #store as per-sample loss
                    
            # Only update after certain number of accululations.
            if is_train and b_count%accum == 0:
                # accum grad here -- outside no_sync
                torch.nn.utils.clip_grad_norm_(ddp_model.parameters(), 0.25)
                loss.backward(retain_graph=True)
                
                optimizer.step()
                optimizer.zero_grad()
                print("TRAIN rank%d Epoch(%s): [%2d/%2d], Batch: [%2d/%2d], loss: %.3f, error %d"
                      %(rank, modelname, epoch, max_epochs, b_count, len(loader), np.sum(temp_loss["total"][-1*accum:]),e_count))

            
    return temp_loss
'''
from GPUtil import showUtilization as gpu_usage
from numba import cuda
def free_gpu_cache():
    print("Initial GPU Usage")
    gpu_usage()
    torch.cuda.empty_cache()
    cuda.select_device(0)
    cuda.close()
    cuda.select_device(0)
    print("GPU Usage after emptying the cache")
    gpu_usage()
'''    
#main
if __name__ == "__main__":
    torch.cuda.empty_cache()
    mp.freeze_support()
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    if ('MASTER_ADDR' not in os.environ):
      os.environ['MASTER_ADDR'] = 'localhost' # multinode requires this set in submit script
    if ('MASTER_PORT' not in os.environ):
      os.environ['MASTER_PORT'] = '12319'
    world_size = torch.cuda.device_count()

    os.system("touch GPU%d"%world_size)
    mp.spawn(train_model, args=(world_size,0), nprocs=world_size, join=True)
  
