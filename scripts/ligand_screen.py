import sys
import os
import copy
import torch
import torch.nn as nn
import numpy as np

import dgl
from dgl.nn import EGATConv

LR = 1.0e-4
device = torch.device("cuda" if (torch.cuda.is_available()) else "cpu")
DATADIR = '/ml/motifnet/MotifGen_output/PDBbind_refine/'
dropout_rate = 0.2
END_EPOCH = 300

ELEMS = ['Null','H','C','N','O','Cl','F','I','Br','P','S'] #0 index goes to "empty node"
    

def read_mol2(mol2,drop_H=True):
    read_cont = 0
    qs = []
    elems = []
    xyzs = []
    bonds = []
    borders = []
    atms = []
    
    for l in open(mol2):
        if l.startswith('@<TRIPOS>ATOM'):
            read_cont = 1
            continue
        if l.startswith('@<TRIPOS>BOND'):
            read_cont = 2
            continue
        if l.startswith('@<TRIPOS>SUBSTRUCTURE'):
            break
        if l.startswith('@<TRIPOS>UNITY_ATOM_ATTR'):
            read_cont = 0
            continue

        words = l[:-1].split()
        if read_cont == 1:

            idx = words[0]
            if words[1].startswith('BR'): words[1] = 'Br'
            if words[1].startswith('Br') or  words[1].startswith('Cl') :
                elem = words[1][:2]
            else:
                elem = words[1][0]

            if elem == 'A' or elem == 'B' :
                elem = words[5].split('.')[0]
            
            if elem not in ELEMS: elem = 'Null'
            
            atms.append(words[1])
            elems.append(elem)
            qs.append(float(words[-1]))
            xyzs.append([float(words[2]),float(words[3]),float(words[4])]) 
                
        elif read_cont == 2:
            bonds.append([int(words[1])-1,int(words[2])-1]) #make 0-index
            bondtypes = {'0':0,'1':1,'2':2,'3':3,'ar':3,'am':2, 'du':0, 'un':0}
            borders.append(bondtypes[words[3]])

    # drop hydrogens
    if drop_H:
        nonHid = [i for i,a in enumerate(elems) if a != 'H']
    else:
        nonHid = [i for i,a in enumerate(elems)]

    borders = [b for b,ij in zip(borders,bonds) if ij[0] in nonHid and ij[1] in nonHid]
    bonds = [[nonHid.index(i),nonHid.index(j)] for i,j in bonds if i in nonHid and j in nonHid]

    return np.array(elems)[nonHid], np.array(qs)[nonHid], bonds, borders, np.array(xyzs)[nonHid], np.array(atms)[nonHid]

def find_dist_neighbors(dX,top_k=8):
    D = torch.sqrt(torch.sum(dX**2, 3) + 1.0e-6)
    top_k_var = min(D.shape[1],top_k+1) # consider tiny ones
    D_neighbors, E_idx = torch.topk(D, top_k_var, dim=-1, largest=False)
    D_neighbor =  D_neighbors[:,:,1:]
    E_idx = E_idx[:,:,1:]
    u = torch.tensor(np.arange(E_idx.shape[1]))[:,None].repeat(1, E_idx.shape[2]).reshape(-1)
    v = E_idx[0,].reshape(-1)
    return u, v, D[0]

def normalize_distance(D,maxd=5.0):
    d0 = 0.5*maxd #center
    m = 5.0/d0 #slope
    feat = 1.0/(1.0+torch.exp(-m*(D-d0)))
    return feat

def generate_ligand_graph(mol2,
                          top_k=8):
    args = read_mol2(mol2)
    xyz, elems, bonds, borders = args[4], args[0], args[2], args[3]

    X = torch.tensor(xyz[None,]) #expand dimension
    dX = torch.unsqueeze(X,1) - torch.unsqueeze(X,2)
    u,v,d = find_dist_neighbors(dX,top_k=top_k)

    #print('lig',len(u),len(v))
    Glig = dgl.graph((u,v))

    ## 1. node features
    elems = [ELEMS.index(a) for a in elems]
    obt = np.eye(len(ELEMS))[elems] # 1hot encoded

    ## 2. edge features
    # 1-hot encode bond order
    ib = np.zeros((xyz.shape[0],xyz.shape[0]),dtype=int)
    bgraph = np.zeros((xyz.shape[0],xyz.shape[0]),dtype=int)
    for k,(i,j) in enumerate(bonds):
        ib[i,j] = ib[j,i] = k
        bgraph[i,j] = bgraph[j,i] = 1
    border_ = np.zeros(u.shape[0], dtype=np.int64)
    d_ = torch.zeros(u.shape[0])
    for k,(i,j) in enumerate(zip(u,v)):
        border_[k] = borders[ib[i,j]]
        d_[k] = d[i,j]
    
    edata = torch.eye(5)[border_] #0~3
    edata[:,-1] = normalize_distance(d_) #4
    
    Glig.ndata['attr'] = torch.tensor(obt).float()
    Glig.ndata['x'] = torch.tensor(xyz).float()[:,None,:]
    Glig.edata['attr'] = edata
    Glig.edata['rel_pos'] = dX[:,u,v].float()[0]
    
    return Glig

def generate_receptor_graph(npz):
    data = np.load(npz,allow_pickle=True)

    h_rec = data['P'] # N x 14
    xyz = torch.tensor(data['grids'])

    dX = torch.unsqueeze(xyz[None,],1) - torch.unsqueeze(xyz[None,],2)
    u, v, d = find_dist_neighbors(dX)
    Grec = dgl.graph((u,v))
    isolated_nodes = ((Grec.in_degrees() == 0) & (Grec.out_degrees() == 0)).nonzero().squeeze(1)
    #Grec.remove_node(isolated_nodes)

    #Grec = dgl.add_self_loop(Grec)
    
    Grec.ndata['attr'] = torch.tensor(h_rec).float()
    Grec.edata['attr'] = normalize_distance(d[u,v]).float()[:,None]
    
    return Grec

def collate(samples):
    return samples[0]

class DataSet(torch.utils.data.Dataset):
    def __init__(self, targets, datadir, nligs=5, random_sample=True):
        self.targets = targets
        self.datadir = datadir
        self.nligs = nligs
        self.random_sample = random_sample

    def __getitem__(self, index):
        label = torch.zeros(self.nligs)
        
        if self.random_sample: #for training
            target = self.targets[index]
            G_rec = generate_receptor_graph(self.datadir+'/'+target+'.score.npz')
        
            G_ligs = []
            G_ligs.append( generate_ligand_graph(self.datadir+'/'+target+'.ligand.mol2') )
        
            decoys = copy.deepcopy(self.targets)
            decoys.remove(target)
        
            decoys = np.random.choice(decoys, self.nligs-1)
            for tag in decoys:
                G_ligs.append( generate_ligand_graph(self.datadir+'/'+tag+'.ligand.mol2' ) )

            ligands = [target]+decoys
            label[0] = 1.0

        else: # for inference
            G_ligs = []
            decoys = self.targets[index*nligs:(index+1)*nligs]
            for tag in decoys:
                if tag.endswith('mol2'):
                    G_ligs.append( generate_ligand_graph(self.datadir+'/'+tag) )
                else:
                    G_ligs.append( generate_ligand_graph(self.datadir+'/'+tag+'.ligand.mol2' ) )
                    
            target = 'input'
            ligands = [l.split('/')[-1].replace('.mol2','') for l in decoys]
            
        info = {'label':label, 'target':target, 'ligands':ligands}
        b_Glig = dgl.batch(G_ligs)
        return G_rec, b_Glig, info

    def __len__(self):
        if self.random_sample:
            return len(self.targets)
        else:
            return int(len(self.targets)/self.nligs)

class Model(nn.Module):
    def __init__(self,
                 lig_in_features=11,
                 rec_in_features=14,
                 num_edge_features_lig=5,
                 num_edge_features_rec=1,
                 num_channels=32, n_heads=4, num_layers=3):
                           
        super().__init__()

        self.dropout = nn.Dropout(p=dropout_rate)
        self.norm = nn.InstanceNorm1d(num_channels)
        
        self.initial_linear_lig = nn.Linear(lig_in_features, num_channels)
        self.initial_linear_lig_edge = nn.Linear(num_edge_features_lig, num_channels)
        self.initial_linear_rec = nn.Linear(rec_in_features, num_channels)
        self.initial_linear_rec_edge = nn.Linear(num_edge_features_rec, num_channels)

        lig_layers, rec_layers = [],[]
        for i in range(num_layers):
            lig_layers.append(EGATConv(in_node_feats=num_channels,
                                       in_edge_feats=num_channels,
                                       out_node_feats=num_channels,
                                       out_edge_feats=num_channels,
                                       num_heads=n_heads))
            rec_layers.append(EGATConv(in_node_feats=num_channels,
                                       in_edge_feats=num_channels,
                                       out_node_feats=num_channels,
                                       out_edge_feats=num_channels,
                                       num_heads=n_heads))

        self.LigandEmbedder = nn.ModuleList( lig_layers )
        self.ReceptorEmbedder = nn.ModuleList( rec_layers )
        
        self.P1cut = nn.parameter.Parameter(torch.zeros(1)+0.1)
        self.P2cut = nn.parameter.Parameter(torch.zeros(1)+0.1)
        
    def forward(self, Grec, Glig, drop_out=True):
        # process ligand
        if drop_out:
            in_node_features = self.dropout(Glig.ndata['attr'])
            in_edge_features = self.dropout(Glig.edata['attr'])
        else:
            in_node_features = Glig.ndata['attr']
            in_edge_features = Glig.edata['attr']

        emb_lig = self.initial_linear_lig( in_node_features )
        edge_emb_lig = self.initial_linear_lig_edge( in_edge_features )
        
        # process rec
        if drop_out:
            in_node_features = self.dropout(Grec.ndata['attr'])
            in_edge_features = self.dropout(Grec.edata['attr'])
        else:
            in_node_features = Grec.ndata['attr']
            in_edge_features = Grec.edata['attr']

        emb_rec = self.initial_linear_rec( in_node_features )
        edge_emb_rec = self.initial_linear_rec_edge( in_edge_features )
        
        for layer_lig,layer_rec in zip(self.LigandEmbedder, self.ReceptorEmbedder):
            emb, edge_emb = layer_lig( Glig, emb_lig, edge_emb_lig )
            emb_lig = torch.nn.functional.elu(self.norm(emb.mean(1)))
            edge_emb_lig = torch.nn.functional.elu(self.norm(edge_emb.mean(1)))

            emb, edge_emb = layer_rec( Grec, emb_rec, edge_emb_rec )
            emb_rec = torch.nn.functional.elu(self.norm(emb.mean(1)))
            edge_emb_rec = torch.nn.functional.elu(self.norm(edge_emb.mean(1)))
            
        h_rec = emb_rec
        b = 0
        pred = []
        for n in Glig.batch_num_nodes():
            h_lig = emb_lig[b:b+n]

            attn = torch.einsum('ic,jc->ij', h_lig, h_rec)
            P1 = attn.softmax(dim=0) # attn over grids
            #P2 = attn.softmax(dim=1) # attn over ligand
            #P1[:,0].sum() == 1.0

            sig = torch.sigmoid((P1 - self.P1cut).sum()/50.0) #scaling factor
            pred.append(sig)
            b += n
        pred = torch.stack(pred)
        
        return pred

def load_model(modelname):
    model = Model()
    model.to(device)

    epoch = 0
    optimizer = torch.optim.Adam(model.parameters(),lr=LR)

    if os.path.exists("models/%s/model.pkl"%modelname):
        print("Loading a checkpoint")
        checkpoint = torch.load("models/"+modelname+"/model.pkl",map_location=device)

        trained_dict = model.state_dict()
        
        model.load_state_dict(trained_dict, strict=False)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        epoch = checkpoint["epoch"]+1
        train_loss = checkpoint["train_loss"]
        valid_loss = checkpoint["valid_loss"]
        print("Restarting at epoch", epoch)
        
    else:
        print("Training a new model")
        train_loss = {'total':[]}
        valid_loss = {'total':[]}
    
        if not os.path.isdir( os.path.join("models", modelname)):
            os.makedirs( os.path.join("models", modelname), exist_ok=True )

    print("Nparams:", sum(p.numel() for p in model.parameters() if p.requires_grad))

    return model,optimizer,epoch,train_loss,valid_loss
    
def train_one_epoch(model, optimizer, loader, epoch, is_train=False, verbose=True):
    losses = []
    for i, inputs in enumerate(loader):
        Grec, Glig, info = inputs
        Glig = Glig.to(device)
        Grec = Grec.to(device)
    
        target = info['target']
        label = info['label'].to(device)
        ligands = info['ligands']
        
        pred = model(Grec, Glig)
        func = torch.nn.CrossEntropyLoss()

        loss = func( pred, label )
        if verbose:
        if is_train:
            header = "%s Epoch %d TRAIN %5d/%5d"%(target,epoch,i,len(loader))
        else:
            header = "%s Epoch %d VALID %5d/%5d"%(target,epoch,i,len(loader))

        ligands = ' '.join(ligands)
            
        print(header, " %4.1f"*5%tuple(label)+" : "+" %6.3f"*5%tuple(pred), "loss %8.3f"%float(loss), ligands)

        if is_train:
            loss.requires_grad_(True)
            loss.backward()
            #torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
            optimizer.step()    
            optimizer.zero_grad()
        losses.append(loss.cpu().detach().numpy())
    return losses
   
def train_model(modelname):
    script_path = os.path.dirname(os.path.abspath(__file__))
    dataf_dir = script_path+'/../data/'
    
    params_loader={
        'shuffle': True,
        'num_workers':0,
        'pin_memory':True,
        'collate_fn':collate,
        'batch_size': 1}

    dataf_train = dataf_dir+'train.txt'
    trainlist = [l[:-1] for l in open(dataf_train)]
    train_set = DataSet(trainlist, datadir=DATADIR)
    train_loader = torch.utils.data.DataLoader(train_set, **params_loader)
    
    dataf_valid = dataf_dir+'valid.txt'
    validlist = [l[:-1] for l in open(dataf_valid)]
    valid_set = DataSet(validlist, datadir=DATADIR)
    valid_loader = torch.utils.data.DataLoader(valid_set, **params_loader)

    model,optimizer,start_epoch,train_loss,valid_loss = load_model(modelname)
    
    for epoch in range(start_epoch, END_EPOCH):
        tloss_tmp = train_one_epoch(model, optimizer, train_loader, epoch, is_train=True)
        vloss_tmp = train_one_epoch(model, optimizer, valid_loader, epoch, is_train=False)
        print("***EPOCH %d, TRAIN/VALID: "%epoch, np.mean(tloss_tmp), np.mean(vloss_tmp))

        train_loss['total'].append(tloss_tmp)
        valid_loss['total'].append(vloss_tmp)
        
        if np.min([np.mean(vl) for vl in valid_loss["total"]]) == np.mean(valid_loss["total"][-1]):
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'valid_loss': valid_loss,
            }, os.path.join("models", modelname, "best.pkl"))
   
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'valid_loss': valid_loss,
        }, os.path.join("models", modelname, "model.pkl"))

   
def infer_model(modelname, npz, mol2files):
    params_loader={
        'shuffle': True,
        'num_workers':0,
        'pin_memory':True,
        'collate_fn':collate,
        'batch_size': 1}

    infer_set = DataSet(mol2files, datadir='')
    data_loader = torch.utils.data.DataLoader(train_set, **params_loader)
    
    model,optimizer,start_epoch,train_loss,valid_loss = load_model(modelname)
    train_one_epoch(model, optimizer, data_loader, 0, is_train=False, verbose=True)
    
if __name__=="__main__":
    #modelname = sys.argv[1]
    #train_model(modelname)

    score_npz = sys.argv[1]
    mol2files = [l[:-1] for l in open(sys.argv[2])]
    infer_model("default", score_npz, mol2files)
    
