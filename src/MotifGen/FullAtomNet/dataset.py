import sys,os
import numpy as np
import torch
import dgl
from torch.utils import data
from os import listdir
from os.path import join, isdir, isfile
from scipy.spatial import distance, distance_matrix
import scipy

from . import myutils
from . import motif

class Dataset(torch.utils.data.Dataset):
    'Characterizes a dataset for PyTorch'
    def __init__(self,
                 targets,
                 root_dir        = "",
                 verbose         = False,
                 ball_radius     = 12.0,
                 displacement    = "",
                 randomize_lig   = 0.5,
                 randomize       = 0.0,
                 tag_substr      = [''],
                 upsample        = None,
                 num_channels    = 32,
                 sasa_method     = "sasa",
                 edgemode        = 'distT',
                 edgek           = (0,0),
                 edgedist        = (10.0,6.0),
                 ballmode        = 'all',
                 distance_feat   = 'std',
                 debug           = False,
                 nsamples_per_p  = 1,
                 CBonly          = False,
                 xyz_as_bb       = False, 
                 sample_mode     = 'random',
                 use_l1          = False,
                 origin_as_node  = False,
                 inference       = False,
    ):
        self.proteins = targets
        
        self.datadir = root_dir
        self.verbose = verbose
        self.ball_radius = ball_radius
        self.randomize_lig = randomize_lig
        self.randomize = randomize
        self.tag_substr = tag_substr
        self.sasa_method = sasa_method
        self.num_channels = num_channels
        self.ballmode = ballmode #["com","all"]
        self.edgemode = edgemode
        self.edgedist = edgedist
        self.edgek = edgek
        self.inference = inference
        self.dist_fn_atm = get_dist_neighbors
        self.debug = debug
        self.distance_feat = distance_feat
        self.nsamples_per_p = nsamples_per_p
        self.xyz_as_bb = xyz_as_bb
        self.sample_mode = sample_mode
        self.CBonly = CBonly
        self.use_l1 = use_l1
        self.origin_as_node = origin_as_node

        if upsample == None:
            self.upsample = sample_uniform
        else:
            self.upsample = upsample

        self.nsamples = max(1,len(self.proteins)*nsamples_per_p)
        
    def __len__(self):
        'Denotes the total number of samples'
        return int(self.nsamples)
    
    def __getitem__(self, index):
        # Select a sample decoy
        info = {}
        info['sname'] = 'none'
        
        skip_this = False
        
        # "proteins" in a format of "protein.idx"
        ip = int(index/self.nsamples_per_p) #==index
        
        pname = '.'.join(self.proteins[ip].split('/')[-1].split('.')[:-1])

        fname = self.datadir+pname+'.lig.npz'
        if not os.path.exists(fname):
             skip_this = True
             cats = []
        else:
            samples = np.load(fname,allow_pickle=True) #motif type & crd only
            if not self.inference:
                if ('cat' not in samples or len(samples['cat']) == 0):
                    skip_this = True
                else:
                    cats = samples['cat']
                    pindices = np.arange(len(samples['name']))
                    pindex = np.random.choice(pindices,p=self.upsample(cats))
            else:
                pindex = ip

        if skip_this:
            info['pname'] = 'none'
            print("failed to read input npz %s"%fname)
            return False, info

        #print(self.proteins[ip], pname, pindex, len(samples['name']))
        info['pindex'] = pindex
        info['pname'] = pname
        info['sname'] = samples['name'][pindex]

        # training-only labels
        rot_motif,motifidx,motif_y = None,-1,np.zeros(3)
        if 'bases' in samples:
            info['bases'] = samples['bases'][pindex]
            motif_y = samples['bases'][pindex][1] # functional group's connectivity
            
        if 'rot' in samples: rot_motif   = samples['rot'][pindex] #quaternion"s" (n,4)
        if 'cat' in samples: motifidx    = samples['cat'][pindex] #integer

        xyz_motif  = samples['xyz'][pindex] #vector; set motif position at the origin
        if self.xyz_as_bb:
            if 'xyz_bb' in samples:
                xyz_pred   = samples['xyz_bb'][pindex][1] #vector
            else:
                #sys.exit("xyz_bb requested but not exist in npz, failed!")
                xyz_pred = xyz_motif
        else:
            xyz_pred   = xyz_motif

        # receptor features that go into se3        
        prop = np.load(self.datadir+pname+".prop.npz")
        xyz         = prop['xyz_rec']
        charges_rec = prop['charge_rec'] 
        atypes_rec  = prop['atypes_rec'] #1-hot
        anames      = prop['atmnames']
        aas_rec     = prop['aas_rec'] #rec only

        #mask neighbor res
        if 'exclude' in samples:
            reschains   = [a[0] for a in prop['atmres_rec']]
            exclrc = samples['exclude'][pindex]
            if len(exclrc) > 0: 
                iexcl = np.array([i for i,rc in enumerate(reschains) if rc in exclrc],dtype=int)
                # send away excl atms not to screw up indexing...
                xyz[iexcl] += 100.0

        # report
        if self.debug:
            self.report_xyz(pname+".%d.xyz"%pindex, atypes_rec, xyz, origin=xyz_motif)
        
        sasa_rec = prop['sasa_rec']
        
        # bond properties
        bnds    = prop['bnds_rec']
        atypes  = np.array([myutils.find_gentype2num(at) for at in atypes_rec]) # string to integers
            
        # randomize motif coordinate
        dxyz = np.zeros(3)
        if self.randomize_lig > 1e-3:
            dxyz = 2.0*self.randomize_lig*(0.5 - np.random.rand(3))

        # orient around motif + delta
        xyz = xyz - xyz_motif + dxyz[None,:]
        xyz_pred = xyz_pred - xyz_motif + dxyz[None,:] #==dxyz if not-bb

        # randomize the rest coordinate
        if self.randomize > 1e-3:
            randxyz = 2.0*self.randomize*(0.5 - np.random.rand(len(xyz),3))
            xyz = xyz + randxyz

        d2o = np.sqrt(np.sum(xyz*xyz,axis=1))

        center_xyz = np.zeros((1,3))
        ball_xyzs = [center_xyz] #origin

        ## append "virtual" node at the 0-th place
        if self.origin_as_node:
            xyz = np.concatenate([ball_xyzs[0],xyz])
            anames = np.concatenate([['origin'],anames])
            aas_rec = np.concatenate([[0], aas_rec]) # unk
            atypes = np.concatenate([[0],atypes]) #null
            sasa_rec = np.concatenate([[0.0],sasa_rec]) # 
            charges_rec = np.concatenate([[0.0],charges_rec])
            d2o = np.concatenate([[0.0], d2o])

        aas1hot = np.eye(myutils.N_AATYPE)[aas_rec]
        atypes  = np.eye(max(myutils.gentype2num.values())+1)[atypes] # convert integer to 1-hot
        sasa    = np.expand_dims(sasa_rec,axis=1)
        charges = np.expand_dims(charges_rec,axis=1)
        d2o = d2o[:,None]
        
        try:
            G_atm = self.make_atm_graphs(xyz, ball_xyzs,
                                         [aas1hot,atypes,sasa,charges,d2o],
                                         bnds, anames, self.CBonly, self.use_l1)

        except:
            G_atm  = self.make_atm_graphs(xyz, ball_xyzs,
                                          [aas1hot,atypes,sasa,charges],
                                          bnds, anames, self.CBonly, self.use_l1)
            print("graphgen fail")
            return False, info

        if G_atm.number_of_nodes() > 500:
            return False, info
            
        info['dxyz']  = xyz_pred 
        info['xyz']   = xyz_motif # where the origin is set
        info['yaxis'] = motif_y  # where the motif points toward
        info['rot']   = rot_motif # may potentially perturb later
        info['numnode'] =  G_atm.number_of_nodes()

        # for testing w/o answer
        info['motifidx']   = motifidx

        # Remove self edges
        G_atm = myutils.remove_self_edges(G_atm)
        
        return G_atm, info

    def report_xyz(self,outname, atypes, xyz, origin=[0,0,0]):
        out = open(outname,'w')
        form = '%1s %8.3f %8.3f %8.3f\n'
        out.write(form%("F",origin[0],origin[1],origin[2]))
        for i,a in enumerate(atypes):
            out.write(form%(a[0],xyz[i,0],xyz[i,1],xyz[i,2]))
        out.close()
            
    def get_a_sample(self,pname,index):
        pindices = []

        samples = np.load(self.datadir+pname+".lig.npz",allow_pickle=True)
        for substr in self.tag_substr:
            pindices += [i for i,n in enumerate(samples['name']) if substr in n]
        fnats = np.array([samples['fnat'][i] for i in pindices])

        if self.sample_mode == 'random':
            pindex  = np.random.choice(pindices,p=self.upsample(fnats))
        elif self.sample_mode == 'serial':
            pindex = index%len(pindices)
        
        return samples, pindex

    def make_res_graph(self, xyz, center_xyz, obt_fs, repsatm_idx, rsds_in_Gatm):
        # get xyz that goes into Gres
        xyz_reps = xyz[repsatm_idx]
        xyz_reps = torch.tensor(xyz_reps-center_xyz).float()

        # Grabbing a < dist neighbor
        kd      = scipy.spatial.cKDTree(xyz_reps)
        kd_ca   = scipy.spatial.cKDTree(center_xyz)

        indices = kd_ca.query_ball_tree(kd, 30.0)[0] # residue idx
        reps_idx = [repsatm_idx[i] for i in indices] # atm idx

        # which idx in reps_idx map to idx_in_Gatm; i.e. what becomes residue-embedding in G_atm
        # rsds_in_Gatm: resno following idx_ord
        r2amap = np.zeros(len(rsds_in_Gatm), dtype=int)
        for i,rsd in enumerate(rsds_in_Gatm):
            if rsd in indices: r2amap[i] = indices.index(rsd)
            else: sys.exit("unexpected resno %d"%(rsd)) #should not happen
        # many entries unused in Gatm may left as 0
        r2amap = np.eye(len(indices))[r2amap]

        # renew xyz on picked indices
        xyz_reps = xyz[indices]
        xyz_reps = torch.tensor(xyz_reps-center_xyz).float()
        
        obt  = []
        for f in obt_fs:
            if len(f) > 0: obt.append(f[reps_idx])
        obt = np.concatenate(obt,axis=-1)

        u,v = self.dist_fn_res(xyz_reps[None,])
        D = torch.sqrt(torch.sum((xyz_reps[v] - xyz_reps[u])**2, axis=-1)+1e-6)[...,None]
        D1hot = distance_feature(self.distance_feat,D,1.0,10.0)

        G_res = dgl.graph((u,v))
        G_res.ndata['0'] = torch.tensor(obt).float()
        G_res.ndata['x'] = xyz_reps[:,None,:]
        G_res.edata['d'] = xyz_reps[v] - xyz_reps[u]
        G_res.edata['w'] = D1hot
        
        return G_res, r2amap

    def make_atm_graphs(self, xyz, ball_xyzs, obt_fs, bnds, atmnames, CBonly=False, use_l1=False ):
        # Do KD-ball neighbour search -- grabbing a <dist neighbor from ligcom
        kd      = scipy.spatial.cKDTree(xyz)
        indices = []
        for ball_xyz in ball_xyzs:
            kd_ca   = scipy.spatial.cKDTree(ball_xyz)
            indices += kd_ca.query_ball_tree(kd, self.ball_radius)[0]
        idx_ord = list(np.unique(indices))
        if CBonly:
            idx_ord = [i for i in idx_ord if atmnames[i] in ['N','CA','C','O','CB']]

        # concatenate all one-body-features
        obt  = []
        for i,f in enumerate(obt_fs):
            if len(f) > 0: obt.append(f[idx_ord])
        obt = np.concatenate(obt,axis=-1)

        xyz     = xyz[idx_ord]
            
        bnds    = [bnd for bnd in bnds if (bnd[0] in idx_ord) and (bnd[1] in idx_ord)]
        bnds_bin = np.zeros((len(xyz),len(xyz)))
        for i,j in bnds:
            k,l = idx_ord.index(i),idx_ord.index(j)
            bnds_bin[k,l] = bnds_bin[l,k] = 1
        for i in range(len(xyz)): bnds_bin[i,i] = 1 #self
        
        # Concatenate coord & centralize xyz to ca.
        xyz = torch.tensor(xyz).float()
        
        ## Graph connection: u,v are nedges & pairs to each other; i.e. (u[i],v[i]) are edges for every i
        # for G_atm
        u,v,dX = self.dist_fn_atm(xyz[None,], mode=self.edgemode, top_k=self.edgek[1],
                                  dcut=self.edgedist[1])

        #if self.bndgraph_type == 'bonded':
        ub,vb = np.where(bnds_bin)
            
        # distance
        w = torch.sqrt(torch.sum((xyz[v] - xyz[u])**2, axis=-1)+1e-6)[...,None]
        w1hot = distance_feature(self.distance_feat,w,0.5,5.0)
        bnds_bin = torch.tensor(bnds_bin[v,u]).float() #replace first bin (0.0~0.5 Ang) to bond info
        w1hot[:,0] = bnds_bin

        ## Construct graphs
        # G_atm: graph for all atms
        G_atm = dgl.graph((u,v))
        G_atm.ndata['attr'] = torch.tensor(obt).float() 
        G_atm.edata['attr'] = w1hot #;  'd' previously
        G_atm.edata['rel_pos'] = dX[:,u,v].float()[0]

        if use_l1:
            G_atm.ndata['x'] = xyz[:,None,:] # unused?

        return G_atm
    
# Given a list of coordinates X, gets top-k neighbours based on eucledian distance
def get_dist_neighbors(X, mode="topk", top_k=16, dcut=4.5, eps=1E-6):
    """ Pairwise euclidean distances """
    dX = torch.unsqueeze(X,1) - torch.unsqueeze(X,2)
    D = torch.sqrt(torch.sum(dX**2, 3) + eps)

    if mode in ['topk','mink']:
        D_neighbors, E_idx = torch.topk(D, top_k+1, dim=-1, largest=False)
        #exclude self-connection
        D_neighbor =  D_neighbors[:,:,1:]
        E_idx = E_idx[:,:,1:]
        u = torch.tensor(np.arange(E_idx.shape[1]))[:,None].repeat(1, E_idx.shape[2]).reshape(-1)
        v = E_idx[0,].reshape(-1)

        # append more than k that are within dcut
        if mode == 'mink':
            nprv = len(u)
            mask = torch.where(torch.tril(D)<1.0e-6,100.0,1.0)
            _,uD,vD = torch.where((mask*D)<dcut)
            uv = np.array(list(zip(u,v))+list(zip(uD,vD)))
            uv = np.unique(uv,axis=0)
            u = [a for a,b in uv]
            v = [b for a,b in uv]
            
    elif mode == 'distT':
        # need to exclude self?
        mask = torch.where(torch.tril(D)<1.0e-6,100.0,1.0)
        _,u,v = torch.where((mask*D)<dcut)
        
    elif mode == 'dist':
        _,u,v = torch.where(D<dcut)

    return u,v,dX

def sample_uniform(fnats):
    return np.array([1.0 for _ in fnats])/len(fnats)

def upsample_nofake(cat):
    p = motif.sampling_weights_nofake[cat]
    return p/np.sum(p)

def upsample_category(cat):
    p = motif.sampling_weights[cat]
    return p/np.sum(p)

def upsample_category2(cat):
    p = motif.sampling_weights2[cat]
    
    return p/np.sum(p)

def collate(samples):
    #samples should be G,info
    valid = [v[0] != None for v in samples]
    try:
        info  = [s[1] for s in samples]
        samples = [s[0] for s in samples]
        bG = dgl.batch(samples)

        # below contains l0 features only
        if 'x' in bG.ndata:
            node_feature = {"0": bG.ndata["attr"][:,:,None].float(), "1": bG.ndata["x"].float()} #1: 3D, N x 1 x 3
        else:
            node_feature = {"0": bG.ndata["attr"][:,:,None].float()}
        edge_feature = {"0": bG.edata["attr"][:,:,None].float()}
    
        return bG, node_feature, edge_feature, info
    except:
        print("failed collation")
        return None, {}, {}, {},

def distance_feature(mode,d,binsize=0.5,maxd=5.0):
    if mode == '1hot':
        b = (d/binsize).long()
        nbin = int(maxd/binsize) # 0.0~5.0<
        b[b>=nbin] = nbin-1
        d1hot = torch.eye(nbin)[b].float()
        feat = d1hot.squeeze()
    elif mode == 'std': #sigmoid
        d0 = 0.5*(maxd-binsize) #center
        m = 5.0/d0 #slope
        feat = 1.0/(1.0+torch.exp(-m*(d-d0)))
        feat = feat.repeat(1,2) # hacky!! make as 2-dim
    return feat

def load_dataset(set_params, generator_params, setsuffix):
    train_set = Dataset(np.load("data/train_proteins%s.npy"%setsuffix),
                        **set_params)
    
    val_set = Dataset(np.load("data/valid_proteins%s.npy"%setsuffix),
                      **set_params)

    train_generator = data.DataLoader(train_set,
                                      #worker_init_fn=lambda _: np.random.seed(),
                                      **generator_params)
    
    valid_generator = data.DataLoader(val_set,
                                      #worker_init_fn=lambda _: np.random.seed(),
                                      **generator_params)

    
    return train_generator,valid_generator
    
