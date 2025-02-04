import numpy as np
import pickle as pkl
import os, sys
from sklearn.cluster import DBSCAN
MYPATH = os.path.dirname(sys.argv[0])
if MYPATH == '': MYPATH = './'
import multiprocessing as mp

def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

def cluster(pred_npz, recxyz):
    data = np.load(pred_npz,allow_pickle=True)
    motifs               = ['','ASP','','LYS','', 'BB','AMD','','','PHE','','PHO','PH2','']
    #Pcut       = [-1,0.20, -1,  0.2, -1, 0.25, 0.2, -1,-1, 0.20, -1, 0.3, 0.30,-1] #"reference correction" for pre-tuned

    #78%
    #333
    Pcut       =  np.array([-1,0.25, -1, 0.15, -1, 0.25, 0.20, -1,-1, 0.20,-1, 0.20,0.20,-1]) #manual v1  -- confirmed

    xyzs = data['grids']
    Ps   = data['P']
    
    usable_ks = [k for k,m in enumerate(motifs) if m != '']
    
    for k in range(5):
        xyzs_trimmed = []
        cats_trimmed = []
        Ps_trimmed = []
        for i,(x,p) in enumerate(zip(xyzs,Ps)):
            for k in usable_ks:
                if p[k] > Pcut[k]:
                    xyzs_trimmed.append(x)
                    cats_trimmed.append(k)
                    Ps_trimmed.append(p[k])

        xyzs_trimmed = np.array(xyzs_trimmed)
        cats_trimmed = np.array(cats_trimmed)
        Ps_trimmed   = np.array(Ps_trimmed)

        #print(pred_npz)
        clusters = DBSCAN(eps=2.4, min_samples=5).fit(xyzs_trimmed) ### change this part !! / note 19.09.
        ncl = max(clusters.labels_)
        clusters = [np.where(clusters.labels_ == i) for i in range(ncl+1)]
        if len(clusters) == 0:
            maxcl = 0
        else:
            maxcl = [cl for cl in clusters if len(cl) > 0]
            maxcl = max([len(cl[0]) for cl in maxcl])
        
        if maxcl > 150:
            Pcut *= 1.05
        else:
            break
        #print(pred_npz,k,maxcl)

    cl_xyzs = [xyzs_trimmed[idx] for idx in clusters]
    cl_cats = [cats_trimmed[idx] for idx in clusters]
    cl_Ps   = [Ps_trimmed[idx] for idx in clusters]
    cl_com = [np.sum(xyz*p[:,None],axis=0)/np.sum(p) for p,xyz in zip(cl_Ps,cl_xyzs)]
    cl_prot = []
    for xyz in cl_xyzs:
        dv = xyz[None,:,:]-recxyz[:,None,:]
        d = np.sqrt(np.sum(dv*dv,axis=2))
        prot = np.sum(d<10.0,axis=0)
        cl_prot.append(prot)
    
    details = []
    i = 0
    ic = 0
    for xyzs,cats,Ps in zip(cl_xyzs,cl_cats,cl_Ps):
        ic += 1
        for x,c,p in zip(xyzs,cats,Ps):
            i += 1
            details.append(f'HETATM {i:4d}  {motifs[c]:3s} UNK X{ic:4d}    {x[0]:8.3f}{x[1]:8.3f}{x[2]:8.3f}  1.00  {p:4.2f}')
        
    return cl_com, cl_cats, cl_Ps, cl_prot, details

def extract_cluster_feature(recxyz, cl_com, cl_Ps, cl_prot, cl_cats):
    cluster_features = []

    for com,Ps,cats,prot in zip(cl_com,cl_Ps,cl_cats, cl_prot):
        feature_vector = np.zeros(8)
        
        # feature 1. protrusion
        feature_vector[0] = max(prot)

        # features 2 and 3. likeliness
        feature_vector[1] = max(Ps)
        feature_vector[2] = np.mean(np.array(Ps), axis = 0)
        
        # features 4-7. cluster properties
        hydrophilic,hydrophobic,pos_charge,neg_charge = [],[],[],[]
        for cat in cats:
            polar = (cat == 1 or cat == 3 or cat == 5 or cat == 6)
            hydrophilic.append(int(polar))
            hydrophobic.append(int(not polar))
            pos_charge.append(int(cat == 3))
            neg_charge.append(int(cat == 1))

        feature_vector[3] = np.mean(hydrophilic)
        feature_vector[4] = np.mean(hydrophobic)
        feature_vector[5] = np.mean(pos_charge)
        feature_vector[6] = np.mean(neg_charge)

        # feature 8. number of members in each cluster
        feature_vector[7] = len(Ps)
        cluster_features.append(feature_vector)
        
    return cluster_features

# set the label for cluster closest to any ligand atom as 1, the rest as 0
def find_closest_cl(cl_com,ligxyz,cut=4.0):
    mind2lig = []
    for clxyz in cl_com:
        dv = np.array([clxyz - xyz for xyz in ligxyz])
        d2lig = np.sqrt(np.einsum("ik,ik->i",dv,dv))
        mind2lig.append(min(d2lig))
    
    return np.where(np.array(mind2lig) < cut)[0]

def predict_site(prednpz, modelname,
           recxyz=[], ligxyz=[], feature_prefix=None, debug=False):
    cl_com, cl_cats, cl_Ps, cl_prot, details = cluster(prednpz, recxyz)
    
    if len(cl_com) == 0:
        print(f'{prefix} 0 cluster')
        return 
    
    if debug:
        out = open('mycl.xyz','w')
        for com in cl_com:
            out.write("C %8.3f %8.3f %8.3f\n"%tuple(com))
        out.close()
            
    feats = extract_cluster_feature(recxyz, cl_com, cl_Ps, cl_prot, cl_cats)

    f = open(modelname, 'rb')
    clf = pkl.load(f)
    f.close()
    
    prediction = clf.predict_proba(feats)

    label = None
    if len(ligxyz) > 0:
        label = find_closest_cl(cl_com, ligxyz)
        label = np.array([(i in labels) for i,_ in enumerate(prediction)],dtype=float)
        
    return prediction,cl_com,feats,label,details

def predict_property(feats,modelname):
    with open(modelname, 'rb') as f:
        regr = pkl.load(f)
    
    logp = regr['logp'].predict(feats)
    tpsa = regr['tpsa'].predict(feats)

    # fit to [0-1]
    logp = 1/(1+np.exp(-5*(logp-0.5)))
    tpsa = 1/(1+np.exp(-10*(tpsa-0.35)))
        
    return logp, tpsa # these are binned probability

def range2str(bins,binrange):
    grp = []
    for i in bins:
        if grp == [] or i-1 not in bins:
            grp.append([i])
        elif grp != []:
            if i-1 in grp[-1]:
                grp[-1].append(i)
            else:
                grp.append([i])

    if len(grp) == 0:
        return ''
    
    if grp[-1][-1] == len(binrange)-1: 
        grp[-1].append(-1)

    outstr = []
    for g in grp:
        if g[-1] == -1:
            outstr.append(' %.1f~'%(binrange[g[0]]))
        else:
            outstr.append(' %.1f~%.1f'%(binrange[g[0]],binrange[g[-1]+1]))
    return ','.join(outstr)

def range2str2(bins,n):
    outstr = ['*' if i in bins else ' ' for i in range(n)]
    return ' '.join(['%3s'%a for a in outstr])

def main(prefix,
         sitemodel="/home/hpark/programs/DANligand/src/MotifNet/misc/site_predictor.pkl",
         propmodel="/home/hpark/programs/DANligand/src/MotifNet/misc/prop_predictor.pkl",
         ligpdb='',
         report_feat=False,report_pdb=True,log=sys.stdout):
    print("> Running ",prefix)

    propnpz = '%s.prop.npz'%prefix
    prednpz = '%s.score.npz'%prefix
    #outprefix='prediction/%s'%prefix

    ligxyz = []
    if ligpdb != '':
        ligcont = [l[:-1] for l in open(ligpdb) if (l.startswith('ATOM') or l.startswith('HETATM')) and l[13] != 'H']
        ligxyz = np.array([[float(l[30:38]),float(l[38:46]),float(l[46:54])] for l in ligcont])
    
    nonH   = [i for i,a in enumerate(np.load(propnpz,allow_pickle=True)['atypes_rec']) if a[0] != 'H']
    recxyz = np.load(propnpz,allow_pickle=True)['xyz_rec'][nonH]
    
    args = predict_site(prednpz, modelname=sitemodel, recxyz=recxyz, ligxyz=ligxyz, debug=True)
    if not args:
        return
        
    pred,com,feats,labels,details = args
    
    logp, tpsa = predict_property(feats, modelname=propmodel)

    out = None
    if report_pdb:
        out = open(prefix+'.cl.pdb','w')
        
    log.write('#'+' '*(len(prefix)-1) + 'Indx Prob Pred     X        Y       Z    | LogP profile                                                 | TPSA profile\n' )
    for i,(p,x,f,l,t) in enumerate(zip(pred,com,feats,logp,tpsa)):
        predstr = " "
        labelstr = " "
        if abs(p[1]-max(pred[:,1])) < 0.001:
            predstr = "*"
            atype = 'ZN'
        else:
            atype = 'Cl'
            
        if labels != None:
            if labels[i]: labelstr = "!"

        logpstr = ' %4.2f'*len(l)%tuple(l)
        tpsastr = ' %4.2f'*len(t)%tuple(t)
        
        log.write(f"{prefix} {i+1:3d} {p[1]:4.2f} {predstr:1s} {labelstr:1s} {x[0]:8.3f} {x[1]:8.3f} {x[2]:8.3f}"+ ' | ' + logpstr + ' | '+tpsastr+'\n')
        details.append(f'HETATM {i+1:4d}  {atype:3s} {atype:3s} X{i+1:4d}    {x[0]:8.3f}{x[1]:8.3f}{x[2]:8.3f}  1.00 {p[0]:5.2f}'+" %6.2f"*len(f)%tuple(f)+'\n')

    log.write('\n')
    isite = np.argmax(pred[:,1])
    
    log.write("#LogP bins: (25,250,25)\n")
    log.write("#TPSA bins: (-5,5,1)\n")
    
    tpsa_r = np.arange(25,251,25)
    logp_r = np.arange(-5,5.1,1)

    log.write('# '+' %3d'*len(logp_r)%tuple(logp_r)+'   | '+' %3d'*len(tpsa_r)%tuple(tpsa_r)+'\n')
    for i,(l,t) in enumerate(zip(logp,tpsa)):
        pts_l = np.where(l>0.9)[0]
        pts_t = np.where(t>0.9)[0]
        #logpstr = range2str(pts_l, logp_r)
        #tpsastr = range2str(pts_t, tpsa_r)
        logpstr = range2str2(pts_l, len(logp_r))
        tpsastr = range2str2(pts_t, len(tpsa_r))
        log.write(f'{i+1:2d} | {logpstr:40s} |   {tpsastr:40s}\n')
        
    if out != None:
        out.writelines('\n'.join(details)+'\n')
        out.close()

# just for unit test
if __name__ == "__main__":
    prefix = sys.argv[1]
    
    if prefix == '-l':
        a = mp.Pool(processes=10)
        prefices = [l[:-1] for l in open(sys.argv[2])]
        for prefix in prefices:
            try:
                main(prefix,report_feat=True,log=open(prefix+'.log','w'))
            except:
                continue
    else:
        main(prefix,report_feat=True,log=open('%s.log'%prefix,'w'))
