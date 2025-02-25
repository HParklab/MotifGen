import glob
import numpy as np
import copy
import os,sys
import time
from scipy.spatial import distance_matrix
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
import scipy
import motifgen.featurize.myutils as myutils

def sasa_from_xyz_old(xyz,reschains,atmres_rec):
    #dmtrx
    D = distance_matrix(xyz,xyz)
    cbcounts = np.sum(D<12.0,axis=0)-1.0

    # convert to apprx sasa
    cbnorm = cbcounts/50.0
    sasa_byres = 1.0 - cbnorm**(2.0/3.0)
    sasa_byres = np.clip(sasa_byres,0.0,1.0)

    # by atm
    sasa = [sasa_byres[reschains.index(res)] for res,atm in atmres_rec]
    
    return sasa

def sasa_from_xyz(xyz, elems, probe_radius=1.4, n_samples=50):
    atomic_radii = {"C":  2.0,"N": 1.5,"O": 1.4,"S": 1.85,"H": 0.0, #ignore hydrogen for consistency
                    "F": 1.47,"Cl":1.75,"Br":1.85,"I": 2.0,'P': 1.8}
    
    areas = []
    normareas = []
    centers = xyz
    radii = np.array([atomic_radii[e] for e in elems])
    n_atoms = len(elems)

    inc = np.pi * (3 - np.sqrt(5)) # increment
    off = 2.0/n_samples

    pts0 = []
    for k in range(n_samples):
        phi = k * inc
        y = k * off - 1 + (off / 2)
        r = np.sqrt(1 - y*y)
        pts0.append([np.cos(phi) * r, y, np.sin(phi) * r])
    pts0 = np.array(pts0)

    kd = scipy.spatial.cKDTree(xyz)
    neighs = kd.query_ball_tree(kd, 8.0)

    occls = []
    for i,(neigh, center, radius) in enumerate(zip(neighs, centers, radii)):
        neigh.remove(i)
        n_neigh = len(neigh)
        d2cen = np.sum((center[None,:].repeat(n_neigh,axis=0) - xyz[neigh]) ** 2, axis=1)
        occls.append(d2cen)
        
        pts = pts0*(radius+probe_radius) + center
        n_neigh = len(neigh)
        
        x_neigh = xyz[neigh][None,:,:].repeat(n_samples,axis=0)
        pts = pts.repeat(n_neigh, 0).reshape(n_samples, n_neigh, 3)
        
        d2 = np.sum((pts - x_neigh) ** 2, axis=2) # Here. time-consuming line
        r2 = (radii[neigh] + probe_radius) ** 2
        r2 = np.stack([r2] * n_samples)

        # If probe overlaps with just one atom around it, it becomes an insider
        n_outsiders = np.sum(np.all(d2 >= (r2 * 0.99), axis=1))  # the 0.99 factor to account for numerical errors in the calculation of d2
        # The surface area of   the sphere that is not occluded
        area = 4 * np.pi * ((radius + probe_radius) ** 2) * n_outsiders / n_samples
        areas.append(area)

        norm = 4 * np.pi * (radius + probe_radius)
        normareas.append(min(1.0,area/norm))

    occls = np.array([np.sum(np.exp(-occl/6.0),axis=-1) for occl in occls])
    occls = (occls-6.0)/3.0 #rerange 3.0~9.0 -> -1.0~1.0
    return areas, np.array(normareas), occls

def featurize_target_properties(pdb,out,extrapath="",verbose=False):
    # get receptor info
    qs_aa, atypes_aa, atms_aa, bnds_aa, repsatm_aa = myutils.get_AAtype_properties(extrapath=extrapath,
                                                                                   extrainfo={})
    resnames,reschains,xyz,atms,elems = myutils.read_pdb(pdb,read_ligand=False)
    
    # read in only heavy + hpol atms as lists
    q_rec = []
    atypes_rec = []
    xyz_rec = []
    atmres_rec = []
    aas_rec = []
    bnds_rec = []
    repsatm_idx = []
    residue_idx = []
    atmnames = []

    skipres = []
    reschains_read = []
    reschains_idx = []

    for i,resname in enumerate(resnames):
        reschain = reschains[i]

        if resname in myutils.ALL_AAS:
            iaa = myutils.ALL_AAS.index(resname)
            qs, atypes, atms, bnds_, repsatm = (qs_aa[iaa], atypes_aa[iaa], atms_aa[iaa], bnds_aa[iaa], repsatm_aa[iaa])
        else:
            if verbose: out.write("unknown residue: %s, skip\n"%resname)
            skipres.append(i)
            continue

        natm = len(xyz_rec)
        atms_r = []
                
        # unify metal index to Calcium for simplification
        if iaa >= myutils.ALL_AAS.index("CA"):
            #print(iaa, myutils.ALL_AAS[iaa])
            iaa = myutils.ALL_AAS.index("CA")
            atypes = atypes_aa[iaa]
            
        for iatm,atm in enumerate(atms):
            is_repsatm = (iatm == repsatm)
            
            if atm not in xyz[reschain]:
                if is_repsatm: return False
                continue
                
            atms_r.append(atm)
            q_rec.append(qs[atm])
            atypes_rec.append(atypes[iatm])
            aas_rec.append(iaa)
            xyz_rec.append(xyz[reschain][atm])
            atmres_rec.append((reschain,atm))
            residue_idx.append(i)
            if is_repsatm: repsatm_idx.append(natm+iatm)

        bnds = [[atms_r.index(atm1),atms_r.index(atm2)] for atm1,atm2 in bnds_ if atm1 in atms_r and atm2 in atms_r]

        # make sure all bonds are right
        for (i1,i2) in copy.copy(bnds):
            dv = np.array(xyz_rec[i1+natm]) - np.array(xyz_rec[i2+natm])
            d = np.sqrt(np.dot(dv,dv))
            if d > 2.0:
                if verbose:
                    out.write("Warning, abnormal bond distance: ", pdb, resname, reschain,  i1,i2, atms_r[i1], atms_r[i2],d+'\n')
                bnds.remove([i1,i2])
                
        bnds = np.array(bnds,dtype=int)
        atmnames += atms_r
        reschains_idx += [reschain for _ in atms_r]
        reschains_read.append(reschain)

        if i == 0:
            bnds_rec = bnds
        elif bnds_ != []:
            bnds += natm
            bnds_rec = np.concatenate([bnds_rec,bnds])
            
    xyz_rec = np.array(xyz_rec)

    if len(atmnames) != len(xyz_rec):
        sys.exit('inconsistent anames <=> xyz')

    '''
    # sasa apprx from coord -- old
    #sasa = sasa_from_xyz(xyz_rec[repsatm_idx],reschains_read,atmres_rec)
    elems_rec = [at[0] for at in atypes_rec]
    t0 = time.time()
    sasa, normsasa, occl = sasa_from_xyz(xyz_rec, elems_rec )
    t1 = time.time()
    
    np.savez(npz,
             # per-atm
             aas_rec=aas_rec,
             xyz_rec=xyz_rec, #just native info
             atypes_rec=atypes_rec, #string
             charge_rec=q_rec,
             bnds_rec=bnds_rec,
             sasa_rec=normsasa, #apo
             occl=occl,
             residue_idx=residue_idx,
             atmres_rec=atmres_rec,
             atmnames=atmnames, #[[],[],[],...]
                 
             # per-res (only for receptor)
             repsatm_idx=repsatm_idx,
             reschains=reschains,
             #atmnames=atmnames, #[[],[],[],...]

        )
    '''

    return xyz_rec, aas_rec, atmres_rec, atypes_rec, residue_idx, reschains, reschains_idx, atmnames


def find_motif_category(aa,atm):
    if aa+atm in ['SEROG','THROG1','TYROH','HISND','HISNE']:
        return 1
    elif atm == 'O' or aa+atm in ['ASPOD1','ASPOD2','ASNOD1','GLUOE1','GLUOE2','GLNOE1']:
        return 2
    elif atm == 'N' or aa+atm in ['TRPNE1','ARGNE','ARGNH1','ARGNH2','LYSNZ','ASND2','GLNNE2']:
        return 3
    elif aa+atm in ['VALCG1','VALCG2','ILECG1','ILECD1','LEUCD1','LEUCD2','ALACB','METCE','PROCG']:
        return 4
    elif aa+atm in ['PHECG','PHECD1','PHECD2','PHECE1','PHECE2','PHECZ',
                    'TYRCG','TYRCD1','TYRCD2','TYRCE1','TYRCE2','TYRCZ',
                    'TRPCG','TRPCD1','TRPCD2','TRPCE2','TRPCE3','TRPCZ2','TRPCZ3','TRPCH2']:
        return 5
    return 0

def get_motifs_from_complex(xyz_rec, atypes_rec, xyzs_lig, atypes_lig, vbase_lig=[],
                            anames_lig=[],
                            debug=False, outprefix=''):

    # ligand -- gentype
    D_lig = [i for i,at in enumerate(atypes_lig) if at in [1,3]]
    A_lig = [i for i,at in enumerate(atypes_lig) if at in [1,2]]
    H_lig = [i for i,at in enumerate(atypes_lig) if at==4]
    R_lig = [i for i,at in enumerate(atypes_lig) if at==5]

    D_rec = [i for i,at in enumerate(atypes_rec) if at in [1,3]]
    A_rec = [i for i,at in enumerate(atypes_rec) if at in [1,2]]
    HR_rec = [i for i,at in enumerate(atypes_rec) if at in [4,5]]


    kd_D   = scipy.spatial.cKDTree(xyz_rec[D_rec])
    kd_A   = scipy.spatial.cKDTree(xyz_rec[A_rec])
    kd_HR   = scipy.spatial.cKDTree(xyz_rec[HR_rec])
    
    kd_lig  = scipy.spatial.cKDTree(xyzs_lig)

    # not super fast but okay
    dv2D = np.array([[y-x for x in xyzs_lig] for y in xyz_rec[D_rec]])
    dv2A = np.array([[y-x for x in xyzs_lig] for y in xyz_rec[A_rec]])

    d2D = np.sqrt(np.einsum('ijk,ijk->ij',dv2D,dv2D))
    d2A = np.sqrt(np.einsum('ijk,ijk->ij',dv2A,dv2A))

    #o2D = np.einsum('jk,ijk->ij', vbase_lig, dv2D)/d2D
    #o2A = np.einsum('jk,ijk->ij', vbase_lig, dv2A)/d2A

    iA = np.unique(np.where(d2D<3.6)[1])
    iD = np.unique(np.where(d2A<3.6)[1])

    iHR = np.unique(np.concatenate(kd_HR.query_ball_tree(kd_lig,5.1)).astype(int))

    xyzs_m = []
    motifs = []
    for i,xyz in enumerate(xyzs_lig):
        mtype = 0
        # let mutually exclusive
        if (i in iA or i in iD) and (i in A_lig and i in D_lig): mtype = 1 #Both

        #buggy
        #elif (i in iA and i not in iD) and (i in A_lig): mtype = 2 # Acc
        #elif (i not in iA and i in iD) and (i in D_lig): mtype = 3 # Don

        #new -- don't mind if lig-acceptor is close to rec-acceptor & vice versa for donor
        elif (i in iA) and (i in A_lig): mtype = 2 # Acc
        elif (i in iD) and (i in D_lig): mtype = 3 # Don
        elif i in H_lig and i in iHR: mtype = 4 # Ali
        elif i in R_lig and i in iHR: mtype = 5 # Aro

        if mtype > 0:
            motifs.append(mtype)
            xyzs_m.append(xyz)

    if debug:
        out = open('%s.motif.xyz'%outprefix,'w')
        aname = ['X','B','O','N','C','R']
        for x,m in zip(xyzs_m,motifs):
            out.write('%-4s %8.3f %8.3f %8.3f\n'%(aname[m],x[0],x[1],x[2]))
        out.close()

    return np.array(xyzs_m), motifs

def get_interaction_from_model(recpdb,
         ligmol2='',
         ligchain=None,
         hotspot=[],
         gridsize=1.5,
         padding=4.0,
         verbose=False,
         debug=False,
         out=sys.stdout,
         inputpath = './',
         propout=None,
         outprefix=None,
         masksize=3,
         include_fake=True,
         skip_if_exist=True):

    if inputpath[-1] != '/': inputpath+='/'

    tag = recpdb.split('/')[-1][:-4]
    
    out.write("Read native info\n")

    # read relevant motif
    aas, reschains, xyz, atms, elems = myutils.read_pdb(recpdb) #rosetta processed
    
    if outprefix == None: outprefix = tag
    args = featurize_target_properties(recpdb,out)
    
    if not args:
        print("failed featurizing target properties, %s"%recpdb)
        return

    # numpy arrays (not dictionary)
    xyzs, aas, _, _, _, reschains, reschains_idx, atmnames = args
    aas = [myutils.ALL_AAS[i] for i in aas]

    irec = [i for i,rc in enumerate(reschains_idx) if rc.split('.')[0] != ligchain]
    xyz_rec = xyzs[irec]
    atypes_rec = [find_motif_category(aas[i],atmnames[i]) for i in irec]
    
    #print("reschains_idx", reschains_idx, "ligchain", ligchain)
    ilig = [i for i,rc in enumerate(reschains_idx) if rc.split('.')[0] == ligchain]
    #print("ilig", ilig)
    xyz_lig = xyzs[ilig]
    atypes_lig = [find_motif_category(aas[i],atmnames[i]) for i in ilig]
    
    xyzs_motif, cats_motif = get_motifs_from_complex(xyz_rec, atypes_rec,
                                                     xyz_lig, atypes_lig,
                                                     debug=debug,outprefix=outprefix)

    print("xyzs_motif", xyzs_motif, "cats_motif", cats_motif)
    return xyzs_motif, cats_motif



