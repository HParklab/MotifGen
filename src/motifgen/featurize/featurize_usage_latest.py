# New grid-making code, which utilises the logic originally intended to filter out 'false positive' hotspots for grid exclusion
import glob
import numpy as np
import copy
import os,sys
import scipy
from scipy.spatial import distance_matrix
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
import myutils

class GridOption:
    def __init__(self,padding,gridsize,option,clash,contact,
                 probe=1.65,dcut=4.0):
        self.padding = padding
        self.gridsize = gridsize
        self.option = option
        self.clash = clash
        self.contact = contact

        self.shellsize=5.0 # through if no contact within this distance
        self.dcut = dcut
        self.probe = probe

def sasa_from_xyz(xyz,reschains,atmres_rec):
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

def featurize_target_properties(pdb,npz,out,extrapath="",verbose=False):
    # get receptor info
    qs_aa, atypes_aa, atms_aa, bnds_aa, repsatm_aa = myutils.get_AAtype_properties(extrapath=extrapath,
                                                                                   extrainfo={})
    resnames,reschains,xyz,atms,_ = myutils.read_pdb(pdb,read_ligand=False)
    
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
        reschains_read.append(reschain)

        if i == 0:
            bnds_rec = bnds
        elif bnds_ != []:
            bnds += natm
            bnds_rec = np.concatenate([bnds_rec,bnds])
            
    xyz_rec = np.array(xyz_rec)

    if len(atmnames) != len(xyz_rec):
        sys.exit('inconsistent anames <=> xyz')

    # sasa apprx from coord
    sasa = sasa_from_xyz(xyz_rec[repsatm_idx],reschains_read,atmres_rec)
    
    np.savez(npz,
             # per-atm
             aas_rec=aas_rec,
             xyz_rec=xyz_rec, #just native info
             atypes_rec=atypes_rec, #string
             charge_rec=q_rec,
             bnds_rec=bnds_rec,
             sasa_rec=sasa, #apo
             residue_idx=residue_idx,
             atmnames=atmnames, #[[],[],[],...]
                 
             # per-res (only for receptor)
             repsatm_idx=repsatm_idx,
             reschains=reschains,
             #atmnames=atmnames, #[[],[],[],...]

        )

    return xyz_rec, aas_rec, atmres_rec, atypes_rec, q_rec, bnds_rec, sasa, residue_idx, repsatm_idx, reschains, atmnames

def grid_from_xyz(xyzs,xyz_lig,gridsize=1.5,
                  clash=2.0,contact=4.0,padding=0.0,
                  gridopt=None,
                  option='ligandxyz',
                  gridout=sys.stdout):

    if gridopt != None:
        clash = gridopt.clash
        option = gridopt.option
        clash = gridopt.clash
        padding = gridopt.padding
        gridsize = gridopt.gridsize
        contact = gridopt.contact
    
    reso = gridsize*0.7
    bmin = [min(xyz_lig[:,k])-padding for k in range(3)]
    bmax = [max(xyz_lig[:,k])+padding for k in range(3)]

    imin = [int(bmin[k]/gridsize)-1 for k in range(3)]
    imax = [int(bmax[k]/gridsize)+1 for k in range(3)]

    grids = []
    print("detected %d grid points..."%((imax[0]-imin[0])*(imax[1]-imin[1])*(imax[2]-imin[2])))
    for ix in range(imin[0],imax[0]+1):
        for iy in range(imin[1],imax[1]+1):
            for iz in range(imin[2],imax[2]+1):
                grid = np.array([ix*gridsize,iy*gridsize,iz*gridsize])
                grids.append(grid)
                #i = len(grids)
                #print("HETATM %4d  CA  CA  X   1    %8.3f%8.3f%8.3f"%(i,grid[0],grid[1],grid[2]))
    grids = np.array(grids)
    #grids = np.array([[-2.,11.,22.],[-2.,11.,23.],[-1.,11.,22.]])
    nfull = len(grids)

    # first remove clashing grids
    kd      = scipy.spatial.cKDTree(grids)
    kd_ca   = scipy.spatial.cKDTree(xyzs)
    excl = kd.query_ball_tree(kd_ca, clash) #nxyz x varlists 
    incl = kd.query_ball_tree(kd_ca, contact)

    if option == 'global':
        nco = np.array([len(a) for a in incl])
        ncl = np.array([len(a) for a in excl])
        idx = np.intersect1d(np.where(nco>80)[0],np.where(ncl==0)[0])
        grids = grids[idx]
        
        incl = [incl[i] for i in idx]
        v2co = np.array([np.mean(xyzs[i],axis=0)-g for i,g in zip(incl,grids)])
        v2co = np.sqrt(np.einsum('ik,ik->i',v2co,v2co))
        idx = np.where(v2co<3.0)
        grids = grids[idx]
        
        # remove outliers
        kd      = scipy.spatial.cKDTree(grids)
        neigh = kd.query_ball_tree(kd, gridsize*2.0)
        idx = [i for i,n in enumerate(neigh) if len(n) > 10]
    else:
        incl = list(np.unique(incl))
        excl = list(np.unique(excl))
        idx = np.intersect1d(incl,noncl)
        
    grids = grids[idx]

    if gridout != None:
        for i,grid in enumerate(grids):
            gridout.write("HETATM %4d  H   H   X%4d    %8.3f%8.3f%8.3f\n"%(i,i,grid[0],grid[1],grid[2]))
    
    print("Search through %d grid points, of %d contact grids %d clash -> %d"%(nfull,len(incl),len(excl),len(grids)))

    if option == 'ligandxyz':
        #regen kdtree
        kd      = scipy.spatial.cKDTree(grids)
        for xyz in xyz_lig:
            kd_ca   = scipy.spatial.cKDTree(xyz[None,:])
            indices += kd_ca.query_ball_tree(kd, reso)[0]
        indices = list(np.unique(indices))
        grids = grids[indices]
        
    elif option == 'ligandcubic':
        pass
    
    return grids

def sasa_grids(xyz, elems, probe_radius=1.4, n_samples=50, d_clash=0.7):
    atomic_radii = {"C":  1.8, "R":1.6, "N": 1.5, "O": 1.3, "S": 1.85,"H": 0.0,
                    "F": 1.47,"Cl":1.75,"Br":1.85,"I": 2.0,'P': 1.8}
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
    
    pts_out = []
    for i, (neigh, center, radius) in enumerate(zip(neighs, centers, radii)):
        neigh.remove(i)
        n_neigh = len(neigh)
        
        pts = pts0*(radius+probe_radius) + center
        x_neigh = xyz[neigh][None,:,:].repeat(n_samples,axis=0)
        pts_expand = pts.repeat(n_neigh, 0).reshape(n_samples, n_neigh, 3)

        d2 = np.sum((pts_expand - x_neigh) ** 2, axis=2)

        r2 = (radii[neigh] + probe_radius) ** 2
        r2 = np.stack([r2] * n_samples)

        # If probe overlaps with just one atom around it, it becomes an insider
        outsiders = np.all(d2 >= (r2 * 0.99), axis=1)  # the 0.99 factor to account for numerical errors in the calculation of d2
        pts_out.append(pts[np.where(outsiders)[0]])

    pts_out = np.concatenate(pts_out)
    
    # drop overlaps
    kd = scipy.spatial.cKDTree(pts_out)
    clash = kd.query_ball_tree(kd, d_clash)
    clashmap = np.zeros((len(pts_out),len(pts_out)))

    for i,js in enumerate(clash):
        clashmap[i][js] = 1
        clashmap[i,i] = 0

    incl = np.zeros(len(pts_out))+1
    for k in range(5):
        nclash = np.sum(clashmap,axis=0)
        if sum(nclash) == 0: break
        
        i_s, j_s = np.where(clashmap>0)
        i_s = [i for i,j in zip(i_s,j_s) if i < j]
        incl[i_s] = 0
        clashmap[i_s,:] = 0
        clashmap[:,i_s] = 0
        
    incl = np.where(incl>0)[0]
    grids = pts_out[incl]
    
    return grids

def filter_by_contacts(xyzs,grids,padding=8.0,ncut=10,depth_cut=3.5):
    kd      = scipy.spatial.cKDTree(grids)
    kd_ca   = scipy.spatial.cKDTree(xyzs)
    incl = kd.query_ball_tree(kd_ca, padding) # defines outer shell

    nco = np.array([len(a) for a in incl])
    idx = np.where(nco>20)[0]
    grids = grids[idx]

    incl = [incl[i] for i in idx] # grid w/ >= 20 atomic neighs?
    
    # distance from protein atoms
    v2co = np.array([np.mean(xyzs[i],axis=0)-g for i,g in zip(incl,grids)])
    v2co = np.sqrt(np.einsum('ik,ik->i',v2co,v2co))
    idx = np.where(v2co<depth_cut) #main param; 3.0 too shallow, 4.0 too messy
    grids = grids[idx]

    # remove isolators
    kd      = scipy.spatial.cKDTree(grids)
    neigh = kd.query_ball_tree(kd, 4.0)
    idx = [i for i,n in enumerate(neigh) if len(n) > ncut]
    grids = grids[idx]

    
    return grids

def main(pdb,outprefix,
         recpdb=None,
         gridsize=1.5,
         padding=10.0,
         clash=0.7,
         contact=4.0,
         ligname=None,
         ligchain=None,
         out=sys.stdout,
         gridoption='ligand',
         maskres=[],
         com=[],
         skip_if_exist=True,
         verbose=False):


    out_dir = os.path.dirname(pdb)
    
    # read relevant motif
    aas, reschains, xyz, atms, elems = myutils.read_pdb(pdb,read_ligand=True,detect_aro=True,skip_H=True)

    gridopt = GridOption(padding,gridsize,gridoption,clash,contact)
    
    if gridoption[:6] == 'ligand':
        if (ligname != None and ligname in aas):
            #i_lig = reschains.index(ligandname)
            reschain_lig = [reschains[aas.index(ligname)]]
        elif ligchain != None:
            reschain_lig = [rc for rc in reschains if rc[0] == ligchain]
        else:
            sys.exit("Unknown ligname or ligchain: ", ligname, ligchain)
        xyz_lig = np.concatenate([list(xyz[rc].values()) for rc in reschain_lig])
        xyz = np.concatenate(np.array([list(xyz[rc].values()) for rc in reschains if rc not in reschain_lig]))

        with open(outprefix+'.grid.pdb','w') as gridout:
            grids = grid_from_xyz(xyz,xyz_lig,gridsize,padding=4.0,option=gridoption,gridout=gridout)
        out.write("Found %d grid points around ligand\n"%(len(grids)))

    elif gridoption == 'global':
        xyz = [np.array(list(xyz[rc].values()),dtype=np.float32) for rc in reschains if rc not in maskres]
        xyz = np.concatenate(xyz)

        with open(outprefix+'.grid.pdb','w') as gridout:
            grids = grid_from_xyz(xyz,xyz,gridopt=gridopt,gridout=gridout)
        out.write("Found %d grid points around ligand\n"%(len(grids)))
        
    elif gridoption == 'sasa':
        xyz = [np.array(list(xyz[rc].values()),dtype=np.float32) for rc in reschains if rc not in maskres]
        xyz = np.concatenate(xyz)
        elems = [np.array(list(elems[rc].values())) for rc in reschains if rc not in maskres]
        elems = np.concatenate(elems)
        
        grids = sasa_grids(xyz, elems, probe_radius=gridopt.probe,
                           n_samples=50, d_clash=gridopt.clash)
        grids = filter_by_contacts(xyz,grids,depth_cut=gridopt.dcut)
        
    elif gridoption == 'com':
        xyz = [np.array(list(xyz[rc].values()),dtype=np.float32) for rc in reschains if rc not in maskres]
        xyz = np.concatenate(xyz)

        # com should be passed through input argument
        assert(len(com) == 3)
        com = com[None,:]
        with open(outprefix+'.grid.pdb','w') as gridout:
            grids = grid_from_xyz(xyz,com,gridopt,gridout=gridout)
    else:
        sys.exit("Error, exit")

    if verbose:
        gridout = open(os.path.join(out_dir, '%s.grid.xyz'%outprefix),'w')
        for x in grids:
            gridout.write('F %8.3f %8.3f %8.3f\n'%(x[0],x[1],x[2]))
        gridout.close()

    kd      = scipy.spatial.cKDTree(grids)
    kd_ca   = scipy.spatial.cKDTree(xyz)
    incl = kd.query_ball_tree(kd_ca, padding) # defines outer shell
    protrusion = np.array([len(a) for a in incl])
    
    recnpz = os.path.join(out_dir,"%s.prop.npz"%(outprefix))
    if recpdb == None: recpdb = pdb
    
    out.write("Featurize receptor info from %s...\n"%recpdb)
    featurize_target_properties(recpdb,recnpz,out,verbose=verbose)
    
    tags = ["grid%04d"%i for i,grid in enumerate(grids)]
    gridnpz = os.path.join(out_dir,"%s.lig.npz"%(outprefix))
    np.savez(gridnpz,
             xyz=grids,
             protrusion=protrusion,
             name=tags)

if __name__ == "__main__":
    pdb = sys.argv[1]
    tag = pdb.split('/')[-1].split(".pdb")[0]
    
    main(pdb, tag, gridoption='sasa', verbose=True)
