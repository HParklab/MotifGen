import numpy as np
import sys

# just for book-keeping
# order as charged -> A&D -> RingD -> RingA -> rest
MOTIFS = ['None', #0
          'COO','Arg','Amine', # 1~3
          'OH','bb','Amide1', # 4~6
          'Nhis','Ntrp', #7~8
          'Phe', 'SH', #9 #10
          'CH3', 'CH32', 'SCH3' #11~13
          ]
SIMPLEMOTIFIDX = [0, 
                  2,4,4, 
                  3,3,3,
                  2,4,
                  6,6,6,6,6] #0:none 2:acceptor 3:both 4:donor 6:non-Hbonder

BBMOTIF = (5,'C','O','CA')
POSSIBLE_MOTIFS = \
    {#'CYS':[('SG1','HG','CB'),BBMOTIF],
        'ASP':[(1,'CG','OD1','CB'),BBMOTIF],
        'GLU':[(1,'CD','OE1','CG'),BBMOTIF],
        'PHE':[(9,'CG','CD1','CB'),BBMOTIF],
        'HIS':[(7,'ND1','HD1','CG'),(7,'NE2','HE2','CD2'), #donor
               (8,'ND1','CE1','CG'),(8,'NE2','CE1','CD2'), #acceptor
               BBMOTIF], # 
        'LYS':[(3,'NZ','1HZ','CE'),BBMOTIF],
        'ASN':[(6,'CG','OD1','CB'),BBMOTIF],
        'SER':[(4,'OG','HG','CB'),BBMOTIF],
        'THR':[(4,'OG1','HG1','CB'),BBMOTIF],
        'GLN':[(6,'CD','OE1','CG'),BBMOTIF],
        'TYR':[(9,'CG','CD1','CB'),(4,'OH','HH','CZ'),BBMOTIF],
        'TRP':[(9,'CD2','CE3','CG'),(7,'NE1','HE1','CD1'),BBMOTIF],
        'ARG':[(2,'CZ','NH1','NE'),BBMOTIF],
        'ALA':[(11,'CA','CB','N'),BBMOTIF],
        'VAL':[(12,'CB','CG1','CA'),BBMOTIF],
        'ILE':[(11,'CG1','CD1','CB'),(11,'CB','CG2','CA'),BBMOTIF],
        'LEU':[(12,'CG','CD1','CB'),BBMOTIF],
        'MET':[(13,'SD','CE','CG'),BBMOTIF],
    }

O  = np.array([ 0.0, 0.0, 0.0])
F1 = np.array([ 1.3, 0.75, 0.0]) #2nd atm.A
F1H = np.array([0.95,0.55, 0.0]) #2nd atm.A
F2 = np.array([-1.3, 0.75, 0.0]) #2nd atm.B
B  = np.array([ 0.0,-1.5, 0.0]) #3rd atm
FH = np.array([ 0.0, 1.1, 0.0])


# my DB
sampling_weights_nofake = 0.5*np.array([0.000001,  3.,  5., 8., #null, charged
                                        4.,  4. , 6., #OH,bb,amide
                                        6.,  6., #Nx
                                        2.,   0.,  1.5,  1.5,  6.]) #phe/SH/HPs

sampling_weights = 0.5*np.array([1.,  3.,  5., 8., #null, charged
                                 4.,  4. , 6., #OH,bb,amide
                                 6.,  6., #Nx
                                 2.,   0.,  1.5,  1.5,  6.]) #phe/SH/HPs

sampling_weights2 = 0.5*np.array([0.1,  3.,  5., 8., #null, charged
                                 4.,  4. , 6., #OH,bb,amide
                                 6.,  6., #Nx
                                 2.,   0.,  1.5,  1.5,  6.]) #phe/SH/HPs

# my + Hyunuk
#sampling_weights = np.array([1.0,  2.,  6., 6.,
#                             2.,  2.5,  2.5,
#                             3.,  6.,
#                             1.5,  0.0, 1.0,  1.0, 5.])

# by func group
REFXYZ = [[],
          [O,F1,B],
          [O,F1,B],
          [O,F1H,B], #amine
          [O,F1H,B], #OH; F1=H
          [O,F1,B], #bb
          [O,F1,B], #amide
          
          [O,FH,-F2],#ring-NH 
          [O,-F1,-F2],#ring-N:
          [O,F1,B], #Phe
          
          [O,F1H,B], #-SH; F1=H
          [O,F1,B], #CH3
          [O,F1,B], #2CH3
          [O,F1,B], #S-CH3
          ]

# symmetry
SYMM = [1,
        2,2,3,1,1,1, # COO ~ amide
        2,2,2, #ring-NH ~ PHE
        1,1,2,1] #SH ~ S-CH3

# define how to get the (y-axis,x-axis,symmetry)
# x-axis: rotation-normal
# y-axis: "connected to"
#
# O   O
#  \C/
#   |
#   C  ->x
#      ^ y

# orth1,2: Gram-Schmidt orthogonalized v1/v2 
# naive1/2: == norm(v1) or norm(v2)
# naive12: == norm(v1+v2) 
# inv12  : == -norm(v1+v2)
# inv2   : == -norm(v2)
MOTIFFRAME = [[], #none
              ['naive2', 'orth1', 2], #OOC
              ['naive2', 'orth1', 2], #guanidinium
              ['naive2', 'orth1', 3], #amine
              ['naive2', 'orth1', 1], #alcohol
              ['naive2', 'orth1', 1], #bb
              ['naive2', 'orth1', 1], #1-st amide
              ['inv1',   'orth2', 2], #ring N-H ##special case!
              ['inv12',  'orth1', 2], #ring N:
              ['naive2', 'orth1', 2], #phe
              ['naive2', 'orth1', 1], # -SH
              ['naive2', 'orth1', 1], # CH3
              ['naive2', 'orth1', 1], # 2CH3
              ['naive2', 'orth1', 1], # SCH3
] 

def qmul(a,b):
    q = np.zeros(4)
    q[0] = a[0]*b[0] - a[1]*b[1] - a[2]*b[2] - a[3]*b[3]
    q[1] = a[0]*b[1] + a[1]*b[0] + a[2]*b[3] - a[3]*b[2]
    q[2] = a[0]*b[2] - a[1]*b[3] + a[2]*b[0] + a[3]*b[1]
    q[3] = a[0]*b[3] + a[1]*b[2] - a[2]*b[1] + a[3]*b[0]
    return q

def R2q( R, dim=4 ):
    Q = np.zeros(4)
    if R[0,0] > R[1,1] and R[0,0] > R[2,2]:
        S  = np.sqrt( 1.0 + R[0,0] - R[1,1] - R[2,2] ) * 2
        Q[1] = (0.25 * S)
        Q[2] = (R[0,1] + R[1,0]) / S
        Q[3] = (R[2,0] + R[0,2] ) / S
        Q[0] = (R[2,1] - R[1,2] ) / S
    elif R[1,1] > R[2,2]:
        S  = np.sqrt( 1.0 + R[1,1] - R[0,0] - R[2,2] ) * 2
        Q[1] = (R[1,0] + R[0,1] ) / S
        Q[2] = 0.25 * S
        Q[3] = (R[2,1] + R[1,2] ) / S
        Q[0] = (R[0,2] - R[2,0] ) / S
    else:
        S  = np.sqrt( 1.0 + R[2,2] - R[0,0] - R[1,1] ) * 2
        Q[1] = (R[0,2] + R[2,0] ) / S
        Q[2] = (R[2,1] + R[1,2] ) / S
        Q[3] = 0.25 * S
        Q[0] = (R[1,0] - R[0,1]) / S

    # make sure normalized
    Q = Q/np.sqrt(np.dot(Q,Q))
    return Q

def xyz2axis(v1,v2,ftype): #v1: 0->1 v2: 0->2
    if ftype == 'orth1': #orthogonalized-v1
        e2 = v2/(np.sqrt(np.dot(v2,v2)+1e-6))
        e2 = v1 - np.dot(e2,v1)*e2
        return e2/np.sqrt(np.dot(e2,e2))
    
    elif ftype == 'orth2': 
        e1 = v1/(np.sqrt(np.dot(v1,v1)+1e-6))
        e1 = v2 - np.dot(e1,v2)*e1
        return e1/np.sqrt(np.dot(e1,e1))

    elif ftype == 'naive2':
        return -v2/np.sqrt(np.dot(v2,v2))
    
    elif ftype == 'inv1':
        return v1/np.sqrt(np.dot(v1,v1))

    elif ftype == 'inv12':
        v = v1+v2
        return -v/np.sqrt(np.dot(v,v))
    return False

def q2R(q,maxang=-1):
    R = np.zeros((3,3))
    R[0] = [1. - 2*(q[2]*q[2]+q[3]*q[3]), 2*(q[1]*q[2]-q[3]*q[0]), 2*(q[1]*q[3]+q[2]*q[0])]
    R[1] = [2*(q[1]*q[2]+q[3]*q[0]), 1. - 2*(q[1]*q[1]+q[3]*q[3]), 2*(q[2]*q[3]-q[1]*q[0])]
    R[2] = [2*(q[1]*q[3]-q[2]*q[0]), 2*(q[2]*q[3]+q[1]*q[0]), 1. - 2*(q[1]*q[1]+q[2]*q[2])]
    return np.array(R)

def LossRot(q0s,q1,K=10.0): #q0s: answer; q1: pred
    import torch
    # for debugging, eval wrt the xyz0 (q = 1,0,0,0)
    q0s = q0s[0,:]
    q1 = torch.transpose(q1,0,1)
    qdots = torch.matmul(q0s,q1)

    # calc weight (== probability)
    P = torch.exp(K*qdots)/torch.sum(torch.exp(K*qdots+1.0e-6))

    # weighted by ?
    qdot = torch.sum(P*qdots) 
    Qscore = 1.0 - torch.exp(qdot*qdot)/np.exp(1.0)

    #dv = self.xyz - self.T - self.xyz0
    #rmsd = torch.sqrt(torch.sum(dv*dv))
    return Qscore

def LossAxis(v0,v1,K=10): 
    import torch
    # dot product
    # 0.0 ~ 2.0
    loss = 1.0 - torch.sum(v0*v1) #make sure v0,v1 normalized
    return loss

class MotifClass:
    def __init__(self,xyz,mtype):
        self.q = np.zeros(4)
        self.T = np.zeros(3)
        self.R = np.zeros((3,3)) # also the base vectors e1,e2,e3
        self.pdbid = ''
        self.aa = ''
        self.chainres = ''
        self.xyz = xyz
        self.xyz0 = REFXYZ[mtype]
        self.mtype = mtype

        self.Qscore = -1
        self.rmsd = -1
        self.symm = SYMM[mtype]
        self.qs_symm = []

    def xyz2frame(self):
        v1 = self.xyz[1] - self.xyz[0]
        v2 = self.xyz[2] - self.xyz[0]

        framefunc1,framefunc2,symmetry = MOTIFFRAME[self.mtype]

        ## base vector
        # make sure x,y are unit vectors
        e2 = xyz2axis(v1,v2,framefunc1)
        e1 = xyz2axis(v1,e2,framefunc2)
        e3 = np.cross(e1,e2)

        # store as e2 (==y-axis) & rotation
        self.T = self.xyz[0]
        R = np.array([e1,e2,e3]) 

        # always the symm axis is e2
        self.q = R2q(R)
        self.R = R
        self.T = self.xyz[0] # atom 1 position

        #if self.symm > 1:
        self.get_symmetric_qs() #always

    # recover xyz from q
    def frame2xyz(self,isymm=-1):
        if isymm < 0:
            R = q2R(self.q)
        else: # symmetric def invoked
            R = q2R(self.qs_symm[isymm])

        # why all transpose stuff?
        R = np.transpose(R)
        xyz = np.transpose(np.matmul(R,np.transpose(self.xyz0)))
        xyz += self.T
        self.xyz = xyz
    
    def report(self):
        form = "%4s %-5s %3s %2d %1d : "+" %7.3f"*7+" | %8.3f %8.3f"
        print(form%(self.pdbid, self.chainres, self.aa, self.mtype, self.symm,
                    self.T[0], self.T[1], self.T[2],
                    self.q[0], self.q[1], self.q[2], self.q[3],
                    self.rmsd, self.Qscore))

    def write_xyz(self,outf,xyz=[]):
        if len(xyz) == 0: xyz = self.xyz
        
        out = open(outf,'w')
        for x in xyz:
            out.write("C %8.3f %8.3f %8.3f\n"%(x[0],x[1],x[2]))
        axis = self.q[1:] + self.T
        out.write("O %8.3f %8.3f %8.3f\n"%(axis[0],axis[1],axis[2]))
        out.close()

    def eval(self):
        # for debugging, eval wrt the xyz0 (q = 1,0,0,0)
        q0 = np.array([1,0,0,0])

        if self.symm == 1:
            # single version
            qdot = np.dot(q0,self.q)

        else:
            # multi-def version
            qdots = np.matmul(self.qs_symm,np.transpose(q0))
            K = 10.0
            P = np.exp(K*qdots)/np.sum(np.exp(K*qdots+1.0e-6))
            qdot = np.dot(P,qdots)
            
        self.Qscore = 1.0 - np.exp(qdot*qdot)/np.exp(1.0)

        dv = self.xyz - self.T - self.xyz0
        self.rmsd = np.sqrt(np.sum(dv*dv))
        #print(self.rmsd, self.Qscore)

    def get_symmetric_qs(self):
        e2 = self.R[1] #rotation axis always y
        self.qs_symm = []
        for k in range(self.symm):
            ang = (360.0*k/self.symm)*np.pi/180.0
            q = np.zeros(4)
            q[0] = np.cos(ang/2)
            q[1:] = np.sin(ang/2)*e2
            
            self.qs_symm.append( qmul(self.q,q) )

def pdb2motifs(pdb):
    motifs = []
    xyzs = {}
    aas = {}
    
    # first read xyz,chainres
    for l in open(pdb):
        if not l.startswith('ATOM'): continue
        aa = l[16:20].strip()
        chainres = l[21]+'.'+l[22:26].strip()
        if chainres not in aas:
            aas[chainres] = aa
            xyzs[chainres] = {}

        aname = l[12:16].strip()
        xyzs[chainres][aname] = [float(l[30:38]),float(l[38:46]),float(l[46:54])]

    # get motifs
    for chainres in xyzs:
        aa = aas[chainres]
        if aa not in POSSIBLE_MOTIFS: continue
        xyz = xyzs[chainres]

        for mtype,a1,a2,a3 in POSSIBLE_MOTIFS[aa]:
            if mtype == 5: continue
            if a1 not in xyz or a2 not in xyz or a3 not in xyz: continue
                
            xyz_base = np.array([xyz[a1],xyz[a2],xyz[a3]])
            motif = MotifClass(xyz_base,mtype)
            motif.aa = aa
            motif.mtype = mtype
            motif.pdbid = pdb.split('/')[-1][:4]
            motif.chainres = chainres

            # calculate motif.q, .R, .T
            motif.xyz2frame()
            motif.eval()
            motifs.append(motif)
                
            motif.report()

            motif.write_xyz("%s.%s.self.xyz"%(chainres,aa))
            motif.write_xyz("%s.%s.ref.xyz"%(chainres,aa),motif.xyz0+motif.T)

            # for book-keeping
            motif.frame2xyz()
            motif.write_xyz("%s.%s.regen.xyz"%(chainres,aa))

            if motif.symm > 1:
                for k in range(motif.symm):
                    motif.frame2xyz(isymm=k)
                    motif.write_xyz("%s.%s.symm%d.xyz"%(chainres,aa,k))
    
    return motifs

def test():
    for i in [1,2,3,4]:
        aa = MOTIFS[i]
        motif = MotifClass(REFXYZ[i],i)
        motif.xyz2frame()
        motif.aa = aa
        motif.report()
        motif.write_xyz("%s.self.xyz"%(aa))
        motif.write_xyz("%s.ref.xyz"%(aa),motif.xyz0+motif.T)

        # for book-keeping
        motif.frame2xyz()
        motif.write_xyz("%s.regen.xyz"%(aa))
        if motif.symm > 1:
            for k in range(motif.symm):
                motif.frame2xyz(isymm=k)
                motif.write_xyz("%s.symm%d.xyz"%(aa,k))
                
if __name__ == "__main__":
    pdb2motifs(sys.argv[1])
    #test()
