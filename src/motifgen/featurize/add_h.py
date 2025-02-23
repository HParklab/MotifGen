import sys
import chimera
from chimera import runCommand as rc
from chimera import replyobj 
import AddH


in_rec_file = sys.argv[3]
out_rec_file = sys.argv[4]

print("in rec file", in_rec_file)
print("out rec_file", out_rec_file)
rec = chimera.openModels.open(in_rec_file)

rc("del H")
rc("addh")
rc("addcharge all method gas")
rc("write format pdb #0 %s"%(out_rec_file))
rc('close all')

print ('finished add H')
