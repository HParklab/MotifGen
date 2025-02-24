# -*- coding: utf-8 -*-
import os
import sys
import chimera
from chimera import runCommand as rc
from chimera import replyobj 
import AddH


in_rec_file =  sys.argv[3]  # 실행하는 디렉토리를 기준으로 경로 완성
out_rec_file = sys.argv[4]

# Change to absolute path
print("=== Absolute Path Check ===")
print("Input PDB:", in_rec_file)
print("Output PDB:", out_rec_file)

rec = chimera.openModels.open(in_rec_file)

rc("del H")
rc("addh")
rc("addcharge all method gas")
rc("write format pdb #0 %s" % (out_rec_file))
rc("close all")

print('Finished adding H')