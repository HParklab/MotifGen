#reformat pdb -> protein chain as chain "A" / peptide chain as chain "B"

wrt = ""
with open("model_complex.pdb", "r") as fp:
    for line in fp:
        if not line.startswith("ATOM"):
            wrt += line
            continue
        res_no = int(line[22:26].strip())
        if res_no >= 240:
            new_resno = res_no - 239
            new_line = line[:21] + "B" + "%4.3s"%(new_resno) + line[26:]
            wrt += new_line
        else:
            wrt += line
fp.close()

#write to new pdb file
with open("renum_model_complex.pdb", "w") as fp:
    fp.write(wrt)
fp.close()