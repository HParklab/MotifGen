import os, sys, glob
import numpy as np
from pathlib import Path
from collections import defaultdict

    
class Atom:
    def __init__(self):
        pass
    def parsed_line(self, line):
        self.header=line[:6]
        self.atm_no = int(line[6:11])
        self.atm_name = line[12:16].strip()
        self.res_name = line[17:20]
        self.res_no = int(line[22:26].strip())
        self.chain_ID = line[21:22]
        self.R = np.array([float(line[30:38]), float(line[38:46]),  float(line[46:54])])
        
        

def to_one_letter(res_name: str) -> str:
    mapping = {
        'ALA': 'A', 'VAL': 'V', 'ILE': 'I', 'LEU': 'L', 'MET': 'M',
        'PHE': 'F', 'TRP': 'W', 'TYR': 'Y', 'ASP': 'D', 'GLU': 'E',
        'LYS': 'K', 'ARG': 'R', 'HIS': 'H', 'SER': 'S', 'THR': 'T',
        'ASN': 'N', 'GLN': 'Q', 'CYS': 'C', 'SEC': 'U', 'GLY': 'G',
        'PRO': 'P'
    }
    return mapping.get(res_name.upper(), 'X')  # 알 수 없는 건 'X'


class PDB:
    def __init__(self, fn: Path):
        self.fn = fn
        self.atom_s = []
        self.rec_num = 0
        self.res_seq = {}
        
        self.dict_rec = {} #key: residue_"index", value: np.array of coordinates
        self.dict_pep = {} #key: residue_"index"(res_no+rec_num), value: np.array of coordinates
        
        self.process_pdb()

    def process_pdb(self) -> None:
        self.atom_s = self._parse_pdb_file(self.fn)
        self.rec_num = self._get_rec_num(self.atom_s)
        self.pep_num = self._get_pep_num(self.atom_s)
        self._separate_chains()

    def _parse_pdb_file(self, fn: str) -> list:
        atoms = []
        with open(fn, 'r') as f:
            for line in f:
                if line.startswith('ATOM'):
                    at = Atom()
                    at.parsed_line(line)
                    atoms.append(at)
        return atoms

    def _get_rec_num(self, atoms: list) -> int:
        res_set = set()
        for at in atoms:
            if at.chain_ID == 'A':
                res_set.add(at.res_no)
        return len(res_set)
    
    def _get_pep_num(self, atoms: list) -> int:
        res_set = set()
        for at in atoms:
            if at.chain_ID == 'B':
                res_set.add(at.res_no)
        return len(res_set)

    def _separate_chains(self) -> None:
        rec_coords = defaultdict(list)
        pep_coords = defaultdict(list)
        
        for atom in self.atom_s:
            #save the residue sequence
            if atom.res_no not in self.res_seq and atom.chain_ID == "A":
                self.res_seq[atom.res_no-1] = to_one_letter(atom.res_name)
            elif atom.res_no + self.rec_num not in self.res_seq and atom.chain_ID == "B":
                self.res_seq[atom.res_no + self.rec_num-1] = to_one_letter(atom.res_name)


            if atom.chain_ID == 'A':
                rec_index = atom.res_no - 1  
                rec_coords[rec_index].append(atom.R)

            elif atom.chain_ID == 'B':
                pep_index = atom.res_no + self.rec_num - 1
                pep_coords[pep_index].append(atom.R)

            else:
                #raise Error -> input pdb file should have only two chains(receptor: A, peptide: B)
                raise ValueError("Input pdb file should have only two chains(receptor: A, peptide: B)")
        
        self.dict_rec = {k: np.array(v, dtype=float) for k, v in rec_coords.items()}
        self.dict_pep = {k: np.array(v, dtype=float) for k, v in pep_coords.items()}
