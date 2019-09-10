# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     featurizing
   Description :
   Author :       erikxiong
   date：          2019-06-18
-------------------------------------------------
   Change Activity:
                   2019-06-18:
-------------------------------------------------
"""
__author__ = 'erikxiong'

import os
from rdkit import Chem
from rdkit.Chem import MolFromSmiles
from rdkit import RDConfig
from rdkit.Chem import ChemicalFeatures
import pickle
import numpy as np

class Graph:
    def __init__(self, smiles, atom, bond, bond_index, label):
        self.smiles = smiles
        self.atom = atom
        self.bond = bond
        self.bond_index = bond_index
        self.label = label
        
    def __str__(self):
        return f'graph of {self.smiles}'
    
    
def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception("input {0} not in allowable set{1}:".format(
            x, allowable_set))
    return [x == s for s in allowable_set]


def one_of_k_encoding_unk(x, allowable_set):
    """Maps inputs not in the allowable set to the last element."""
    if x not in allowable_set:
        x = allowable_set[-1]
    return [x == s for s in allowable_set]

def generate_graph(smiles, label=None):
    mol = MolFromSmiles(smiles)
    if not mol:
        raise ValueError("Could not parse SMILES string:", smiles)
        
    SYMBOL = ['B','C','N','O','F','Si','P','S','Cl','As','Se','Br','Te','I','At','other']
    HYBRIDIZATION = [
        Chem.rdchem.HybridizationType.SP,
        Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3,
        Chem.rdchem.HybridizationType.SP3D,
        Chem.rdchem.HybridizationType.SP3D2,
        'other',
    ]


    num_atom = Chem.RemoveHs(mol).GetNumAtoms()

    symbol = np.zeros((num_atom, 16), np.uint8)
    hybridization = np.zeros((num_atom, 6), np.uint8)
    degree = np.zeros((num_atom, 6), np.uint8)
    num_h = np.zeros((num_atom, 5), np.uint8)  
    chirality = np.zeros((num_atom, 3), np.uint8)
    aromatic = np.zeros((num_atom, 1), np.uint8)
    formal_charge = np.zeros((num_atom, 1), np.float32)
    radical_electrons = np.zeros((num_atom, 1), np.float32)

    for i in range(num_atom):
        atom = mol.GetAtomWithIdx(i)
        symbol[i] = one_of_k_encoding_unk(atom.GetSymbol(), SYMBOL)
        hybridization[i] = one_of_k_encoding_unk(atom.GetHybridization(), HYBRIDIZATION)
        degree[i] = one_of_k_encoding_unk(atom.GetDegree(), [0, 1, 2, 3, 4, 5])
        num_h[i] = one_of_k_encoding_unk(atom.GetTotalNumHs(includeNeighbors=True), [0, 1, 2, 3, 4])
        try:
            chirality[i] = one_of_k_encoding_unk(atom.GetProp('_CIPCode'), ['R', 'S', 'unknown'])
        except:
            chirality[i] = [0, 0, 0]
        aromatic[i] = atom.GetIsAromatic()
        formal_charge[i] = atom.GetFormalCharge()
        radical_electrons[i] = atom.GetNumRadicalElectrons()
        
        
        
#     abundant features
#     won't bring substantial change to predictive performance, sometimes even worse 
    
    AtomicWeight = np.zeros((num_atom, 1), np.float32)
    AtomicNumber = np.zeros((num_atom, 1), np.float32)
    Rvdw = np.zeros((num_atom, 1), np.float32)
    RCovalent = np.zeros((num_atom, 1), np.float32)
    DefaultValence = np.zeros((num_atom, 1), np.float32)
    valence = np.zeros((num_atom, 1), np.float32)
    NOuterElecs = np.zeros((num_atom, 1), np.float32)
    ring = np.zeros((num_atom, 7), np.uint8)
    acceptor = np.zeros((num_atom, 1), np.uint8)
    donor = np.zeros((num_atom, 1), np.uint8)

    for i in range(num_atom):
        atom = mol.GetAtomWithIdx(i)
        AtomicNum = atom.GetAtomicNum()
        AtomicNumber[i] = AtomicNum
        AtomicWeight[i] = Chem.GetPeriodicTable().GetAtomicWeight(AtomicNum)
        Rvdw[i] = Chem.GetPeriodicTable().GetRvdw(AtomicNum)  # (van der Waals radius)
        RCovalent[i] = Chem.GetPeriodicTable().GetRcovalent(AtomicNum) #(covalent radius)
        DefaultValence[i] = Chem.GetPeriodicTable().GetDefaultValence(AtomicNum)  
        valence[i] = atom.GetExplicitValence()
        NOuterElecs[i] = Chem.GetPeriodicTable().GetNOuterElecs(AtomicNum)
        ring[i] = [int(atom.IsInRing()), int(atom.IsInRingSize(3)), \
                   int(atom.IsInRingSize(4)), int(atom.IsInRingSize(5)), \
                   int(atom.IsInRingSize(6)), int(atom.IsInRingSize(7)), int(atom.IsInRingSize(8))]
              

    factory = ChemicalFeatures.BuildFeatureFactory(os.path.join(RDConfig.RDDataDir, 'BaseFeatures.fdef'))
    feature = factory.GetFeaturesForMol(mol)
    for t in range(0, len(feature)):
        if feature[t].GetFamily() == 'Donor':
            for i in feature[t].GetAtomIds():
                donor[i] = 1
        elif feature[t].GetFamily() == 'Acceptor':
            for i in feature[t].GetAtomIds():
                acceptor[i] = 1
        

    num_bond = mol.GetNumBonds()
    if num_bond == 0:
        num_bond = 1 # except error caused by CH4, NH3
    bond_feat = np.zeros((num_bond*2, 10), np.int16)
    bond_index = np.zeros((num_bond*2, 2), np.int16)

    BOND_TYPE = [
        Chem.rdchem.BondType.SINGLE,
        Chem.rdchem.BondType.DOUBLE,
        Chem.rdchem.BondType.TRIPLE,
        Chem.rdchem.BondType.AROMATIC,
    ]

    BOND_STEREO = ["STEREONONE", "STEREOANY", "STEREOZ", "STEREOE"]
    ij = 0
    for i in range(num_atom):
        for j in range(num_atom):
            if i == j: continue
            bond = mol.GetBondBetweenAtoms(i, j)
            if bond is not None:
                atom1 = mol.GetAtomWithIdx(i)
                atom2 = mol.GetAtomWithIdx(j)
                bond_index[ij] = [i, j]
                bond_type = one_of_k_encoding(bond.GetBondType(), BOND_TYPE) 
                bond_ring = [bond.GetIsConjugated(), bond.IsInRing()]
                bond_stereo = one_of_k_encoding(str(bond.GetStereo()), BOND_STEREO)
                bond_feat[ij] = bond_type + bond_ring + bond_stereo            
                ij += 1

    graph = Graph(
        smiles,
        [symbol, hybridization, degree, num_h, chirality, aromatic, formal_charge, radical_electrons, \
        AtomicWeight, AtomicNumber, Rvdw, RCovalent, DefaultValence, valence, NOuterElecs, ring, acceptor, donor],
        bond_feat,
        bond_index,
        np.array(label).reshape((1, 1)),
    )

    return graph
    
    
def graph_dict(smiles_list, label_list, filename):

    try:
        graph_dict = pickle.load(open(filename+'_abundant.pkl',"rb"))
        print('graph dicts loaded from '+ filename+'_abundant.pkl')
        
    except:
        graph_dict = {}
        for i, smiles in enumerate(smiles_list):        
            graph_dict[smiles] = generate_graph(smiles, label_list[i])

        pickle.dump(graph_dict,open(filename+'_abundant.pkl',"wb"))
        print('graph dicts saved as '+ filename+'_abundant.pkl')

    return graph_dict

if __name__ == '__main__':

    smiles_list = ['C1=CC=CC=C1', 'CNC', 'N', 'C', ]
    label_list = [2.,3.,4.,4.34]
    graph_dict = graph_dict(smiles_list, label_list, 'test')
    print(graph_dict['N'].atom)
    print(graph_dict['N'].bond)
    print(graph_dict['N'].bond_index)
    print(graph_dict['N'].label)
    print('load done.')
