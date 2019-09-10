# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     model
   Description :
   Author :       erik_xiong
   date：          2019-05-28
-------------------------------------------------
   Change Activity:
                   2019-05-28:
-------------------------------------------------
"""
__author__ = 'erik_xiong'

import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.utils import scatter_
from torch_scatter import *
import math
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
import pickle
import numpy as np
import sys
import os

class LinearBn(nn.Module):
    def __init__(self, in_channel, out_channel, act=None):
        super(LinearBn, self).__init__()
        self.linear = nn.Linear(in_channel, out_channel)
        self.bn = nn.BatchNorm1d(out_channel, eps=1e-06, momentum=0.1)
        self.act = act

    def forward(self, x):
        x = self.linear(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.act is not None:
            x = self.act(x)
        return x
    

class graphAttention(nn.Module):
    def softmax(self, x, index, num=None):
        x = x - scatter_max(x, index, dim=0, dim_size=num)[0][index]
        x = x.exp()
        x = x / (scatter_add(x, index, dim=0, dim_size=num)[index] + 1e-8)
        return x

    def __init__(self, bond_dim, fingerprint_dim, p_dropout):
        super(graphAttention, self).__init__()

        self.dropout = nn.Dropout(p=p_dropout)
        
        self.encoder = nn.Sequential(
            LinearBn(bond_dim, fingerprint_dim * fingerprint_dim),
            nn.ReLU(inplace=True),
        )

        self.align = nn.Linear(fingerprint_dim*2,1)
        self.attend = LinearBn(fingerprint_dim, fingerprint_dim)
        self.gru = nn.GRUCell(fingerprint_dim, fingerprint_dim)

    def forward(self, atom, bond_index, bond):
        num_atom, atom_dim = atom.shape 
        num_bond, bond_dim = bond.shape
        bond_index = bond_index.t().contiguous()
        
        neighbor_atom = atom[bond_index[1]]  # 获得所有边的第二个节点的特征 
        bond = self.encoder(bond).view(-1, atom_dim, atom_dim)  # 将边特征转变为节点特征大小  (12,128,128)

        neighbor = neighbor_atom.view(-1, 1, atom_dim) @ bond  # 12,1,128        
        neighbor = neighbor.view(-1, atom_dim) 
                
        target_atom = atom[bond_index[0]]  # 获得所有边的第一个节点的特征 
        
        feature_align = torch.cat([target_atom, neighbor],dim=-1)
        
        align_score = F.leaky_relu(self.align(self.dropout(feature_align)))

        attention_weight = self.softmax(align_score, bond_index[0], num=num_atom)

        context = scatter_('add', torch.mul(attention_weight, self.attend(self.dropout(neighbor))), \
                           bond_index[0], dim_size=num_atom)
        
        context = F.elu(context) 

        update = self.gru(context, atom)

        return update

class superatomAttention(torch.nn.Module):

    def softmax(self, x, index, num=None):
        x = x - scatter_max(x, index, dim=0, dim_size=num)[0][index]
        x = x.exp()
        x = x / (scatter_add(x, index, dim=0, dim_size=num)[index] + 1e-8)
        return x
    
    def __init__(self, fingerprint_dim, p_dropout):
        super(superatomAttention, self).__init__()
        
        self.dropout = nn.Dropout(p=p_dropout)
        self.align = nn.Linear(2*fingerprint_dim,1)
        self.attend = LinearBn(fingerprint_dim, fingerprint_dim)

        self.gru = nn.GRUCell(fingerprint_dim, fingerprint_dim)
        
    def forward(self, superatom, atom, mol_index):
        
        superatom_num = mol_index.max().item() + 1 # number of molecules in a batch

        superatom_expand = superatom[mol_index]

        feature_align = torch.cat([superatom_expand, atom],dim=-1)

        align_score = F.leaky_relu(self.align(self.dropout(feature_align)))

        attention_weight = self.softmax(align_score, mol_index, num=superatom_num)

        context = scatter_('add', torch.mul(attention_weight, self.attend(self.dropout(atom))), \
                           mol_index, dim_size=superatom_num)

        context = F.elu(context)

        update = self.gru(context, superatom) 

        return update, attention_weight

class Fingerprint_viz(torch.nn.Module):
    def __init__(self, num_target, fingerprint_dim, K=3, T=3, p_dropout=0.2, atom_dim=39, bond_dim=10):
        super(Fingerprint_viz, self).__init__()
        self.K = K
        self.T = T
        self.dropout = nn.Dropout(p=p_dropout)

        self.preprocess = nn.Sequential(
            LinearBn(atom_dim, fingerprint_dim),
            nn.ReLU(inplace=True),
        )
    
        self.propagate = nn.ModuleList([graphAttention(bond_dim, fingerprint_dim, p_dropout) for _ in range(K)])
#         self.superGather = superatomAttention(fingerprint_dim, p_dropout=p_dropout)
        self.superGather = nn.ModuleList([superatomAttention(fingerprint_dim, p_dropout) for _ in range(T)])
    
        self.predict = nn.Sequential(
            LinearBn(fingerprint_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=p_dropout),
            nn.Linear(512, num_target),
        )

    def forward(self, atom, bond, bond_index, mol_index):
        atom = self.preprocess(atom)
        num_atom, atom_dim = atom.shape

        atoms = []
        atoms.append(atom)
        for k in range(self.K):
            atom = self.propagate[k](atom, bond_index, bond)
            atoms.append(atom)

#         atom = torch.stack(atoms)
#         atom = torch.mean(atom, dim=0)
        superatom_num = mol_index.max()+1
        superatom = scatter_('add', atom, mol_index, dim_size=superatom_num) # get init superatom by sum
        superatoms = []
        attention_weight_viz = []
        superatoms.append(superatom)
        
        for t in range(self.T):
            superatom, attention_weight = self.superGather[t](superatom, atom, mol_index) 
            attention_weight_viz.append(attention_weight)
            superatoms.append(superatom)

        predict = self.predict(superatom)

        return predict, atoms, superatoms, attention_weight_viz

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)
    

def null_collate(batch):
    batch_size = len(batch)

    atom = []
    bond = []
    bond_index = []
    mol_index = []  
    label = []
    smiles = []
    offset = 0
    for b in range(batch_size):
        graph = batch[b]
        smiles.append(graph.smiles)
        num_atom = len(graph.atom)
        atom.append(graph.atom)
        if graph.bond.size == 0:
            bond = np.zeros((1, 10), np.uint32)
            bond_index = np.zeros((1, 2), np.uint32)
        bond.append(graph.bond)
        bond_index.append(graph.bond_index + offset)
        mol_index.append(np.array([b] * num_atom))
        
        offset += num_atom
        label.append(graph.label)
        
    atom = torch.from_numpy(np.concatenate(atom)).float()
    bond = torch.from_numpy(np.concatenate(bond)).float()
    bond_index = torch.from_numpy(np.concatenate(bond_index).astype(np.int32)).long()
    mol_index = torch.from_numpy(np.concatenate(mol_index).astype(np.int32)).long()
    label = torch.from_numpy(np.concatenate(label).astype(np.int32)).float()
    
    return smiles, atom, bond, bond_index, mol_index, label


class graph_dataset(Dataset):
    def __init__(self, smiles_list, graph_dict):

        self.graph_dict = graph_dict
        self.smiles_list = smiles_list

    def __getitem__(self, x):

        smiles = self.smiles_list[x]

        graph = self.graph_dict[smiles]

        graph.atom = np.concatenate(graph.atom, -1)

        return graph

    def __len__(self):
        return len(self.smiles_list)

class Graph:
    def __init__(self, smiles, atom, bond, bond_index, label):
        self.smiles = smiles
        self.atom = atom
        self.bond = bond
        self.bond_index = bond_index
        self.label = label
        
    def __str__(self):
        return f'graph of {self.smiles}'
    
class Logger(object):
    def __init__(self):
        self.terminal = sys.stdout  # stdout
        self.file = None

    def open(self, file, mode=None):
        if mode is None: mode = 'w'
        if os.path.exists(file):
            os.remove(file)
        self.file = open(file, mode)

    def write(self, message, is_terminal=1, is_file=1):
        if '\r' in message: is_file = 0

        if is_terminal == 1:
            self.terminal.write(message)
            self.terminal.flush()

        if is_file == 1:
            self.file.write(message)
            self.file.flush()

    def flush(self):
        pass

def time_to_str(t, mode='min'):
    if mode == 'min':
        t = int(t) / 60
        hr = t // 60
        min = t % 60
        return '%2d hr %02d min' % (hr, min)
    elif mode == 'sec':
        t = int(t)
        min = t // 60
        sec = t % 60
        return '%2d min %02d sec' % (min, sec)
    else:
        raise NotImplementedError
        
if __name__ == '__main__':
    
    smiles_list = ['C1=CC=CC=C1', 'CNC']
    graph_dict = pickle.load(open('test.pkl',"rb"))
    train_loader = DataLoader(graph_dataset(smiles_list, graph_dict), batch_size=2, collate_fn=null_collate)
    net = Fingerprint(atom_dim=39, bond_dim=10, num_target=2)
    for b, (smiles, atom, bond, bond_index, mol_index, label) in enumerate(train_loader):
#         print(atom, bond, bond_index, mol_index, label)
        _ = net(atom, bond, bond_index, mol_index)
        break

    print('model success!')
