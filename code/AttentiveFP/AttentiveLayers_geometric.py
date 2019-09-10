# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     geometrics
   Description :
   Author :       Erik Xiong
   date：          2019-05-28
-------------------------------------------------
   Change Activity:
                   2019-05-28:
-------------------------------------------------
"""
__author__ = 'Erik Xiong'

import torch
from torch import nn
from torch.nn import functional as F, Parameter
from torch_geometric.utils import scatter_
from torch_scatter import *
import math
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
import pickle
import numpy as np
import sys
import os
from torch_geometric.nn.pool.topk_pool import topk, filter_adj
from torch_geometric.nn import MessagePassing, GCNConv
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.utils import remove_self_loops, add_self_loops, softmax

def softmax(x, index, num=None):
    x = x - scatter_max(x, index, dim=0, dim_size=num)[0][index]
    x = x.exp()
    x = x / (scatter_add(x, index, dim=0, dim_size=num)[index] + 1e-8)
    return x


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
    
class BiLinearBn(nn.Module):
    def __init__(self, in_channel1, in_channel2, out_channel, act=None):
        super(BiLinearBn, self).__init__()
        self.bilinear = nn.Bilinear(in_channel1, in_channel2, out_channel)
        self.bn = nn.BatchNorm1d(out_channel, eps=1e-06, momentum=0.1)
        self.act = act

    def forward(self, x):
        x = self.linear(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.act is not None:
            x = self.act(x)
        return x
    
class Preprocess(nn.Module):
    def __init__(self, in_channel, out_channel, p_dropout=0.1, act=None):
        super(Preprocess, self).__init__()
        self.preprocess = nn.Sequential(
            LinearBn(in_channel, out_channel),
            nn.ReLU(inplace=True),
            nn.Dropout(p=p_dropout),
            LinearBn(out_channel, out_channel),
            nn.ReLU(inplace=True),
        )
        self.act = act
            
    def forward(self, x):
        x = self.preprocess(x)
        if self.act is not None:
            x = self.act(x)
        return x    

class graphAttention(nn.Module):
    
    def __init__(self, fingerprint_dim, p_dropout):
        super(graphAttention, self).__init__()

        self.dropout = nn.Dropout(p=p_dropout)
        
        self.align = LinearBn(fingerprint_dim*2,1)
        self.attend = LinearBn(fingerprint_dim, fingerprint_dim)
        self.gru = nn.GRUCell(fingerprint_dim, fingerprint_dim)

    def forward(self, atom, bond_index):
        
        num_atom, atom_dim = atom.shape
                
        feature_align = torch.cat([atom[bond_index[0]], atom[bond_index[1]]],dim=-1)
        
        align_score = F.leaky_relu(self.align(self.dropout(feature_align)))

        attention_weight = softmax(align_score, bond_index[0], num=num_atom)

        context = scatter_('add', torch.mul(attention_weight, self.attend(self.dropout(atom[bond_index[1]]))), \
                           bond_index[0], dim_size=num_atom)
        
        context = F.elu(context) 

        update = self.gru(context, atom)

        return update

    
class superatomAttention(torch.nn.Module):

    def __init__(self, fingerprint_dim, p_dropout):
        super(superatomAttention, self).__init__()
        
        self.dropout = nn.Dropout(p=p_dropout)
        self.align = LinearBn(2*fingerprint_dim,1)
        self.attend = LinearBn(fingerprint_dim, fingerprint_dim)

        self.gru = nn.GRUCell(fingerprint_dim, fingerprint_dim)
        
    def forward(self, superatom, atom, mol_index):
        
        superatom_num = mol_index.max().item() + 1 # number of molecules in a batch

        superatom_expand = superatom[mol_index]

        feature_align = torch.cat([superatom_expand, atom],dim=-1)

        align_score = F.leaky_relu(self.align(self.dropout(feature_align)))

        attention_weight = softmax(align_score, mol_index, num=superatom_num)

        context = scatter_('add', torch.mul(attention_weight, self.attend(self.dropout(atom))), \
                           mol_index, dim_size=superatom_num)

        context = F.elu(context)

        update = self.gru(context, superatom) 

        return update, attention_weight

    
class SAGPool(torch.nn.Module):
    def __init__(self, in_channels, ratio=0.8, Conv=GCNConv, non_linearity=torch.tanh):
        super(SAGPool, self).__init__()
        self.in_channels = in_channels
        self.ratio = ratio
        self.score_layer = Conv(in_channels, 1)
        self.non_linearity = non_linearity

    def forward(self, x, edge_index, edge_attr, batch):
        score = self.score_layer(x, edge_index).squeeze()

        perm = topk(score, self.ratio, batch)

        x = x[perm] * self.non_linearity(score[perm]).view(-1, 1)

        batch = batch[perm]

        edge_index, edge_attr = filter_adj(
            edge_index, edge_attr, perm, num_nodes=score.size(0))

        a = gmp(x, batch)
        m = gap(x, batch)

        return torch.cat([m, a], dim=1)

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    
class Fingerprint(torch.nn.Module):
    
    def __init__(self, num_target, fingerprint_dim, K=3, T=3, p_dropout=0.2, atom_dim=39, bond_dim=10):
        super(Fingerprint, self).__init__()
        self.K = K
        self.T = T
        self.dropout = nn.Dropout(p=p_dropout)

        self.atom_preprocess = Preprocess(atom_dim, fingerprint_dim)    
        self.bond_preprocess = Preprocess(bond_dim, fingerprint_dim)    
#         self.bilinear = BiLinearBn(fingerprint_dim, fingerprint_dim, fingerprint_dim)
        
        self.align = LinearBn(4*fingerprint_dim,1)
        
        self.attend = LinearBn(3*fingerprint_dim, 3*fingerprint_dim)

        self.gru = nn.GRUCell(3*fingerprint_dim, fingerprint_dim)
        
        
#         self.propagate = GATAtom(fingerprint_dim, fingerprint_dim, dropout=p_dropout)
#         self.superGather = GATAtom(fingerprint_dim, fingerprint_dim, dropout=p_dropout)
        self.propagate = graphAttention(fingerprint_dim, p_dropout=p_dropout)
#         self.propagate = nn.ModuleList([graphAttention(bond_dim, fingerprint_dim, p_dropout) for _ in range(K)])
        self.superGather = superatomAttention(fingerprint_dim, p_dropout=p_dropout)
#         self.superGather = nn.ModuleList([superatomAttention(fingerprint_dim, p_dropout) for _ in range(T)])
        # weight for initialize superAtom state 
#         self.sum_importance = torch.nn.Parameter(torch.Tensor(1), requires_grad=True)
#         self.sum_importance.data.fill_(0)
    
        self.predict = nn.Sequential(
            LinearBn(fingerprint_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=p_dropout),
            nn.Linear(512, num_target),
        )

    def forward(self, atom, bond, bond_index, mol_index):
        num_atom, atom_dim = atom.shape 
        num_bond, bond_dim = bond.shape
        bond_index = bond_index.t().contiguous()
        atom = self.atom_preprocess(atom)
        bond = self.bond_preprocess(bond)  # 将边特征转变为节点特征大小  (12,128,128)
        
        neighbor_atom = atom[bond_index[1]]  # 获得所有边的第二个节点的特征 

        mixture = neighbor_atom + bond - neighbor_atom * bond  # 12,1,128
        
        neighbor = torch.cat([bond, neighbor_atom, mixture],-1)
        
        target_atom = atom[bond_index[0]]
        
        feature_align = torch.cat([target_atom, neighbor],dim=-1)
        
        align_score = F.leaky_relu(self.align(self.dropout(feature_align)))

        attention_weight = softmax(align_score, bond_index[0], num=num_atom)

        context = scatter_('add', torch.mul(attention_weight, self.attend(neighbor)), \
                           bond_index[0], dim_size=num_atom)
        
        context = F.elu(context) 

        atom = self.gru(context, atom)
        
        atoms = []
        for k in range(self.K-1):
            atom = self.propagate(atom, bond_index)

        superatom_num = mol_index.max()+1
        superatom = scatter_('add', atom, mol_index, dim_size=superatom_num) # get init superatom by sum
        superatoms = []
        
        for t in range(self.T):
            superatom, attention_weight = self.superGather(superatom, atom, mol_index) 

        predict = self.predict(superatom)
        
        return predict

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)
    

class Fingerprint_viz(torch.nn.Module):
    def __init__(self, num_target, fingerprint_dim, K=3, T=3, p_dropout=0.2, atom_dim=40, bond_dim=10):
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
    label = torch.from_numpy(np.concatenate(label).astype(np.float)).float()
    
    return smiles, atom, bond, bond_index, mol_index, label


class graph_dataset(Dataset):
    def __init__(self, smiles_list, graph_dict):

        self.graph_dict = graph_dict
        self.smiles_list = smiles_list

    def __getitem__(self, x):

        smiles = self.smiles_list[x]

        graph = self.graph_dict[smiles]

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
    net = Fingerprint(2, 32, atom_dim=39, bond_dim=10)
    for b, (smiles, atom, bond, bond_index, mol_index, label) in enumerate(train_loader):
#         print(atom, bond, bond_index, mol_index, label)
        _ = net(atom, bond, bond_index, mol_index)
        break

    print('model success!')
