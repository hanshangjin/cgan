# encoding: utf-8
import numpy as np
import rdkit
import rdkit.Chem.ChemicalFeatures
import rdkit.Chem as Chem
import rdkit.Chem.rdDepictor as rdDepictor
import rdkit.Chem.Draw as Draw
import scipy
import scipy.sparse as sp
import networkx as nx
import utility.global_var as g



np.random.seed(0)

######from DeepCDR############
Max_atoms = g.Max_atoms #100
def NormalizeAdj(adj):
    adj = adj + np.eye(adj.shape[0])
    d = sp.diags(np.power(np.array(adj.sum(1)), -0.5).flatten(), 0).toarray()
    a_norm = adj.dot(d).transpose().dot(d)
    return a_norm

def CalculateGraphFeatPadding(feat_mat,adj_m): #***
    feat = np.zeros((Max_atoms,feat_mat.shape[-1]),dtype='float32')
    adj_mat = np.zeros((Max_atoms,Max_atoms),dtype='float32')
    feat[:feat_mat.shape[0],:] = feat_mat

    adj_ = adj_m
    norm_adj_ = NormalizeAdj(adj_)
    adj_mat[:norm_adj_.shape[0],:norm_adj_.shape[0]] = norm_adj_
    return feat, adj_mat
#----------------------------------------------------------------------
def CalculateGraphFeat(feat_mat,adj_list): #***
    assert feat_mat.shape[0]-1 == len(adj_list)
    feat = np.zeros((Max_atoms,feat_mat.shape[-1]),dtype='float32')
    adj_mat = np.zeros((Max_atoms,Max_atoms),dtype='float32')
    # if israndom:
    #     feat = np.random.rand(Max_atoms,feat_mat.shape[-1])
    #     adj_mat[feat_mat.shape[0]:,feat_mat.shape[0]:] = random_adjacency_matrix(Max_atoms-feat_mat.shape[0])
    feat[:feat_mat.shape[0],:] = feat_mat
    unknown_ohe = feat_mat[-1]
    for j in range(feat_mat.shape[0], Max_atoms):
        feat[j, :] = unknown_ohe

    for i in range(len(adj_list)):
        nodes = adj_list[i]
        for each in nodes:
            adj_mat[i,int(each)] = 1
    assert np.allclose(adj_mat,adj_mat.T)
    adj_ = adj_mat[:len(adj_list),:len(adj_list)]
    adj_2 = adj_mat[len(adj_list):,len(adj_list):]
    norm_adj_ = NormalizeAdj(adj_)
    norm_adj_2 = NormalizeAdj(adj_2)
    adj_mat[:len(adj_list),:len(adj_list)] = norm_adj_
    adj_mat[len(adj_list):,len(adj_list):] = norm_adj_2    
    return feat.tolist(), adj_mat.tolist()
#############################################################



def get_i2element():
    i2element = {1:'H', 5:'B', 6:'C', 7:'N', 8: 'O', 9:'F', 11:'Na', 12:'Mg',
                13:'Al', 14:'Si', 15:'P', 16:'S', 17:'Cl', 19:'K', 20:'Ca', 25:'Mn',
                26:'Fe', 29:'Cu', 30:'Zn', 34:'Se', 35:'Br', 53:'I', 0:'unknown'}
    return i2element


def get_mol(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None: 
        return None
    Chem.Kekulize(mol)
    return mol

# Project: PyBioMed   Author: gadsbyfly   
# File: topology.py    License: BSD 3-Clause "New" or "Revised" 
def CalculateSchiultz(mol):
    """
    #################################################################
    Calculation of Schiultz number

    ---->Tsch(log value)

    Usage:

        result=CalculateSchiultz(mol)

        Input: mol is a molecule object

        Output: result is a numeric value
    #################################################################
    """
    Distance = np.array(Chem.GetDistanceMatrix(mol), "d")
    Adjacent = np.array(Chem.GetAdjacencyMatrix(mol), "d")
    VertexDegree = sum(Adjacent)

    return sum(scipy.dot((Distance + Adjacent), VertexDegree))


# modified from smiles2graph
def mol_graph_feature(smiles, i2element=get_i2element()):
    mol = rdkit.Chem.MolFromSmiles(smiles)
    mol = rdkit.Chem.AddHs(mol)
    num_atoms = len(list(mol.GetAtoms()))
    mol_ohe = np.zeros((num_atoms+1, len(i2element)))
    key_list = list(i2element.keys())
    labels = []
    for a in mol.GetAtoms():
        idx = a.GetAtomicNum()
        key_list_idx = key_list.index(idx)
        labels.append(key_list_idx)
        mol_ohe[a.GetIdx(), key_list_idx] = 1.0
    mol_ohe[num_atoms, key_list.index(0)] = 1.0
    labels.append(key_list.index(0))

    adj = Chem.GetAdjacencyMatrix(mol)

    A = adj
    degreee = sum(adj)
    adj_list = []
    for lst in adj.tolist():
        adjlst = []
        for idx in range(len(lst)):
            if lst[idx]>0:
                adjlst.append(idx)
        adj_list.append(adjlst)

    return mol_ohe, adj_list, degreee.tolist(), labels


def smiles2graph(sml, i2element={}):
    '''Argument for the RD2NX function should be a valid SMILES sequence
    returns: the graph
    '''
    m = rdkit.Chem.MolFromSmiles(sml)
    m = rdkit.Chem.AddHs(m)
    order_string = {rdkit.Chem.rdchem.BondType.SINGLE: 1,
                    rdkit.Chem.rdchem.BondType.DOUBLE: 2,
                    rdkit.Chem.rdchem.BondType.TRIPLE: 3,
                    rdkit.Chem.rdchem.BondType.AROMATIC: 4}
    N = len(list(m.GetAtoms()))
    nodes = np.zeros((N,len(i2element)))
    lookup = list(i2element.keys())

    for a in m.GetAtoms():
        idx = a.GetAtomicNum()
        nodes[a.GetIdx(), idx] = 1

    
    adj = np.zeros((N,N,5)) 
    for j in m.GetBonds():
        u = min(j.GetBeginAtomIdx(),j.GetEndAtomIdx())
        v = max(j.GetBeginAtomIdx(),j.GetEndAtomIdx())        
        order = j.GetBondType()
        if order in order_string:
            order = order_string[order]
        else:
            raise Warning('Ignoring bond order' + order)
        adj[u, v, order] = 1        
        adj[v, u, order] = 1        
    return nodes, adj


if __name__ == '__main__':
    mol_ohe, adj_list, degree, labels = mol_graph_feature('CCN(CC)CC')
    print(degree)
    print(mol_ohe)
    print(adj_list)

    feat, dAd_In = CalculateGraphFeat(mol_ohe, adj_list)
    print(feat)
