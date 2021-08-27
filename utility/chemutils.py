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



np.random.seed(0)

######拷贝自DeepCDR，用于使分子的特征长度为都Max_atoms###############
Max_atoms = 240 #100
# israndom = False
# def random_adjacency_matrix(n):   
#     matrix = [[random.randint(0, 1) for i in range(n)] for j in range(n)]
#     # No vertex connects to itself
#     for i in range(n):
#         matrix[i][i] = 0
#     # If i is connected to j, j is connected to i
#     for i in range(n):
#         for j in range(n):
#             matrix[j][i] = matrix[i][j]
#     return matrix

# 感觉来自PyGAT的Normalize: D^0.5*A*D^0.5
def NormalizeAdj(adj):
    adj = adj + np.eye(adj.shape[0])
    d = sp.diags(np.power(np.array(adj.sum(1)), -0.5).flatten(), 0).toarray()
    a_norm = adj.dot(d).transpose().dot(d)
    return a_norm

# padding feat_mat为Max_atoms长的feat，
# 把邻接列表adj_list，变成邻接对称矩阵adj_mat, shape为(Max_atoms, Max_atoms)
def CalculateGraphFeatPadding(feat_mat,adj_m): #***
    # assert feat_mat.shape[0] == len(adj_list)
    feat = np.zeros((Max_atoms,feat_mat.shape[-1]),dtype='float32')
    adj_mat = np.zeros((Max_atoms,Max_atoms),dtype='float32')
    # if israndom:
    #     feat = np.random.rand(Max_atoms,feat_mat.shape[-1])
    #     adj_mat[feat_mat.shape[0]:,feat_mat.shape[0]:] = random_adjacency_matrix(Max_atoms-feat_mat.shape[0])
    feat[:feat_mat.shape[0],:] = feat_mat

    # assert np.allclose(adj_mat,adj_mat.T)
    adj_ = adj_m
    norm_adj_ = NormalizeAdj(adj_)
    adj_mat[:norm_adj_.shape[0],:norm_adj_.shape[0]] = norm_adj_
    return feat, adj_mat # adj_mat == D^0.5*A*D^0.5，但没有加I_n
    # adj_mat = np.sum(adj_mat, axis=-1) + np.eye(adj_mat.shape[0]) # adj_mat == D^0.5*A*D^0.5 + I_n
    # return [feat,adj_mat]
#----------------------------------------------------------------------
# padding feat_mat为Max_atoms长的feat，
# 把邻接列表adj_list，变成邻接对称矩阵adj_mat, shape为(Max_atoms, Max_atoms)
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
    return feat.tolist(), adj_mat.tolist() # adj_mat == D^0.5*A*D^0.5，但没有加I_n
    # adj_mat = np.sum(adj_mat, axis=-1) + np.eye(adj_mat.shape[0]) # adj_mat == D^0.5*A*D^0.5 + I_n
    # return [feat,adj_mat]
#############################################################



# element2i = {'H':1, 'B':5, 'C':6, 'N':7, 'O':8, 'F':9, 'Na':11, 'Mg':12,
#             'Al':13, 'Si':14, 'P':15, 'S':16, 'Cl':17, 'K':19, 'Ca':20, 'Mn':25,
#             'Fe':26, 'Cu':29, 'Zn':30, 'Se':34, 'Br':35, 'I':53, 'unknown':0}

# ELEM_LIST = ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca', 'Fe', 'Al', 'I', 'B', 'K', 'Se', 'Zn', 'H', 'Cu', 'Mn', 'unknown']
# ATOM_FDIM = len(ELEM_LIST) + 6 + 5 + 4 + 1
# BOND_FDIM = 5 + 6
# MAX_NB = 6
# ### basic setting from https://github.com/wengong-jin/iclr19-graph2graph/blob/master/fast_jtnn/mpn.py

# 生成分子ohe时用到
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
    mol_ohe = np.zeros((num_atoms+1, len(i2element))) #num_atoms处放无效原子ohe，以利于解码
    key_list = list(i2element.keys()) #把i2element表中得元素序号整型值，按顺序放到key_list
    labels = []
    for a in mol.GetAtoms():
        idx = a.GetAtomicNum() #元素周期表中的序号
        key_list_idx = key_list.index(idx) #按元素序号整型值，到key_list获取列表序号，从0开始
        labels.append(key_list_idx)
        mol_ohe[a.GetIdx(), key_list_idx] = 1.0 #key_list.index(idx)获取的值小于等于22
    mol_ohe[num_atoms, key_list.index(0)] = 1.0 #num_atoms处放无效原子ohe
    labels.append(key_list.index(0))

    adj = Chem.GetAdjacencyMatrix(mol)
    # adj_mat = adj + np.eye(adj.shape[0])
    # adj，每一行对应分子的一个原子序号(注意不是周期表中的元素序号)，
    # 这个序号与分子特征列表中原子的index对应，
    # 与之邻接的原子按列数序号与之邻接,
    # 目前邻接的话，相应的列置1;
    # 把1改成bond值(0.0,1.0,1.5,2.0,3.0)

    A = adj
    degreee = sum(adj)
    # adj_list = [[idx for idx in range(lst) if lst.index(idx)>0] for lst in adj.tolist() ]
    adj_list = []
    for lst in adj.tolist():
        adjlst = []
        for idx in range(len(lst)):
            if lst[idx]>0:
                adjlst.append(idx)
        adj_list.append(adjlst)

    # return mol_ohe.tolist(), adj_mat.tolist(), degreee.tolist()
    # return mol_ohe, adj_mat, degreee #np.array

    # return mol_ohe, adj_list, degreee #np.array
    # return mol_ohe, A, degreee #np.array
    return mol_ohe, adj_list, degreee.tolist(), labels

def mol_graph_feature2(smiles, i2element=get_i2element()):
    mol = rdkit.Chem.MolFromSmiles(smiles)
    mol = rdkit.Chem.AddHs(mol)
    num_atoms = len(list(mol.GetAtoms()))
    mol_ohe = np.zeros((num_atoms+1, len(i2element))) #num_atoms处放无效原子ohe，以利于解码
    key_list = list(i2element.keys()) #把i2element表中得元素序号整型值，按顺序放到key_list
    labels = []
    for a in mol.GetAtoms():
        idx = a.GetAtomicNum() #元素周期表中的序号
        key_list_idx = key_list.index(idx) #按元素序号整型值，到key_list获取列表序号，从0开始
        labels.append(key_list_idx)
        mol_ohe[a.GetIdx(), key_list_idx] = 1.0 #key_list.index(idx)获取的值小于等于22
    mol_ohe[num_atoms, key_list.index(0)] = 1.0 #num_atoms处放无效原子ohe
    labels.append(key_list.index(0))


    for bond in mol.GetBonds():
        G.add_edge(bond.GetBeginAtomIdx(),
                   bond.GetEndAtomIdx(),
                   bond_type=bond.GetBondType())

    adj = Chem.GetAdjacencyMatrix(mol)
    # adj_mat = adj + np.eye(adj.shape[0])
    # adj，每一行对应分子的一个原子序号(注意不是周期表中的元素序号)，
    # 这个序号与分子特征列表中原子的index对应，
    # 与之邻接的原子按列数序号与之邻接,
    # 目前邻接的话，相应的列置1;
    # 把1改成bond值(0.0,1.0,1.5,2.0,3.0)

    A = adj
    degreee = sum(adj)
    # adj_list = [[idx for idx in range(lst) if lst.index(idx)>0] for lst in adj.tolist() ]
    adj_list = []
    for lst in adj.tolist():
        adjlst = []
        for idx in range(len(lst)):
            if lst[idx]>0:
                adjlst.append(idx)
        adj_list.append(adjlst)

    # return mol_ohe.tolist(), adj_mat.tolist(), degreee.tolist()
    # return mol_ohe, adj_mat, degreee #np.array

    # return mol_ohe, adj_list, degreee #np.array
    # return mol_ohe, A, degreee #np.array
    return mol_ohe, adj_list, degreee.tolist(), labels


def smiles2graph(sml, i2element={}): #此函数有错误
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
        # nodes[i.GetIdx(), lookup.index(i.GetAtomicNum())] = 1
        idx = a.GetAtomicNum()
        nodes[a.GetIdx(), idx] = 1

    
    adj = np.zeros((N,N,5)) # 5 num of BondType?
    # adj = np.zeros((N,N,N-1))
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
    # mol = get_mol('CO')
    # mol = Chem.AddHs(mol)
    # SchiultzNumber = CalculateSchiultz(mol)

    # nodes, adj = smiles2graph('CO', i2element=get_i2element())
    # adj_mat = np.sum(adj, axis=-1) + np.eye(adj.shape[0])
    # print(nodes)
    # # print(adj)
    # print(adj_mat)

    # mol_ohe, adj_mat, degree = mol_graph_feature('CO')
    # mol_ohe: 分子中的每个原子的ohe
    # adj_list: 邻接列表，用于下面的CalculateGraphFeat
    # degree：Vertex的度
    mol_ohe, adj_list, degree, labels = mol_graph_feature('CO')
    print(degree)
    print(mol_ohe)
    print(adj_list)

    # feat, DAD = CalculateGraphFeatPadding(mol_ohe, adj_mat)
    # feat：按原子数Max_atoms扩展的mol_ohe
    # dAd_In：按原子数Max_atoms扩展并归一化的邻接矩阵，归一化方法与GCN论文中A的处理方法一样
    feat, dAd_In = CalculateGraphFeat(mol_ohe, adj_list)
    print(feat)
    # print(dAd_In)
    a=1
