import torch
import numpy as np
from rdkit import Chem
from utility.chemutils import mol_graph_feature


with open('./preprocessed_data/pro_fp.txt', 'r') as f:
    lines = f.read().split('\n')
f.close()

nfeat = 23
nhid = 8
nclass = 23
dropout = 0.5
alpha=0.2
nheads=8
enc_out_dim=64
latent_dim_gat=64

# self.pro = []
node_ohe_list = []
mol_emb = []
adj_ohe_list = []
for line in lines:
    name = line.split(' : ')[0]
    name_1 = name.split('_')[0]
    name_2 = name.split('_')[-2] 
    file_path = './scPDB/' + name_1.lower() + '_' + name_2 + '/ligand.sdf'
    mol = Chem.SDMolSupplier(file_path)[0]
    smiles = Chem.MolToSmiles(mol)
    node_ohe, adj_list, _, _ = mol_graph_feature(smiles)
    adj_mat = np.zeros((239, 239),dtype='float32')
    for i in range(len(adj_list)):
        nodes = adj_list[i]
        for each in nodes:
            adj_mat[i,int(each)] = 1
    # adj ohe
    adj_mat_ohe = np.zeros((adj_mat.shape[0], adj_mat.shape[1], 5), dtype='float32')
    for i in range(adj_mat.shape[0]):
        for j in range(adj_mat.shape[1]):
            if adj_mat[i, j] == 1.0:
                adj_mat_ohe[i, j, 0] = 1.0
            elif adj_mat[i, j] == 2.0:
                adj_mat_ohe[i, j, 1] = 1.0
            elif adj_mat[i, j] == 3.0:
                adj_mat_ohe[i, j, 2] = 1.0
            elif adj_mat[i, j] == 1.5:
                adj_mat_ohe[i, j, 3] = 1.0
            elif adj_mat[i, j] == 0.0:
                adj_mat_ohe[i, j, 4] = 1.0
            else:
                print('Error')
    adj_ohe_list.append(adj_mat_ohe)
adj_mat_ohe = torch.Tensor(adj_mat_ohe)
torch.save(adj_ohe_list, './pro_emb/adj_ohe.pt')

