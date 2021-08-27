
# encoding: utf-8
import torch
import numpy as np
from torch.autograd import Variable
from mydb.db_sql_database_tablename import MySqliteDatabaseWithTablename
import utility.global_var as g

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def reduce_triu(triu):
    triu = triu[:-1]
    reduced = triu[:int(len(triu)/2)]
    for i in range(reduced.shape[0]):
        # print(triu[triu.shape[0]-i-1, triu.shape[1]-i-1 : triu.shape[1]])
        reduced[i, 0:i+1] = triu[triu.shape[0]-i-1, triu.shape[1]-i-1 : triu.shape[1]]
    return reduced

class ProMolDataset4FuzCav:
	def __init__(self, file):
		super(ProMolDataset4FuzCav, self).__init__()
		self.db = MySqliteDatabaseWithTablename(file)
		self.db.set_tablename('fuzcavpromol')

	def __len__(self):
		size = self.db.get_size()
		return size

	def close(self):
		self.db.close()

	def __getitem__(self, idx):
		# data = {'pro_id': pro_id,
		# 		'pro_emb': pro_emb,
		# 		'mol_smiles': mol_smiles,
		# 		'node_feature': node_feature,
		# 		'adj_mat': adj_mat,
		# 		'node_atom_label': node_atom_label}
		# self.db.set_tablename('fuzcavpromol')
		mol_data = self.db.get_tuple(idx)
		pro_id = mol_data['pro_id']
		pro_emb = torch.Tensor(mol_data['pro_emb'])

		# for GAT
		Max_atoms = g.Max_atoms #100
		node_feature = torch.Tensor(mol_data['node_feature'])
		adj_mat = torch.Tensor(mol_data['adj_mat'])
		labls_A = mol_data['node_atom_label']
		labls_A = labls_A + [22]*(Max_atoms - len(labls_A)) #padding labels
		node_atom_label = torch.Tensor(labls_A).long()
		adj_ohe = np.array(mol_data['adj_ohe'])
		adj_ohe = adj_ohe[:-1, :-1, :]
		adj_ohe = torch.Tensor(reduce_triu(adj_ohe))


		pro_emb = Variable(pro_emb).to(device)
		# for GAT
		node_feature = Variable(node_feature).to(device)
		adj_mat = Variable(adj_mat).to(device)
		node_atom_label = Variable(node_atom_label, requires_grad=False).to(device)
		adj_ohe = Variable(adj_ohe, requires_grad=False).to(device)

		return pro_id, pro_emb, node_feature, adj_mat, node_atom_label, adj_ohe



