# encoding: utf-8
import torch
import os
import rdkit.Chem as Chem


from mydb.db_sql_database_tablename import MySqliteDatabaseWithTablename
from utility.chemutils import mol_graph_feature, CalculateGraphFeat

def datalist2database(data_list, file_path=None):
	pro_mol_list = data_list
	db = MySqliteDatabaseWithTablename(file_path)

	adj_ohe_list = torch.load('./preprocessed_data/adj_ohe.pt')

	db.set_tablename('fuzcavpromol')
	db.create_table()
	for i in range(len(pro_mol_list)):
		if (i%4 == 0) and i != 0: # To reduce the memory resident during database operation
			db.commit()
		# ++++++++++++++++++++++++
		pro_id, pro_emb, mol_smiles = pro_mol_list[i]

		node_ohe, adj_list, degree, node_atom_label = mol_graph_feature(mol_smiles)
		# normalize and padding: node_ohe-->node_feature, adj_list-->adj_mat
		# normalize and padding are from DeepCDR
		# node_feature, adj_mat are used as input of GAT
		node_feature, adj_mat = CalculateGraphFeat(node_ohe, adj_list) #padding mol_ohe, adj_list

		adj_ohe = adj_ohe_list[i]
		adj_ohe = adj_ohe.tolist()

		data = {'pro_id': pro_id,
				'pro_emb': pro_emb,
				'mol_smiles': mol_smiles,
				'node_feature': node_feature,
				'adj_mat': adj_mat,
				'node_atom_label': node_atom_label,
				'adj_ohe': adj_ohe}
		# ++++++++++++++++++++++++
		db.add_tuple(data)
		pro_mol_list[i] = None # release memory
		adj_ohe_list[i] = None # release memory

	db.close_with_commit()


def data2database(train_list=None, support_list=None, query_list=None, folder=None, file_prefix=None):
	if train_list is not None:
		datalist2database(train_list, file_path = folder + file_prefix + 'train.db')


if __name__ == '__main__':

	with open('./preprocessed_data/pro_fp.txt', 'r') as f:
		lines = f.read().split('\n')
	f.close()


	pro_mol_list = []
	for line in lines:
		# mol
		name = line.split(' : ')[0]
		name_1 = name.split('_')[0]
		name_2 = name.split('_')[-2]
		file_path = '/home/hanshangjin/cgan-upload/scPDB/' + name_1.lower() + '_' + name_2 + '/ligand.sdf'
		try:
			mol = Chem.SDMolSupplier(file_path)[0]
		except Exception as e:
			print(str(e))
			continue
		pro_folder_id = name_1 + '_' + name_2
		smiles = Chem.MolToSmiles(mol)


		# pro
		pro = line.split(' : ')[1]
		pro = pro.split(';')
		pro_emb = list(map(int, pro))
		pro_mol_list.append((pro_folder_id, pro_emb, smiles))


	train_list = pro_mol_list

	data2database(train_list=train_list, folder='./database/', file_prefix='FuzCav_')



