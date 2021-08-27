import torch
from torch.autograd import Variable
from rdkit import Chem
from utility.chemutils import mol_graph_feature, CalculateGraphFeat
from layers.model4FuzCav import GAT4CGAN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def mol_emb(opt, mols, adj):
    gat4Cgan = GAT4CGAN(opt, nfeat=opt.feature_dim, nhid=opt.num_heads, 
            nclass=opt.num_class, dropout=opt.dropout, 
            alpha=opt.alpha, nheads=opt.num_heads, 
            enc_out_dim=opt.enc_out_dim, latent_dim=opt.latent_dim).to(device)
    emb = gat4Cgan.encoder(mols, adj)
    return emb
